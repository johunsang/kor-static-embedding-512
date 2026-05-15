"""세계 최고 한국어 초소형 Static Embedding 학습.

4-stage 파이프라인:
  Stage 0: KoSimCSE-roberta-multitask teacher로 vocab 토큰 distillation 초기화
  Stage 1: KorNLI (multi_nli + snli) MultipleNegativesRankingLoss
  Stage 2: OPUS 한-영 parallel cross-lingual MNRL (다국어 호환 + 데이터 증강)
  Stage 3: KorSTS + KLUE-STS + 번역된 영어 STS-B Matryoshka regression
            64/128/256/512 차원 동시 지원

목표: KorSTS-test Spearman 0.83+ (512d), 0.75+ (64d 초경량)
"""

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.training_args import BatchSamplers
from transformers import AutoTokenizer, AutoModel


# -------- 설정 --------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="BM-K/KoSimCSE-roberta-multitask")
    p.add_argument("--base-tokenizer", default="klue/roberta-base")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--matryoshka-dims", default="64,128,256,512")
    p.add_argument("--output-dir", default="/workspace/output/kor-static-v2")
    p.add_argument("--xling-pairs", type=int, default=200_000, help="OPUS cross-lingual pair 수")
    p.add_argument("--translated-sts-path", default="/workspace/data/translated_sts.json",
                   help="번역된 STS 데이터(번역 단계가 따로 만든 파일)")
    p.add_argument("--skip-stage0", action="store_true")
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -------- Stage 0: Distillation 초기화 --------
@torch.no_grad()
def distill_init(static_embedding, tokenizer, teacher_id, device, embedding_dim):
    """teacher 모델로 vocab의 각 토큰별 임베딩 추출 → PCA → StaticEmbedding 초기화.

    각 토큰을 단독으로 teacher에 입력하고, 결과 토큰 임베딩 (또는 CLS pooled)을 그 토큰의 representation으로 사용.
    """
    print(f"[Stage 0] Distillation 초기화 (teacher={teacher_id})")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    teacher_model = AutoModel.from_pretrained(teacher_id).to(device).eval()

    vocab_size = tokenizer.vocab_size
    teacher_dim = teacher_model.config.hidden_size
    print(f"  vocab={vocab_size}, teacher_dim={teacher_dim}, target_dim={embedding_dim}")

    # 각 vocab 토큰의 teacher embedding 추출
    # 토큰 자체를 입력해 teacher의 hidden state 평균을 쓰는 것이 model2vec 방식
    embeddings = np.zeros((vocab_size, teacher_dim), dtype=np.float32)
    batch_size = 256
    tokens = [tokenizer.decode([i]) for i in range(vocab_size)]

    for start in range(0, vocab_size, batch_size):
        batch = tokens[start:start + batch_size]
        # 빈 토큰 / 특수 토큰은 그대로 둠 (zero vector)
        valid = [(i, t) for i, t in enumerate(batch) if t and not t.startswith("["+":") and len(t.strip()) > 0]
        if not valid:
            continue
        valid_texts = [t for _, t in valid]
        enc = teacher_tokenizer(valid_texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=8).to(device)
        out = teacher_model(**enc, output_hidden_states=False)
        # mean pooling over tokens (excluding padding)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = pooled.cpu().numpy()
        for (li, _), vec in zip(valid, pooled):
            embeddings[start + li] = vec
        if start % (batch_size * 20) == 0:
            print(f"    progress: {start}/{vocab_size}")

    # PCA로 차원 축소
    from sklearn.decomposition import PCA
    print(f"  PCA: {teacher_dim} → {embedding_dim}")
    pca = PCA(n_components=embedding_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)
    print(f"  설명 분산 비율: {pca.explained_variance_ratio_.sum():.4f}")

    # Zipf weighting (단어 빈도가 낮은 토큰은 약하게)
    # 간단히 토큰 인덱스 기반 1/log(rank+2) — model2vec 원본은 corpus freq 사용
    ranks = np.arange(vocab_size) + 1
    zipf_w = 1.0 / np.log(ranks + 1)
    zipf_w = zipf_w / zipf_w.max()
    reduced = reduced * zipf_w[:, None]

    # StaticEmbedding 레이어에 주입
    static_embedding.embedding.weight.data = torch.tensor(reduced, dtype=torch.float32)
    print(f"  Stage 0 완료. 초기화 가중치 주입")

    # teacher 메모리 해제
    del teacher_model, teacher_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------- 데이터 --------
def build_kornli_triplets():
    print("[데이터] KorNLI (multi_nli + snli)...")
    rows = []
    for cfg in ("multi_nli", "snli"):
        d = load_dataset("kakaobrain/kor_nli", cfg, split="train")
        print(f"  {cfg}: {len(d)}")
        rows.extend(d)
    grouped = defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        a, b, lab = row["premise"], row["hypothesis"], row["label"]
        if lab == 0:
            grouped[a]["pos"].append(b)
        elif lab == 2:
            grouped[a]["neg"].append(b)
    triplets = []
    for anchor, d in grouped.items():
        if not d["pos"] or not d["neg"]:
            continue
        triplets.append({"anchor": anchor, "positive": d["pos"][0], "negative": d["neg"][0]})
    print(f"  triplets: {len(triplets)}")
    return Dataset.from_list(triplets)


def build_xlingual_pairs(n=200_000):
    """OPUS-100 ko-en parallel pair."""
    print(f"[데이터] OPUS-100 ko-en parallel...")
    try:
        d = load_dataset("Helsinki-NLP/opus-100", "en-ko", split="train")
    except Exception:
        d = load_dataset("opus100", "en-ko", split="train")
    print(f"  raw: {len(d)}")
    pairs = []
    for i, row in enumerate(d):
        if i >= n:
            break
        ko = row["translation"]["ko"].strip()
        en = row["translation"]["en"].strip()
        if len(ko) < 3 or len(en) < 3:
            continue
        pairs.append({"sentence1": ko, "sentence2": en})
    print(f"  pairs: {len(pairs)}")
    return Dataset.from_list(pairs)


def build_sts_regression(translated_path=None):
    print("[데이터] STS regression (KorSTS + KLUE-STS + 번역 STS)...")
    examples = []
    for x in load_dataset("mteb/KorSTS", split="train"):
        examples.append({"sentence1": x["sentence1"], "sentence2": x["sentence2"],
                         "score": float(x["score"]) / 5.0})
    for x in load_dataset("klue/klue", "sts", split="train"):
        examples.append({"sentence1": x["sentence1"], "sentence2": x["sentence2"],
                         "score": float(x["labels"]["real-label"]) / 5.0})
    if translated_path and os.path.exists(translated_path):
        with open(translated_path) as f:
            translated = json.load(f)
        for x in translated:
            examples.append({"sentence1": x["sentence1"], "sentence2": x["sentence2"],
                             "score": float(x["score"]) / 5.0})
        print(f"  + 번역 STS: {len(translated)}")
    print(f"  총: {len(examples)}")
    return Dataset.from_list(examples)


# -------- Evaluator --------
def build_evaluator():
    ds = load_dataset("mteb/KorSTS", split="valid")
    return EmbeddingSimilarityEvaluator(
        sentences1=[x["sentence1"] for x in ds],
        sentences2=[x["sentence2"] for x in ds],
        scores=[float(x["score"]) / 5.0 for x in ds],
        main_similarity=SimilarityFunction.COSINE, name="korsts-valid",
    )


def final_eval(model, dims):
    """KorSTS-test/valid + KLUE-STS-val 평가 + 각 Matryoshka 차원별 평가."""
    print("\n" + "=" * 60)
    print(" 최종 평가")
    print("=" * 60)
    results = {}

    for ds_id, name, split, getter in [
        ("mteb/KorSTS", "KorSTS-test", "test", lambda x: x["score"]),
        ("mteb/KorSTS", "KorSTS-valid", "valid", lambda x: x["score"]),
    ]:
        d = load_dataset(ds_id, split=split)
        s1 = [x["sentence1"] for x in d]
        s2 = [x["sentence2"] for x in d]
        gold = np.array([getter(x) for x in d], dtype=np.float32)
        ea = model.encode(s1, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        eb = model.encode(s2, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        results[name] = _per_dim(ea, eb, gold, dims, name)

    d = load_dataset("klue/klue", "sts", split="validation")
    s1 = [x["sentence1"] for x in d]
    s2 = [x["sentence2"] for x in d]
    gold = np.array([x["labels"]["real-label"] for x in d], dtype=np.float32)
    ea = model.encode(s1, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    eb = model.encode(s2, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    results["KLUE-STS-val"] = _per_dim(ea, eb, gold, dims, "KLUE-STS-val")

    return results


def _per_dim(ea, eb, gold, dims, label):
    out = {}
    for dim in dims + [ea.shape[1]]:
        if dim > ea.shape[1]:
            continue
        # truncate + renormalize
        a = ea[:, :dim]
        b = eb[:, :dim]
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        cos = (a * b).sum(axis=1)
        p = pearsonr(cos, gold).statistic
        s = spearmanr(cos, gold).statistic
        out[f"{dim}d"] = {"pearson": float(p), "spearman": float(s)}
        print(f"  {label:<14} dim={dim:>4}  P={p:.4f}  S={s:.4f}")
    return out


# -------- 메인 --------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    matryoshka_dims = [int(d) for d in args.matryoshka_dims.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[Matryoshka dims] {matryoshka_dims}")

    # 모델 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    static_embedding = StaticEmbedding(tokenizer, embedding_dim=args.embedding_dim)
    model = SentenceTransformer(modules=[static_embedding], device=device)
    print(f"\n[모델] StaticEmbedding(tokenizer={args.base_tokenizer}, dim={args.embedding_dim})")
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    evaluator = build_evaluator()
    print(f"  학습 전 KorSTS-valid: {evaluator(model)['korsts-valid_spearman_cosine']:.4f}")

    # ====== Stage 0: Distillation init ======
    if not args.skip_stage0:
        distill_init(static_embedding, tokenizer, args.teacher, device, args.embedding_dim)
        print(f"  Stage 0 후 KorSTS-valid: {evaluator(model)['korsts-valid_spearman_cosine']:.4f}")

    # ====== Stage 1: KorNLI MNRL ======
    if not args.skip_stage1:
        print("\n" + "=" * 60)
        print(" Stage 1: KorNLI MNRL")
        print("=" * 60)
        train_ds = build_kornli_triplets()
        targs = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(args.output_dir, "stage1"),
            num_train_epochs=1, per_device_train_batch_size=2048,
            learning_rate=2e-1, warmup_ratio=0.1,
            bf16=(device == "cuda"), batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no", logging_steps=50, report_to=[], seed=args.seed,
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=targs, train_dataset=train_ds,
            loss=losses.MultipleNegativesRankingLoss(model),
        )
        t = time.time()
        trainer.train()
        print(f"  Stage 1 시간: {(time.time()-t)/60:.2f}분")
        print(f"  Stage 1 후 KorSTS-valid: {evaluator(model)['korsts-valid_spearman_cosine']:.4f}")

    # ====== Stage 2: Cross-lingual parallel ======
    if not args.skip_stage2:
        print("\n" + "=" * 60)
        print(" Stage 2: Cross-lingual OPUS ko-en MNRL")
        print("=" * 60)
        train_ds = build_xlingual_pairs(n=args.xling_pairs)
        targs = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(args.output_dir, "stage2"),
            num_train_epochs=1, per_device_train_batch_size=1024,
            learning_rate=5e-2, warmup_ratio=0.1,
            bf16=(device == "cuda"), batch_sampler=BatchSamplers.NO_DUPLICATES,
            save_strategy="no", logging_steps=50, report_to=[], seed=args.seed,
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=targs, train_dataset=train_ds,
            loss=losses.MultipleNegativesRankingLoss(model),
        )
        t = time.time()
        trainer.train()
        print(f"  Stage 2 시간: {(time.time()-t)/60:.2f}분")
        print(f"  Stage 2 후 KorSTS-valid: {evaluator(model)['korsts-valid_spearman_cosine']:.4f}")

    # ====== Stage 3: Matryoshka KorSTS regression ======
    if not args.skip_stage3:
        print("\n" + "=" * 60)
        print(" Stage 3: Matryoshka KorSTS + KLUE-STS + 번역 STS regression")
        print("=" * 60)
        train_ds = build_sts_regression(args.translated_sts_path)
        base_loss = losses.CosineSimilarityLoss(model)
        matryoshka_loss = losses.MatryoshkaLoss(
            model, base_loss, matryoshka_dims=matryoshka_dims,
        )
        targs = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(args.output_dir, "stage3"),
            num_train_epochs=5, per_device_train_batch_size=64,
            learning_rate=2e-2, warmup_ratio=0.1,
            bf16=(device == "cuda"),
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            logging_steps=50, report_to=[], seed=args.seed,
            load_best_model_at_end=True,
            metric_for_best_model="eval_korsts-valid_spearman_cosine",
            greater_is_better=True,
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=targs, train_dataset=train_ds,
            loss=matryoshka_loss, evaluator=evaluator,
        )
        t = time.time()
        trainer.train()
        print(f"  Stage 3 시간: {(time.time()-t)/60:.2f}분")

    # 저장
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"\n[저장] {final_dir}")

    # 평가
    results = final_eval(model, matryoshka_dims)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"matryoshka_dims": matryoshka_dims, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[결과 JSON] {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
