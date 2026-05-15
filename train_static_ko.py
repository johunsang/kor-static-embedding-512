"""한국어 특화 Static Embedding 모델 학습.

- 아키텍처: sentence_transformers.models.StaticEmbedding (model2vec / minish 계열)
- Base 토크나이저: klue/roberta-base (한국어 vocab 32K)
- 차원: 512 (고정)
- 데이터: KorNLI (triplet) + KorSTS-train + KLUE-STS-train
- Loss:
    Stage 1 — MultipleNegativesRankingLoss on KorNLI (anchor/pos/neg)
    Stage 2 — CosineSimilarityLoss on KorSTS+KLUE-STS

목표: KorSTS-test Spearman 0.78~0.80, 모델 크기 ~60MB, CPU 추론 10만+ pairs/s
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer


# -------- 설정 --------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-tokenizer", default="klue/roberta-base")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--output-dir", default="/workspace/output/kor-static-512")
    p.add_argument("--stage1-batch", type=int, default=2048)
    p.add_argument("--stage1-lr", type=float, default=2e-1)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage2-batch", type=int, default=64)
    p.add_argument("--stage2-lr", type=float, default=2e-2)
    p.add_argument("--stage2-epochs", type=int, default=4)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -------- 데이터 로드 --------
def build_kornli_triplets():
    """KorNLI (kakaobrain): multi_nli + snli → anchor/positive/negative triplet.

    label: 0=entailment(positive), 1=neutral(skip), 2=contradiction(hard negative).
    """
    print("[데이터] KorNLI 로딩 (multi_nli + snli)...")
    rows = []
    for cfg in ("multi_nli", "snli"):
        d = load_dataset("kakaobrain/kor_nli", cfg, split="train")
        print(f"  {cfg}: {len(d)} rows")
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
        # anchor당 (positive, negative) 쌍을 최대 1개만 — 균형 + 메모리
        triplets.append({
            "anchor": anchor,
            "positive": d["pos"][0],
            "negative": d["neg"][0],
        })

    print(f"  triplets: {len(triplets)}")
    return Dataset.from_list(triplets)


def build_sts_regression():
    """KorSTS-train + KLUE-STS-train → (s1, s2, score[0,1])."""
    print("[데이터] KorSTS + KLUE-STS 로딩...")
    examples = []

    kor = load_dataset("mteb/KorSTS", split="train")
    for x in kor:
        examples.append({
            "sentence1": x["sentence1"],
            "sentence2": x["sentence2"],
            "score": float(x["score"]) / 5.0,
        })

    klue = load_dataset("klue/klue", "sts", split="train")
    for x in klue:
        examples.append({
            "sentence1": x["sentence1"],
            "sentence2": x["sentence2"],
            "score": float(x["labels"]["real-label"]) / 5.0,
        })

    print(f"  pairs: {len(examples)}")
    return Dataset.from_list(examples)


# -------- Evaluator --------
def build_evaluator(name="korsts-valid"):
    ds = load_dataset("mteb/KorSTS", split="valid")
    return EmbeddingSimilarityEvaluator(
        sentences1=[x["sentence1"] for x in ds],
        sentences2=[x["sentence2"] for x in ds],
        scores=[float(x["score"]) / 5.0 for x in ds],
        main_similarity=SimilarityFunction.COSINE,
        name=name,
    )


def final_test(model):
    print("\n" + "=" * 60)
    print(" 최종 테스트")
    print("=" * 60)
    results = {}

    for ds_name, split, label_fn in [
        ("KorSTS-test", "test", lambda x: x["score"]),
        ("KorSTS-valid", "valid", lambda x: x["score"]),
    ]:
        ds = load_dataset("mteb/KorSTS", split=split)
        s1 = [x["sentence1"] for x in ds]
        s2 = [x["sentence2"] for x in ds]
        gold = np.array([label_fn(x) for x in ds], dtype=np.float32)
        ea = model.encode(s1, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        eb = model.encode(s2, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        cos = (ea * eb).sum(axis=1)
        p, s = pearsonr(cos, gold).statistic, spearmanr(cos, gold).statistic
        results[ds_name] = (p, s)
        print(f"  {ds_name:<22} N={len(gold):>5}  P={p:.4f}  S={s:.4f}")

    ds = load_dataset("klue/klue", "sts", split="validation")
    s1 = [x["sentence1"] for x in ds]
    s2 = [x["sentence2"] for x in ds]
    gold = np.array([x["labels"]["real-label"] for x in ds], dtype=np.float32)
    ea = model.encode(s1, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    eb = model.encode(s2, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    cos = (ea * eb).sum(axis=1)
    p, s = pearsonr(cos, gold).statistic, spearmanr(cos, gold).statistic
    results["KLUE-STS-val"] = (p, s)
    print(f"  {'KLUE-STS-val':<22} N={len(gold):>5}  P={p:.4f}  S={s:.4f}")

    return results


# -------- 메인 --------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    print(f"\n[모델 초기화] StaticEmbedding(tokenizer={args.base_tokenizer}, dim={args.embedding_dim})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    static_embedding = StaticEmbedding(tokenizer, embedding_dim=args.embedding_dim)
    model = SentenceTransformer(modules=[static_embedding], device=device)
    print(f"  vocab size: {tokenizer.vocab_size}")
    print(f"  파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    evaluator = build_evaluator()

    print("\n[초기 평가 — 학습 전]")
    init_score = evaluator(model)
    print(f"  KorSTS-valid Spearman: {init_score.get('korsts-valid_spearman_cosine', 'n/a')}")

    # ====== Stage 1: KorNLI MultipleNegativesRankingLoss ======
    if not args.skip_stage1:
        print("\n" + "=" * 60)
        print(" Stage 1: KorNLI MultipleNegativesRankingLoss")
        print("=" * 60)
        train_ds = build_kornli_triplets()
        loss = losses.MultipleNegativesRankingLoss(model)

        targs = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(args.output_dir, "stage1"),
            num_train_epochs=args.stage1_epochs,
            per_device_train_batch_size=args.stage1_batch,
            learning_rate=args.stage1_lr,
            warmup_ratio=0.1,
            fp16=False,  # StaticEmbedding은 fp32가 안정적
            bf16=(device == "cuda"),
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            run_name="ko-static-stage1",
            load_best_model_at_end=True,
            metric_for_best_model="eval_korsts-valid_spearman_cosine",
            greater_is_better=True,
            report_to=[],
            seed=args.seed,
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator,
        )
        t0 = time.time()
        trainer.train()
        print(f"  Stage 1 학습 시간: {(time.time() - t0)/60:.1f}분")
        model.save_pretrained(os.path.join(args.output_dir, "stage1_final"))
        print(f"  → {args.output_dir}/stage1_final 저장")
        print("\n[Stage 1 종료 후 평가]")
        final_test(model)

    # ====== Stage 2: KorSTS+KLUE-STS CosineSimilarityLoss ======
    if not args.skip_stage2:
        print("\n" + "=" * 60)
        print(" Stage 2: KorSTS + KLUE-STS CosineSimilarityLoss")
        print("=" * 60)
        train_ds = build_sts_regression()
        loss = losses.CosineSimilarityLoss(model)

        targs = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(args.output_dir, "stage2"),
            num_train_epochs=args.stage2_epochs,
            per_device_train_batch_size=args.stage2_batch,
            learning_rate=args.stage2_lr,
            warmup_ratio=0.1,
            fp16=False,
            bf16=(device == "cuda"),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            logging_steps=50,
            run_name="ko-static-stage2",
            load_best_model_at_end=True,
            metric_for_best_model="eval_korsts-valid_spearman_cosine",
            greater_is_better=True,
            report_to=[],
            seed=args.seed,
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=targs, train_dataset=train_ds, loss=loss, evaluator=evaluator,
        )
        t0 = time.time()
        trainer.train()
        print(f"  Stage 2 학습 시간: {(time.time() - t0)/60:.1f}분")

    # 최종 모델 저장
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    print(f"\n[최종 모델 저장] {final_dir}")

    # 최종 테스트
    results = final_test(model)

    # 결과 파일로 저장
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Base tokenizer: {args.base_tokenizer}\n")
        f.write(f"Embedding dim: {args.embedding_dim}\n")
        f.write(f"Params: {sum(p.numel() for p in model.parameters()):,}\n\n")
        for name, (p, s) in results.items():
            f.write(f"{name}: Pearson={p:.4f}, Spearman={s:.4f}\n")
    print(f"\n[결과 저장] {args.output_dir}/results.txt")


if __name__ == "__main__":
    main()
