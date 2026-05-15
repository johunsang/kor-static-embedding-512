"""자동화 파이프라인: 어떤 토크나이저든 → 한국어 Static Embedding 모델 자동 변환.

사용법:
  # 단일 변환
  .venv/bin/python pipeline.py --tokenizer klue/roberta-base --dim 512

  # 배치 비교 (미리 정의된 리스트)
  .venv/bin/python pipeline.py --batch

  # 결과 JSON 저장
  .venv/bin/python pipeline.py --tokenizer klue/bert-base --dim 256 --json results.json
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
from transformers import AutoTokenizer


# -------- 데이터 (캐시) --------
_DATA_CACHE = {}


def get_triplets():
    if "triplets" in _DATA_CACHE:
        return _DATA_CACHE["triplets"]
    rows = []
    for cfg in ("multi_nli", "snli"):
        d = load_dataset("kakaobrain/kor_nli", cfg, split="train")
        rows.extend(d)
    grouped = defaultdict(lambda: {"pos": [], "neg": []})
    for row in rows:
        a, b, lab = row["premise"], row["hypothesis"], row["label"]
        if lab == 0:
            grouped[a]["pos"].append(b)
        elif lab == 2:
            grouped[a]["neg"].append(b)
    triplets = [
        {"anchor": a, "positive": d["pos"][0], "negative": d["neg"][0]}
        for a, d in grouped.items() if d["pos"] and d["neg"]
    ]
    _DATA_CACHE["triplets"] = Dataset.from_list(triplets)
    return _DATA_CACHE["triplets"]


def get_sts_train():
    if "sts_train" in _DATA_CACHE:
        return _DATA_CACHE["sts_train"]
    examples = []
    for x in load_dataset("mteb/KorSTS", split="train"):
        examples.append({"sentence1": x["sentence1"], "sentence2": x["sentence2"],
                         "score": float(x["score"]) / 5.0})
    for x in load_dataset("klue/klue", "sts", split="train"):
        examples.append({"sentence1": x["sentence1"], "sentence2": x["sentence2"],
                         "score": float(x["labels"]["real-label"]) / 5.0})
    _DATA_CACHE["sts_train"] = Dataset.from_list(examples)
    return _DATA_CACHE["sts_train"]


def get_evaluator():
    if "evaluator" in _DATA_CACHE:
        return _DATA_CACHE["evaluator"]
    ds = load_dataset("mteb/KorSTS", split="valid")
    ev = EmbeddingSimilarityEvaluator(
        sentences1=[x["sentence1"] for x in ds],
        sentences2=[x["sentence2"] for x in ds],
        scores=[float(x["score"]) / 5.0 for x in ds],
        main_similarity=SimilarityFunction.COSINE,
        name="korsts-valid",
    )
    _DATA_CACHE["evaluator"] = ev
    return ev


# -------- 평가 --------
def eval_model(model):
    out = {}
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
        cos = (ea * eb).sum(axis=1)
        out[name] = {"pearson": float(pearsonr(cos, gold).statistic),
                     "spearman": float(spearmanr(cos, gold).statistic), "n": len(gold)}

    d = load_dataset("klue/klue", "sts", split="validation")
    s1 = [x["sentence1"] for x in d]
    s2 = [x["sentence2"] for x in d]
    gold = np.array([x["labels"]["real-label"] for x in d], dtype=np.float32)
    ea = model.encode(s1, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    eb = model.encode(s2, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    cos = (ea * eb).sum(axis=1)
    out["KLUE-STS-val"] = {"pearson": float(pearsonr(cos, gold).statistic),
                           "spearman": float(spearmanr(cos, gold).statistic), "n": len(gold)}
    return out


def measure_speed(model, n=200):
    """단일 문장 + batch 처리량."""
    sent = "오늘 날씨가 정말 좋아서 산책하기 딱 좋은 날이네요."
    # 워밍업
    _ = model.encode([sent], normalize_embeddings=True)

    # 단일 지연
    times = []
    for _ in range(50):
        t = time.time()
        _ = model.encode([sent], normalize_embeddings=True)
        times.append((time.time() - t) * 1000)

    # batch=64 처리량
    sents = [sent] * 64
    t = time.time()
    for _ in range(5):
        _ = model.encode(sents, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sps = (64 * 5) / (time.time() - t)

    return {"single_p50_ms": float(np.percentile(times, 50)),
            "batch64_sent_per_sec": float(sps)}


# -------- 파이프라인 --------
def run_pipeline(tokenizer_id, embedding_dim=512, output_dir=None,
                 stage1_batch=1024, stage2_batch=64, verbose=True, seed=42):
    if output_dir is None:
        safe = tokenizer_id.replace("/", "__")
        output_dir = f"./output/auto-{safe}-{embedding_dim}"
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n{'='*60}\n  파이프라인: {tokenizer_id} → static-{embedding_dim}d\n{'='*60}")
        print(f"  device={device}")

    # 1. 모델 초기화
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as e:
        return {"error": f"토크나이저 로드 실패: {e}", "tokenizer": tokenizer_id}
    static_embedding = StaticEmbedding(tokenizer, embedding_dim=embedding_dim)
    model = SentenceTransformer(modules=[static_embedding], device=device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  vocab={tokenizer.vocab_size}, dim={embedding_dim}, params={n_params:,}")

    evaluator = get_evaluator()

    # 2. 학습 전 평가
    pre_score = evaluator(model)
    if verbose:
        print(f"  학습 전 KorSTS-valid Spearman: {pre_score.get('korsts-valid_spearman_cosine'):.4f}")

    # 3. Stage 1: KorNLI MNRL
    if verbose:
        print(f"\n  [Stage 1] KorNLI MNRL (batch={stage1_batch})")
    triplets = get_triplets()
    if verbose:
        print(f"    triplets={len(triplets)}")
    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "stage1"),
        num_train_epochs=1,
        per_device_train_batch_size=stage1_batch,
        learning_rate=2e-1,
        warmup_ratio=0.1,
        bf16=(device == "cuda"),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no", logging_steps=200, report_to=[], seed=seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=triplets,
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    t = time.time()
    trainer.train()
    stage1_time = time.time() - t

    stage1_results = eval_model(model)

    # 4. Stage 2: STS regression
    if verbose:
        print(f"\n  [Stage 2] STS regression (batch={stage2_batch})")
    sts = get_sts_train()
    targs = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(output_dir, "stage2"),
        num_train_epochs=4,
        per_device_train_batch_size=stage2_batch,
        learning_rate=2e-2,
        warmup_ratio=0.1,
        bf16=(device == "cuda"),
        save_strategy="no", logging_steps=100, report_to=[], seed=seed,
        eval_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="eval_korsts-valid_spearman_cosine",
        greater_is_better=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=sts,
        loss=losses.CosineSimilarityLoss(model), evaluator=evaluator,
    )
    t = time.time()
    trainer.train()
    stage2_time = time.time() - t

    # 5. 최종 평가
    final_results = eval_model(model)
    speed = measure_speed(model)
    model.save_pretrained(os.path.join(output_dir, "final"))

    summary = {
        "tokenizer": tokenizer_id,
        "embedding_dim": embedding_dim,
        "n_params": int(n_params),
        "stage1_time_sec": float(stage1_time),
        "stage2_time_sec": float(stage2_time),
        "total_time_sec": float(stage1_time + stage2_time),
        "stage1_results": stage1_results,
        "final_results": final_results,
        "speed": speed,
        "model_path": os.path.join(output_dir, "final"),
    }
    if verbose:
        print(f"\n  [최종 결과] tokenizer={tokenizer_id}")
        for name, r in final_results.items():
            print(f"    {name:<22}  Spearman={r['spearman']:.4f}  Pearson={r['pearson']:.4f}")
        print(f"  추론: single={speed['single_p50_ms']:.2f}ms, batch64={speed['batch64_sent_per_sec']:.0f}/s")
        print(f"  학습 시간: Stage1={stage1_time:.1f}s + Stage2={stage2_time:.1f}s = {stage1_time+stage2_time:.1f}s")

    # 메모리 정리
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# -------- 배치 모드 --------
BATCH_CANDIDATES = [
    # (tokenizer_id, dim, 메모)
    ("klue/roberta-base", 512, "baseline (우리 모델)"),
    ("klue/bert-base", 512, "KLUE BERT"),
    ("klue/roberta-large", 512, "더 큰 vocab?"),
    ("monologg/koelectra-base-v3-discriminator", 512, "KoELECTRA"),
    ("BAAI/bge-m3", 512, "BGE-M3 vocab (다국어)"),
    ("intfloat/multilingual-e5-large", 512, "e5 vocab"),
    ("snunlp/KR-SBERT-V40K-klueNLI-augSTS", 512, "KR-SBERT vocab"),
]


def run_batch(only=None):
    results = []
    for tok, dim, note in BATCH_CANDIDATES:
        if only and tok not in only:
            continue
        try:
            r = run_pipeline(tok, embedding_dim=dim)
            r["note"] = note
            results.append(r)
        except Exception as e:
            print(f"  ❌ {tok} 실패: {e}")
            results.append({"tokenizer": tok, "error": str(e), "note": note})

    # 비교 표
    print(f"\n\n{'='*100}\n  배치 결과 (KorSTS-test Spearman 기준 정렬)\n{'='*100}")
    valid = [r for r in results if "final_results" in r]
    valid.sort(key=lambda r: -r["final_results"]["KorSTS-test"]["spearman"])

    print(f"  {'Tokenizer':<55}{'KorSTS-t':>10}{'KorSTS-v':>10}{'KLUE':>10}{'params':>13}{'time':>8}")
    for r in valid:
        fr = r["final_results"]
        print(f"  {r['tokenizer']:<55}"
              f"{fr['KorSTS-test']['spearman']:>10.4f}"
              f"{fr['KorSTS-valid']['spearman']:>10.4f}"
              f"{fr['KLUE-STS-val']['spearman']:>10.4f}"
              f"{r['n_params']:>13,}"
              f"{r['total_time_sec']:>7.1f}s")

    return results


# -------- CLI --------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", help="HF 토크나이저 ID (예: klue/roberta-base)")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--batch", action="store_true", help="배치 모드 (여러 후보 비교)")
    p.add_argument("--output", default=None)
    p.add_argument("--json", default=None, help="결과를 JSON으로 저장")
    args = p.parse_args()

    if args.batch:
        results = run_batch()
    elif args.tokenizer:
        results = run_pipeline(args.tokenizer, embedding_dim=args.dim, output_dir=args.output)
    else:
        p.print_help()
        exit(1)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.json}")
