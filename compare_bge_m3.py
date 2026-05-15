"""우리 모델(kor-static-512) vs BAAI/bge-m3 정면 비교.

같은 데이터셋(KorSTS-test/valid, KLUE-STS-val)에 동일 평가지표 적용.
추론 속도/모델 크기까지 비교.
"""

import os
import time

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

MODELS = [
    ("kor-static-512 (ours)", "models/kor-static-512"),
    ("BAAI/bge-m3", "BAAI/bge-m3"),
]

# 평가용 데이터 한 번만 로드
print("[데이터 로드]")
datasets_to_eval = {}
for split in ("test", "valid"):
    ds = load_dataset("mteb/KorSTS", split=split)
    datasets_to_eval[f"KorSTS-{split}"] = (
        [x["sentence1"] for x in ds],
        [x["sentence2"] for x in ds],
        np.array([x["score"] for x in ds], dtype=np.float32),
    )
ds = load_dataset("klue/klue", "sts", split="validation")
datasets_to_eval["KLUE-STS-val"] = (
    [x["sentence1"] for x in ds],
    [x["sentence2"] for x in ds],
    np.array([x["labels"]["real-label"] for x in ds], dtype=np.float32),
)
for name, (s1, s2, g) in datasets_to_eval.items():
    print(f"  {name}: N={len(g)}")


def eval_model(label, path):
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    t0 = time.time()
    model = SentenceTransformer(path)
    load_t = time.time() - t0
    dim = model.get_embedding_dimension()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  로드시간: {load_t:.2f}s  /  dim={dim}  /  params={n_params:,}")

    results = {}
    for name, (s1, s2, gold) in datasets_to_eval.items():
        t = time.time()
        ea = model.encode(s1, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        eb = model.encode(s2, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        encode_t = time.time() - t
        cos = (ea * eb).sum(axis=1)
        p = pearsonr(cos, gold).statistic
        s = spearmanr(cos, gold).statistic
        n_pairs = len(gold)
        results[name] = {
            "pearson": p, "spearman": s,
            "encode_time": encode_t,
            "pairs_per_sec": (n_pairs * 2) / encode_t,
        }
        print(f"  {name:<22}  P={p:.4f}  S={s:.4f}  "
              f"({encode_t:.2f}s, {(n_pairs*2)/encode_t:.0f} sents/s)")
    return {"dim": dim, "params": n_params, "load_time": load_t, "results": results}


all_results = {}
for label, path in MODELS:
    all_results[label] = eval_model(label, path)


# -------- 비교 표 --------
print(f"\n\n{'=' * 70}\n  최종 비교 (Spearman)\n{'=' * 70}")
benches = list(datasets_to_eval.keys())
print(f"  {'Model':<28}  " + "  ".join(f"{b:<14}" for b in benches))
for label in all_results:
    r = all_results[label]["results"]
    print(f"  {label:<28}  " + "  ".join(f"{r[b]['spearman']:.4f}{'':<8}" for b in benches))

print(f"\n{'=' * 70}\n  모델 크기·속도\n{'=' * 70}")
print(f"  {'Model':<28}  {'dim':>5}  {'params':>15}  {'load':>7}  {'sents/s (avg)':>14}")
for label in all_results:
    a = all_results[label]
    avg_speed = np.mean([r["pairs_per_sec"] for r in a["results"].values()])
    print(f"  {label:<28}  {a['dim']:>5}  {a['params']:>15,}  {a['load_time']:>6.2f}s  {avg_speed:>14.0f}")
