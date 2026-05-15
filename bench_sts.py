"""한국어 공식 STS 벤치마크: KorSTS, KLUE-STS.

평가지표:
  - Spearman / Pearson 상관계수 (cosine similarity vs. gold score)
"""

import time

import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL_NAME = "kekeappa/kor-minish-bge-m3-ko"

print(f"[모델 로드] {MODEL_NAME}")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"  완료 ({time.time() - t0:.2f}s, dim={model.get_embedding_dimension()})\n")


def evaluate_sts(pairs_a, pairs_b, gold, name):
    print(f"[{name}] N = {len(gold)}")
    t = time.time()
    emb_a = model.encode(pairs_a, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_b = model.encode(pairs_b, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    # 행별 cosine
    cos = (emb_a * emb_b).sum(axis=1)
    elapsed = time.time() - t
    pearson = pearsonr(cos, gold).statistic
    spearman = spearmanr(cos, gold).statistic
    print(f"  Pearson  = {pearson:.4f}")
    print(f"  Spearman = {spearman:.4f}")
    print(f"  추론시간 = {elapsed:.2f}s ({len(gold) / elapsed:.1f} pairs/s)\n")
    return pearson, spearman


results = {}

# -------- KorSTS (KakaoBrain, MTEB mirror) --------
print("=" * 60)
print(" KorSTS (KakaoBrain) - test split")
print("=" * 60)
ds = load_dataset("mteb/KorSTS", split="test")
s1 = [x["sentence1"] for x in ds]
s2 = [x["sentence2"] for x in ds]
gold = np.array([x["score"] for x in ds], dtype=np.float32)
results["KorSTS-test"] = evaluate_sts(s1, s2, gold, "KorSTS-test")

print("=" * 60)
print(" KorSTS (KakaoBrain) - valid split")
print("=" * 60)
ds = load_dataset("mteb/KorSTS", split="valid")
s1 = [x["sentence1"] for x in ds]
s2 = [x["sentence2"] for x in ds]
gold = np.array([x["score"] for x in ds], dtype=np.float32)
results["KorSTS-valid"] = evaluate_sts(s1, s2, gold, "KorSTS-valid")

# -------- KLUE-STS --------
print("=" * 60)
print(" KLUE-STS (validation, 공식 dev 셋)")
print("=" * 60)
ds = load_dataset("klue/klue", "sts", split="validation")
s1 = [x["sentence1"] for x in ds]
s2 = [x["sentence2"] for x in ds]
gold = np.array([x["labels"]["real-label"] for x in ds], dtype=np.float32)
results["KLUE-STS-val"] = evaluate_sts(s1, s2, gold, "KLUE-STS-validation")

# -------- 요약 --------
print("=" * 60)
print(" 요약 (Spearman 기준 — 한국어 STS 표준 메트릭)")
print("=" * 60)
print(f"  {'Benchmark':<22}  {'Pearson':>8}  {'Spearman':>9}")
for name, (p, s) in results.items():
    print(f"  {name:<22}  {p:>8.4f}  {s:>9.4f}")
