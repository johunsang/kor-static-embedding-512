"""세부 속도 벤치마크: 단일 쿼리 지연시간, 배치별 처리량, 실전 시나리오."""

import gc
import time

import numpy as np
from sentence_transformers import SentenceTransformer

SENT = "오늘 날씨가 정말 좋고 햇살이 따뜻해서 산책하기 딱 좋은 날이네요."

MODELS = [
    ("kor-static-512", "models/kor-static-512"),
    ("BAAI/bge-m3", "BAAI/bge-m3"),
]


def bench(label, path):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    # 로드 시간
    gc.collect()
    t = time.time()
    m = SentenceTransformer(path)
    load_t = time.time() - t
    print(f"  로드 시간: {load_t*1000:.1f}ms")

    # 워밍업
    _ = m.encode([SENT], normalize_embeddings=True)

    # 단일 쿼리 지연시간 (50회 평균)
    times = []
    for _ in range(50):
        t = time.time()
        _ = m.encode([SENT], normalize_embeddings=True)
        times.append((time.time() - t) * 1000)
    single_p50 = np.percentile(times, 50)
    single_p95 = np.percentile(times, 95)
    single_p99 = np.percentile(times, 99)
    print(f"  단일 쿼리 지연: p50={single_p50:.2f}ms  p95={single_p95:.2f}ms  p99={single_p99:.2f}ms")

    # 배치별 처리량
    print("  배치별 처리량 (CPU):")
    throughput = {}
    for batch_size in [1, 8, 32, 128, 512]:
        sents = [SENT] * batch_size
        # 워밍업
        for _ in range(2):
            _ = m.encode(sents, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=False)
        # 측정
        t = time.time()
        n_iter = max(1, 100 // batch_size)
        for _ in range(n_iter):
            _ = m.encode(sents, batch_size=batch_size, normalize_embeddings=True,
                         show_progress_bar=False)
        total_t = time.time() - t
        total_sents = batch_size * n_iter
        sps = total_sents / total_t
        throughput[batch_size] = sps
        print(f"    batch={batch_size:>4}: {sps:>10.1f} sent/s")

    return {
        "load_ms": load_t * 1000,
        "single_p50_ms": single_p50,
        "single_p95_ms": single_p95,
        "throughput": throughput,
    }


results = {}
for label, path in MODELS:
    results[label] = bench(label, path)

# 비교 출력
print(f"\n\n{'='*70}\n  비교 (BGE-M3 = 100%)\n{'='*70}")
ours = results["kor-static-512"]
bge = results["BAAI/bge-m3"]

print(f"\n  [모델 로드 시간 — 낮을수록 좋음]")
print(f"    BGE-M3:        {bge['load_ms']:>10.1f}ms  (100%)")
print(f"    kor-static:    {ours['load_ms']:>10.1f}ms  ({ours['load_ms']/bge['load_ms']*100:.3f}%)")
print(f"    → 우리 모델이 {bge['load_ms']/ours['load_ms']:.0f}배 빠름")

print(f"\n  [단일 쿼리 지연 p50 — 낮을수록 좋음]")
print(f"    BGE-M3:        {bge['single_p50_ms']:>10.2f}ms  (100%)")
print(f"    kor-static:    {ours['single_p50_ms']:>10.2f}ms  ({ours['single_p50_ms']/bge['single_p50_ms']*100:.2f}%)")
print(f"    → 우리 모델이 {bge['single_p50_ms']/ours['single_p50_ms']:.1f}배 빠름")

print(f"\n  [배치 처리량 — 높을수록 좋음]")
print(f"    {'batch':>6}  {'BGE-M3':>14}  {'kor-static':>14}  {'비율':>10}")
for b in [1, 8, 32, 128, 512]:
    bv, ov = bge["throughput"][b], ours["throughput"][b]
    print(f"    {b:>6}  {bv:>10.1f}/s  {ov:>10.1f}/s  {ov/bv*100:>8.0f}%  ({ov/bv:.1f}×)")

# 실전 시나리오
print(f"\n  [실전 시나리오 — 인덱싱·검색 시간 환산]")
avg_ours = np.mean(list(ours["throughput"].values()))
avg_bge = np.mean(list(bge["throughput"].values()))

for n_docs, label in [(10_000, "1만건"), (100_000, "10만건"), (1_000_000, "100만건"), (10_000_000, "1천만건")]:
    t_bge = n_docs / avg_bge
    t_ours = n_docs / avg_ours
    def fmt(s):
        if s < 60: return f"{s:.1f}초"
        if s < 3600: return f"{s/60:.1f}분"
        if s < 86400: return f"{s/3600:.1f}시간"
        return f"{s/86400:.1f}일"
    print(f"    {label:>8} 인덱싱: BGE-M3 {fmt(t_bge):>10}  /  kor-static {fmt(t_ours):>10}  ({t_ours/t_bge*100:.2f}%)")
