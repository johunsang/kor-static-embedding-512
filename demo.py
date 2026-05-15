"""kor-minish-bge-m3-ko 한국어 문장 유사도 데모."""

import time

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL_NAME = "kekeappa/kor-minish-bge-m3-ko"

print(f"[1] 모델 로드: {MODEL_NAME}")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"    로드 완료 ({time.time() - t0:.2f}s)")
print(f"    임베딩 차원: {model.get_sentence_embedding_dimension()}")
print(f"    max_seq_length: {model.max_seq_length}\n")

sentences = [
    "오늘 날씨가 정말 좋네요.",
    "햇살이 따뜻하고 기분 좋은 하루입니다.",
    "비가 와서 우산을 챙겨야 합니다.",
    "파이썬으로 머신러닝 모델을 학습시키고 있어요.",
    "딥러닝 프레임워크로 신경망을 훈련 중입니다.",
    "김치찌개에 라면 사리를 넣어 먹었다.",
    "점심으로 매콤한 찌개와 면을 곁들였다.",
]

print("[2] 문장 임베딩 추출")
t0 = time.time()
embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"    완료 ({time.time() - t0:.2f}s, shape={embeddings.shape})\n")

print("[3] 코사인 유사도 매트릭스")
sim = cos_sim(embeddings, embeddings).numpy()

col_w = 6
header = " " * 5 + "".join(f"S{i+1:<{col_w-1}}" for i in range(len(sentences)))
print(header)
for i, row in enumerate(sim):
    cells = "".join(f"{v:.2f} " for v in row)
    print(f"S{i+1}: {cells}")

print("\n[4] 문장 인덱스")
for i, s in enumerate(sentences):
    print(f"  S{i+1}: {s}")

print("\n[5] 검색 데모: 쿼리 → 가장 유사한 문장 Top-3")
queries = [
    "맑은 하늘과 햇볕",
    "인공지능 모델 훈련",
    "매운 한식 식사",
]
query_emb = model.encode(queries, normalize_embeddings=True)
q_sim = cos_sim(query_emb, embeddings).numpy()

for q, scores in zip(queries, q_sim):
    print(f"\n  쿼리: {q}")
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"    {rank}. [{score:.4f}] {sentences[idx]}")
