"""실전 검색 프로그램 — kor-static-embedding-512 동작 확인.

용도: 한국어 문장 코퍼스에서 자연어 쿼리로 의미 검색.
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

print("[로딩] HuggingFace에서 모델 다운로드 중...")
t0 = time.time()
model = SentenceTransformer("kekeappa/kor-static-embedding-512")
print(f"  완료 ({(time.time()-t0)*1000:.0f}ms, dim={model.get_embedding_dimension()})\n")

# 다양한 도메인의 한국어 문서 코퍼스
corpus = [
    # 요리
    "김치찌개는 신김치와 돼지고기를 넣고 끓이는 한국 전통 음식이다.",
    "된장찌개 만드는 법: 멸치 육수에 된장을 풀고 채소를 넣어 끓인다.",
    "비빔밥은 밥 위에 다양한 나물과 고추장을 넣어 비벼 먹는 요리다.",
    "라면 끓일 때 계란을 넣으면 더 맛있어진다.",
    "파스타는 면을 삶고 소스와 함께 볶아서 만든다.",
    # 프로그래밍
    "파이썬은 배우기 쉽고 활용도가 높은 프로그래밍 언어다.",
    "자바스크립트는 웹 브라우저에서 동작하는 스크립트 언어다.",
    "리액트는 페이스북이 만든 사용자 인터페이스 라이브러리다.",
    "도커는 컨테이너 기반 가상화 플랫폼이다.",
    "쿠버네티스는 컨테이너 오케스트레이션 도구다.",
    "git은 소스코드 버전 관리 시스템이다.",
    # AI / ML
    "딥러닝은 다층 신경망을 사용하는 머신러닝 기법이다.",
    "트랜스포머는 어텐션 메커니즘 기반의 신경망 구조다.",
    "GPT는 OpenAI가 개발한 대규모 언어 모델이다.",
    "BERT는 양방향 인코더를 사용하는 언어 모델이다.",
    "RAG는 검색 증강 생성으로 외부 지식을 활용한다.",
    # 여행
    "제주도는 한국 남쪽에 위치한 화산섬으로 관광지로 유명하다.",
    "부산은 한국 제2의 도시이며 해운대 해수욕장이 유명하다.",
    "경주는 신라 시대 유적이 많이 남아 있는 역사 도시다.",
    "강릉은 동해안에 위치하며 커피 거리로 유명하다.",
    # 건강
    "유산소 운동은 심혈관 건강에 좋다.",
    "근력 운동은 근육량을 늘리고 기초대사량을 높인다.",
    "충분한 수면은 면역력 강화에 필수적이다.",
    "스트레스를 줄이기 위해 명상이나 요가를 하면 좋다.",
    # 날씨
    "오늘은 햇살이 따뜻하고 맑은 날씨다.",
    "장마철에는 우산을 항상 챙겨야 한다.",
    "겨울에는 두꺼운 외투와 목도리가 필수다.",
    "황사가 심한 날은 마스크 착용을 권장한다.",
]

print(f"[인덱싱] {len(corpus)}개 문서 임베딩 생성")
t0 = time.time()
corpus_emb = model.encode(corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
print(f"  완료 ({(time.time()-t0)*1000:.1f}ms = {len(corpus)/(time.time()-t0):.0f} doc/s)\n")


def search(query, top_k=5):
    """쿼리 → 코퍼스에서 Top-K 검색."""
    t = time.time()
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (q_emb @ corpus_emb.T)[0]
    top_idx = np.argsort(-scores)[:top_k]
    elapsed = (time.time() - t) * 1000
    print(f"  쿼리: \"{query}\"  ({elapsed:.2f}ms)")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    {rank}. [{scores[idx]:.4f}] {corpus[idx]}")
    print()


# 다양한 시나리오 테스트
print("=" * 70)
print(" Top-5 검색 테스트")
print("=" * 70 + "\n")

queries = [
    # 직접 매칭 — 단어가 코퍼스에 그대로 있음
    "맛있는 한국 음식 만드는 법",
    # 동의어/의역 — 단어는 다르지만 의미 같음
    "인공지능 모델 구조",
    # 추론 필요 — 단어 매칭 거의 없음
    "건강하게 살려면 어떻게 해야 할까",
    # 도메인 특화
    "코드 버전 관리는 어떻게 하지?",
    "여행하기 좋은 한국 도시 추천",
    # 어려운 케이스
    "비가 올 때 준비물",
    # 다른 언어/외래어
    "Python 머신러닝 라이브러리",
    # 짧은 쿼리
    "GPT",
]
for q in queries:
    search(q)


# === 부정문 vs 긍정문 한계 확인 ===
print("=" * 70)
print(" 한계 확인: 부정문 처리 (Static Embedding 약점)")
print("=" * 70 + "\n")

probe = [
    "나는 김치찌개를 좋아한다",
    "나는 김치찌개를 좋아하지 않는다",
    "나는 김치찌개를 싫어한다",
]
embs = model.encode(probe, normalize_embeddings=True)
sim = embs @ embs.T
print("  문장:")
for i, s in enumerate(probe):
    print(f"    S{i+1}: {s}")
print("\n  유사도 매트릭스:")
print(f"            S1      S2      S3")
for i, row in enumerate(sim):
    print(f"    S{i+1}: " + "  ".join(f"{v:.4f}" for v in row))
print()
print("  → S1(좋아함) vs S2(좋아하지 않음): 단어가 거의 같아 높은 유사도가 나옴")
print("  → S2(좋아하지 않음) vs S3(싫어함): 의미는 같지만 단어가 달라 낮을 수 있음")
