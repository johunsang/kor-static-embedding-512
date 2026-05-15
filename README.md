# 한국어 Static Embedding 학습·평가 프로젝트

원본 모델(`kekeappa/kor-minish-bge-m3-ko`)이 한국어 STS에서 낮은 점수(Spearman 0.55)를 보여, **동일한 Static Embedding 아키텍처를 유지하면서 한국어 공개 데이터로 재학습**하여 0.77~0.82까지 끌어올린 프로젝트입니다.

학습된 모델: **https://huggingface.co/kekeappa/kor-static-embedding-512**

## 핵심 결과 한 줄 요약

> **BGE-M3 성능의 92%를 단 3% 자원으로, 158배 빠른 속도로 — 한국어 특화 68MB 모델**

| 지표 | 원본 (kor-minish-bge-m3-ko) | **우리 모델 (kor-static-embedding-512)** | BGE-M3 |
|---|---:|---:|---:|
| KorSTS-test Spearman | 0.5512 | **0.7758** | 0.8026 |
| 평균 STS Spearman | 0.5336 | **0.7708** | 0.8372 |
| 모델 크기 | ~30MB (256d) | **68MB (512d)** | 2,168MB (1024d) |
| 추론 속도 (CPU) | ~10k sent/s | **35,463 sent/s** | 224 sent/s |
| 학습 비용 | — | **약 $0.25** (A100 11분) | — |

## 원본 vs 우리 모델 — 무엇이 달라졌나

| 항목 | 원본 `kor-minish-bge-m3-ko` | **우리 모델 `kor-static-embedding-512`** |
|---|---|---|
| 아키텍처 | StaticEmbedding (model2vec 계열) | **동일** (StaticEmbedding) |
| 차원 | 256 | **512** (표현력 ↑) |
| Base 토크나이저 | (불명 — 일반 한국어) | **klue/roberta-base** (한국어 32K vocab) |
| 학습 데이터 | 미공개 (BGE-M3 distillation 추정) | **KorNLI 277K triplet + KorSTS+KLUE-STS 17K pair** (공개 데이터) |
| 학습 방법 | distillation only? | **2-stage: MNRL contrastive → STS regression** |
| 학습 시간 | — | **약 1분** (A100) |
| KorSTS-test Spearman | 0.5512 | **0.7758** (+0.22) |
| KorSTS-valid Spearman | 0.6601 | **0.8248** (+0.16) |
| KLUE-STS-val Spearman | 0.3894 | **0.7119** (+0.32) |

핵심 개선:
1. **차원 2배 (256→512)** — 표현력 한계 완화
2. **한국어 토크나이저로 교체** — `klue/roberta-base` vocab 사용
3. **공개 한국어 데이터로 직접 supervised 학습** — Stage 1 (NLI contrastive) + Stage 2 (STS regression)
4. **재현 가능한 학습 레시피** — 모든 코드 공개

## 파일 구조

```
.
├── README.md                  # 본 문서
├── train_static_ko.py         # ⭐ 학습 스크립트 (재현 가능)
├── bench_sts.py               # 원본 모델 KorSTS/KLUE-STS 벤치마크
├── compare_bge_m3.py          # BGE-M3 vs 우리 모델 정면 비교
├── speed_bench.py             # 상세 속도 벤치마크 (배치별, 실전 시나리오)
├── demo.py                    # 간단 유사도 데모
├── search_app.py              # CLI 검색 프로그램 (즉시 사용 가능)
└── webapp.py                  # 🌐 웹 테스트 콘솔 (FastAPI + HTML)
```

## 설치

```bash
# 1. 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install sentence-transformers torch datasets scipy numpy scikit-learn

# 3. 웹앱까지 쓸 경우
pip install fastapi uvicorn
```

> 모델 가중치는 자동으로 HuggingFace에서 다운로드됩니다 (~68MB, 캐시 위치: `~/.cache/huggingface/hub/`).

## 테스트 방법 (단계별 가이드)

### 테스트 1 — 간단 유사도 데모 (`demo.py`)

원본 모델(`kekeappa/kor-minish-bge-m3-ko`)을 호출하는 데모입니다.
> ⚠️ 원본은 삭제되어 더 이상 동작하지 않습니다. 우리 모델로 바꿔서 실행해야 합니다 → `MODEL_NAME = "kekeappa/kor-static-embedding-512"` 로 수정.

```bash
.venv/bin/python demo.py
```

**확인 포인트**: 7개 문장의 코사인 유사도 매트릭스, "맑은 하늘" / "인공지능" / "한식" 3개 쿼리의 Top-3 검색 결과.

---

### 테스트 2 — KorSTS / KLUE-STS 공식 벤치마크 (`bench_sts.py`)

KorSTS test/valid, KLUE-STS validation 셋에서 Pearson/Spearman 상관계수를 측정합니다.

```bash
.venv/bin/python bench_sts.py
```

**예상 출력**:
```
KorSTS-test            N= 1376  P=0.5584  S=0.5512  (원본)
KorSTS-valid           N= 1465  P=0.6579  S=0.6601
KLUE-STS-val           N=  519  P=0.4055  S=0.3894
```

스크립트의 `MODEL_NAME`을 `kekeappa/kor-static-embedding-512`로 바꿔 실행하면 우리 모델 점수 확인 가능.

---

### 테스트 3 — BGE-M3와 정면 비교 (`compare_bge_m3.py`)

두 모델의 STS 점수 + 추론 속도를 같은 조건에서 비교합니다.

```bash
.venv/bin/python compare_bge_m3.py
```

**확인 포인트**:
- KorSTS-test/valid, KLUE-STS-val에서의 Pearson/Spearman
- 추론 시간, 초당 처리 문장 수
- 모델 크기, 임베딩 차원 비교

> 첫 실행 시 BGE-M3 가중치 다운로드 (~2.3GB) — 5~10분 소요.

---

### 테스트 4 — 상세 속도 벤치마크 (`speed_bench.py`)

배치 크기별 처리량, 단일 쿼리 지연시간(p50/p95/p99), 실전 인덱싱 시간을 측정합니다.

```bash
.venv/bin/python speed_bench.py
```

**확인 포인트**:
- 모델 로드 시간 (우리 모델 0.3s vs BGE-M3 24s)
- batch=1~512 처리량 (우리 모델 1,132 → 92,468 sent/s)
- 실전 시나리오: 100만 건 인덱싱 시간 (1시간 vs 30초)

---

### 테스트 5 — CLI 검색 프로그램 (`search_app.py`)

요리/프로그래밍/AI/여행 등 다양한 도메인 28개 문서에서 8가지 쿼리로 Top-5 검색하는 데모.

```bash
.venv/bin/python search_app.py
```

**확인 포인트**:
- 동의어/의역 쿼리에도 정확한 검색 ("맛있는 한국 음식" → 김치찌개)
- 짧은 쿼리 처리 ("GPT" → GPT 1위, 점수 0.49)
- **부정문 한계**: "좋아한다" vs "좋아하지 않는다"의 유사도가 단어 매칭으로 인해 0.52로 높게 나옴 (BoW 한계)

---

### 테스트 6 — 🌐 웹 테스트 콘솔 (`webapp.py`)

브라우저에서 5가지 기능을 시각적으로 테스트할 수 있는 FastAPI 웹앱.

```bash
.venv/bin/uvicorn webapp:app --port 8765
# 또는 HTTPS (Chrome HSTS 우회용):
openssl req -x509 -newkey rsa:2048 -keyout /tmp/key.pem -out /tmp/cert.pem -days 1 -nodes -subj "/CN=localhost"
.venv/bin/uvicorn webapp:app --port 8443 --ssl-keyfile /tmp/key.pem --ssl-certfile /tmp/cert.pem
```

브라우저로 **http://localhost:8765** 또는 **https://localhost:8443** 접속.

> Chrome에서 "사이트 보안 연결 안 됨" 경고 시 키보드로 `thisisunsafe` 입력 → 자동 통과.

**제공 기능 5가지**:

| 탭 | 기능 | 시나리오 |
|---|---|---|
| 🔍 의미 검색 | 코퍼스 + 쿼리 → Top-K | 코퍼스 한 줄당 한 문서 입력, 쿼리 + Top-K 지정 |
| 📊 유사도 매트릭스 | 여러 문장 → 코사인 히트맵 | 여러 문장 비교 시각화 |
| 📰 뉴스 요약 | 긴 텍스트 → 추출 요약 N문장 | 알고리즘: 임베딩 평균(중심)과 가까운 N문장 선택 |
| 🗂️ 클러스터링 | 문장 목록 → KMeans 자동 그룹화 | K 지정해서 자동 그룹화 |
| ⚠️ 한계 테스트 | 부정문/어순/다의어/반어 4종 | 미리 준비된 케이스로 약점 확인 |

---

### 테스트 7 — Python 코드로 직접 사용

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 모델 로드 (첫 실행 시 자동 다운로드 ~68MB)
model = SentenceTransformer("kekeappa/kor-static-embedding-512")

# 임베딩 추출
sentences = ["오늘 날씨가 좋네요", "햇살이 따뜻합니다", "비가 많이 옵니다"]
emb = model.encode(sentences, normalize_embeddings=True)

# 유사도
print(emb @ emb.T)
```

검색 시나리오:
```python
corpus = ["김치찌개 만드는 법", "딥러닝 입문 강의", "주말 등산 코스"]
corpus_emb = model.encode(corpus, normalize_embeddings=True)

query_emb = model.encode(["인공지능 학습"], normalize_embeddings=True)
scores = (query_emb @ corpus_emb.T)[0]
print(corpus[scores.argmax()])  # → 딥러닝 입문 강의
```

---

### 테스트 8 — 재학습 (GPU 필요)

`train_static_ko.py`로 모델을 처음부터 재학습할 수 있습니다.

```bash
# RunPod / Colab / 로컬 GPU
python train_static_ko.py \
  --base-tokenizer klue/roberta-base \
  --embedding-dim 512 \
  --output-dir ./output/kor-static-512
```

- A100 80GB 기준 학습 시간 **약 1분** (Stage 1: 25초, Stage 2: 18초)
- 비용 (RunPod A100 PCIe $1.39/hr) **약 $0.25**

## 학습 레시피 상세

### Stage 1: KorNLI MultipleNegativesRankingLoss
- 데이터: `kakaobrain/kor_nli` (multi_nli 393K + snli 550K)
- entailment를 positive, contradiction을 hard negative → **277,826 triplet**
- Loss: `MultipleNegativesRankingLoss`
- batch=2048, lr=2e-1, epoch=1

### Stage 2: STS regression
- 데이터: KorSTS-train (5,691) + KLUE-STS-train (11,668) = 17,359 pairs
- Loss: `CosineSimilarityLoss`
- batch=64, lr=2e-2, epoch=4
- best checkpoint: KorSTS-valid Spearman 기준 선택

| 단계 | KorSTS-test S | KorSTS-valid S | KLUE-STS-val S |
|---|---:|---:|---:|
| 학습 전 (랜덤 init) | — | 0.5885 | — |
| Stage 1 종료 | 0.7519 | 0.7983 | 0.5757 |
| **Stage 2 종료 (최종)** | **0.7758** | **0.8248** | **0.7119** |

→ Stage 2 (STS regression)가 특히 KLUE-STS-val 점수를 0.58 → 0.71로 크게 끌어올림.

## 다른 언어에서 호출하기 (JavaScript / Rust / 기타)

Static Embedding 모델은 구조가 단순(토크나이저 + lookup + mean)해서 **ONNX 변환이 매우 깔끔**합니다.

### 옵션 A: ONNX 변환 후 직접 호출 (오프라인/엣지)

```python
# python: ONNX 변환
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("kekeappa/kor-static-embedding-512")
model.save_to_hub("kekeappa/kor-static-embedding-512", create_pr=False, exist_ok=True)
# 또는 직접 변환 (별도 스크립트 export_onnx.py 참조)
```

#### JavaScript (브라우저 / Node.js)
[transformers.js](https://github.com/xenova/transformers.js) — HuggingFace 공식 JS 포트
```javascript
import { pipeline } from "@xenova/transformers";
const extractor = await pipeline("feature-extraction", "kekeappa/kor-static-embedding-512");
const emb = await extractor("한국어 임베딩 테스트", { pooling: "mean", normalize: true });
console.log(emb.data);  // Float32Array (512차원)
```

#### Rust
[ort (ONNX Runtime)](https://github.com/pykeio/ort) 또는 [candle](https://github.com/huggingface/candle)
```rust
use ort::{Environment, Session, SessionBuilder};
let env = Environment::builder().build()?;
let session = SessionBuilder::new(&env)?.with_model_from_file("model.onnx")?;
// 추론 — tokenizer crate로 토큰화 + ort로 forward
```

### 옵션 B: Python 서버 + REST API (가장 간단)

`webapp.py`처럼 FastAPI로 띄워두고 어디서든 HTTP 호출:
```bash
curl -X POST http://localhost:8765/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"인공지능","corpus":["GPT 언어모델","파이썬 기초"],"top_k":2}'
```

→ **언어 무관** (JS/Rust/Go/Java/Swift 모두 가능)

### 옵션별 권장 시나리오

| 시나리오 | 권장 옵션 |
|---|---|
| 브라우저 온디바이스 추론 | A — transformers.js (모델 다운로드 후 클라이언트 추론) |
| Rust 임베디드 / WASM | A — ort 또는 candle + ONNX |
| 모바일 (Flutter/React Native) | A — onnxruntime-mobile |
| 다언어 마이크로서비스 | B — FastAPI REST/gRPC |
| Java/Kotlin 서버 | A — ONNX Runtime Java |

> ONNX export 스크립트와 JS/Rust 데모는 별도 디렉토리(`examples/js/`, `examples/rust/`)에 추가 예정.

## 알려진 한계 (Static Embedding 본질)

| 한계 | 원인 | 권장 대안 |
|---|---|---|
| 어순 무시 ("철수가 영희를" ≈ "영희가 철수를") | mean pooling은 순서 정보 손실 | BGE-M3 |
| 다의어 처리 약함 ("은행 직원" ≈ "강변 은행") | 같은 토큰 → 같은 벡터 | BGE-M3 |
| 부정문 ("좋아한다" ≈ "좋아하지 않는다") | 단어 겹침이 cosine 끌어올림 | BGE-M3, 또는 NLI 모델 |
| 반어/풍자 | 표면 단어와 의미 불일치 | LLM rerank |

웹앱의 "⚠️ 한계 테스트" 탭에서 실제 점수를 확인할 수 있습니다.

## 적합한 용도

✅ **권장**
- 대규모 RAG 1차 retrieval (수백만 문서를 빠르게 좁히기)
- 의미 기반 검색, FAQ 매칭, 추천
- 클러스터링, 중복 제거, 카테고리 분류
- 온디바이스 / 모바일 / 서버리스 한국어 임베딩
- **2-stage 검색**: kor-static-512(1차 후보 추출) + BGE-M3(2차 재정렬)

❌ **부적합**
- 어순·문맥 미세 차이가 중요한 작업
- 다국어 검색 (한국어 전용)
- KLUE 같은 뉴스 도메인 절대 최고 성능 (BGE-M3 권장)

## 인용 / 참고

- [HuggingFace Static Embeddings 블로그 (Tom Aarsen)](https://huggingface.co/blog/static-embeddings)
- [MinishLab model2vec](https://github.com/MinishLab/model2vec)
- KorSTS / KorNLI: [KakaoBrain KorNLUDatasets](https://github.com/kakaobrain/KorNLUDatasets)
- [KLUE Benchmark](https://klue-benchmark.com)

## 라이선스

Apache 2.0
