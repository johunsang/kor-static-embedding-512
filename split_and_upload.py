"""Matryoshka로 학습된 512d 모델을 64/128/256으로 잘라서 4개 별도 모델로 HF 업로드.

각 차원별로:
  - StaticEmbedding의 weight를 [:, :dim] 으로 truncate
  - 새 SentenceTransformer로 저장
  - kekeappa/kor-static-embedding-{dim} repo에 업로드
"""

import os
import shutil
import time

import numpy as np
import torch
from huggingface_hub import HfApi, create_repo
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from transformers import AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("환경변수 HF_TOKEN 필요: export HF_TOKEN=hf_xxx")
BASE_MODEL_PATH = "models-v2/kor-static-512"
DIMS = [64, 128, 256]  # 512는 이미 업로드됨

# 점수 (results.json에서)
SCORES = {
    64:  {"korsts_test_s": 0.7337, "korsts_test_p": 0.7382,
          "korsts_valid_s": 0.7885, "klue_s": 0.6582, "size_mb": 9},
    128: {"korsts_test_s": 0.7521, "korsts_test_p": 0.7569,
          "korsts_valid_s": 0.8082, "klue_s": 0.6656, "size_mb": 17},
    256: {"korsts_test_s": 0.7690, "korsts_test_p": 0.7738,
          "korsts_valid_s": 0.8234, "klue_s": 0.6838, "size_mb": 34},
    512: {"korsts_test_s": 0.7718, "korsts_test_p": 0.7760,
          "korsts_valid_s": 0.8330, "klue_s": 0.7033, "size_mb": 68},
}


def make_model_card(dim, scores):
    return f"""---
language:
- ko
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- static-embedding
- model2vec
- korean
- ko
- matryoshka
datasets:
- kakaobrain/kor_nli
- mteb/KorSTS
- klue/klue
- Helsinki-NLP/opus-100
base_model: klue/roberta-base
---

# kor-static-embedding-{dim}

한국어 특화 **초경량 Static Embedding** 모델 — **{scores['size_mb']}MB**, **{dim}차원**.

[kekeappa/kor-static-embedding-512](https://huggingface.co/kekeappa/kor-static-embedding-512)를 Matryoshka 학습으로 만들고 **{dim}차원으로 잘라낸 변종**입니다. 같은 모델 패밀리에 4개 차원 존재 — 용도에 맞게 선택:

| 차원 | 크기 | 용도 |
|---:|---:|---|
| **[64](https://huggingface.co/kekeappa/kor-static-embedding-64)** | 9MB | 🌐 브라우저 · 모바일 · 엣지 |
| **[128](https://huggingface.co/kekeappa/kor-static-embedding-128)** | 17MB | ⚡ 가벼운 검색·분류 |
| **[256](https://huggingface.co/kekeappa/kor-static-embedding-256)** | 34MB | ⚖️ 가성비 |
| **[512](https://huggingface.co/kekeappa/kor-static-embedding-512)** | 68MB | 🎯 최고 정확도 |

## 성능 (KorSTS / KLUE-STS)

| 벤치마크 | Pearson | **Spearman** |
|---|---:|---:|
| KorSTS-test | {scores['korsts_test_p']:.4f} | **{scores['korsts_test_s']:.4f}** |
| KorSTS-valid | — | **{scores['korsts_valid_s']:.4f}** |
| KLUE-STS-val | — | **{scores['klue_s']:.4f}** |

## 사용

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("kekeappa/kor-static-embedding-{dim}")
emb = model.encode(["한국어 문장", "임베딩 테스트"], normalize_embeddings=True)
print(emb.shape)  # (2, {dim})
```

## 특징

- **아키텍처**: StaticEmbedding (model2vec 계열) — 트랜스포머 attention 없음
- **추론**: CPU 최적, GPU 불필요
- **속도**: 단일 쿼리 < 1ms (브라우저에서도 빠름)
- **한영 호환**: cross-lingual 학습됨 — 한국어 쿼리로 영어 문서 검색 가능

## 학습 방법

4-stage 학습:
1. **Distillation 초기화**: `BM-K/KoSimCSE-roberta-multitask` teacher의 vocab 임베딩 → PCA + Zipf weighting
2. **KorNLI MNRL**: `kakaobrain/kor_nli` (multi_nli + snli) 277K triplet
3. **Cross-lingual MNRL**: OPUS-100 ko-en parallel 200K pair
4. **Matryoshka regression**: KorSTS + KLUE-STS + NLLB로 번역한 영어 STS-B
   - 64/128/256/512 차원 동시 최적화 (`MatryoshkaLoss`)

학습 코드: https://github.com/johunsang/kor-static-embedding-512

## 라이선스

Apache 2.0
"""


def main():
    print(f"[기본 모델 로드] {BASE_MODEL_PATH}")
    base_model = SentenceTransformer(BASE_MODEL_PATH)
    base_static = base_model[0]
    full_weight = base_static.embedding.weight.data  # [vocab, 512]
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    print(f"  weight shape: {tuple(full_weight.shape)}")

    api = HfApi(token=HF_TOKEN)

    for dim in DIMS:
        print(f"\n{'='*60}\n  차원 {dim} 생성\n{'='*60}")
        repo_id = f"kekeappa/kor-static-embedding-{dim}"
        out_dir = f"models-v2/kor-static-{dim}"

        # 1. 가중치 truncate
        new_static = StaticEmbedding(tokenizer, embedding_dim=dim)
        truncated = full_weight[:, :dim].clone().detach()
        new_static.embedding.weight.data = truncated
        new_model = SentenceTransformer(modules=[new_static])

        # 2. 로컬 저장
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        new_model.save_pretrained(out_dir)
        size_mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))) / 1024 / 1024
        print(f"  저장: {out_dir}, embedding shape: {tuple(new_static.embedding.weight.shape)}")

        # 3. README 작성
        readme_path = os.path.join(out_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(make_model_card(dim, SCORES[dim]))

        # 4. 동작 확인 (로컬 로드 후 임베딩)
        check = SentenceTransformer(out_dir)
        emb = check.encode(["테스트"], normalize_embeddings=True)
        print(f"  동작 확인: shape={emb.shape}")
        assert emb.shape[1] == dim

        # 5. HF repo 생성 + 업로드
        print(f"  HF repo 생성/업로드: {repo_id}")
        create_repo(repo_id=repo_id, token=HF_TOKEN, repo_type="model",
                    private=False, exist_ok=True)
        api.upload_folder(
            folder_path=out_dir, repo_id=repo_id, repo_type="model",
            commit_message=f"Initial: kor-static-embedding-{dim} (Matryoshka 분리, {SCORES[dim]['size_mb']}MB)",
        )
        print(f"  ✅ https://huggingface.co/{repo_id}")

    # 512 README도 동일한 패밀리 표로 업데이트
    print(f"\n{'='*60}\n  512 README 업데이트\n{'='*60}")
    readme_512 = make_model_card(512, SCORES[512])
    api.upload_file(
        path_or_fileobj=readme_512.encode("utf-8"),
        path_in_repo="README.md",
        repo_id="kekeappa/kor-static-embedding-512",
        repo_type="model",
        commit_message="Add Matryoshka family table (64/128/256/512)",
    )
    print("  ✅ 512 README 업데이트 완료")


if __name__ == "__main__":
    main()
