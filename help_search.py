"""실시간 CS 도움말 검색 데모 (다중 모델 선택).

사용법:
  .venv/bin/uvicorn help_search:app --port 8765
  → http://localhost:8765 (또는 https://localhost:8443 자가서명)
"""

import os
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# -------- 사용 가능한 모델들 --------
MODELS = {
    "kor-static-v2": {
        "path": "models-v2/kor-static-512",
        "label": "kor-static-512 v2 (NEW, 4-stage 학습)",
        "size_mb": 68, "dim": 512, "type": "Static",
        "fallback": "kekeappa/kor-static-embedding-512",
    },
    "kor-static-v1": {
        "path": "kekeappa/kor-static-embedding-512",
        "label": "kor-static-512 v1 (HF, 2-stage)",
        "size_mb": 68, "dim": 512, "type": "Static",
    },
    "kr-sbert": {
        "path": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "label": "KR-SBERT (한국어 SBERT, 트랜스포머)",
        "size_mb": 440, "dim": 768, "type": "Transformer",
    },
    "multilingual-e5": {
        "path": "intfloat/multilingual-e5-small",
        "label": "multilingual-e5-small (다국어 트랜스포머)",
        "size_mb": 470, "dim": 384, "type": "Transformer",
    },
}

# -------- 모델 lazy 캐시 --------
_MODEL_CACHE = {}
_EMB_CACHE = {}  # (model_key, "cat" | "faq") → embeddings


def get_model(key):
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], MODELS[key]["label"]
    info = MODELS[key]
    try:
        m = SentenceTransformer(info["path"])
        _MODEL_CACHE[key] = m
        return m, info["label"]
    except Exception as e:
        if "fallback" in info:
            print(f"  fallback to {info['fallback']}: {e}")
            m = SentenceTransformer(info["fallback"])
            _MODEL_CACHE[key] = m
            return m, info["label"] + " (fallback)"
        raise


# -------- 카테고리 + FAQ --------
CATEGORIES = {
    "결제/환불": [
        "결제했는데 카드가 두 번 빠졌어요",
        "환불받고 싶은데 어떻게 하나요",
        "포인트로 결제하고 싶어요",
        "주문 취소하고 환불받고 싶어요",
        "할인쿠폰이 적용 안 됐어요",
    ],
    "배송 문의": [
        "물건이 아직 안 왔어요",
        "배송이 너무 늦어요",
        "배송 추적은 어떻게 하나요",
        "주소를 잘못 입력했어요",
        "배송지 변경 가능한가요",
    ],
    "상품/재고": [
        "이 상품 재입고 되나요",
        "사이즈 정보 어디서 봐요",
        "다른 색상도 있나요",
        "품질이 마음에 안 들어요 교환 가능한가요",
        "상품 상세 정보 더 보고 싶어요",
    ],
    "계정/로그인": [
        "비밀번호를 잊어버렸어요",
        "로그인이 안 돼요",
        "회원가입 어떻게 하나요",
        "이메일 변경하고 싶어요",
        "계정 탈퇴는 어떻게 하나요",
    ],
    "기술 지원": [
        "앱이 자꾸 꺼져요",
        "결제창이 안 떠요",
        "이미지가 안 보여요",
        "사이트가 느려요",
        "검색이 안 돼요",
    ],
    "쿠폰/이벤트": [
        "이벤트 참여하고 싶어요",
        "쿠폰은 언제 발급되나요",
        "신규 가입 쿠폰 어디 있나요",
        "할인 이벤트 언제 끝나요",
        "친구 추천 혜택이 뭐예요",
    ],
}

FAQ = [
    {"category": "결제/환불", "q": "결제 두 번 됐을 때", "a": "마이페이지 > 주문내역에서 결제 확인 후 1:1 문의로 환불 요청해주세요. 영업일 기준 3~5일 내 처리됩니다."},
    {"category": "결제/환불", "q": "환불 방법", "a": "주문완료 후 7일 이내에 마이페이지 > 주문내역 > 환불요청 버튼으로 신청 가능합니다."},
    {"category": "결제/환불", "q": "쿠폰 적용 안 됨", "a": "쿠폰은 결제 직전 단계에서 '쿠폰 사용' 체크 후 적용됩니다. 최소 주문 금액과 유효기간을 확인하세요."},
    {"category": "배송 문의", "q": "배송 지연", "a": "주문 후 영업일 기준 2~3일 내 출고됩니다. 출고 후 1~2일 내 도착 예정이며, 지연 시 SMS로 안내드립니다."},
    {"category": "배송 문의", "q": "배송 추적", "a": "마이페이지 > 주문내역 > 배송조회 버튼을 누르면 실시간 위치 확인 가능합니다."},
    {"category": "배송 문의", "q": "배송지 변경", "a": "출고 전이라면 마이페이지에서 직접 변경 가능합니다. 출고 후에는 택배사로 직접 연락해주세요."},
    {"category": "상품/재고", "q": "재입고 알림", "a": "품절 상품 페이지의 '재입고 알림 신청' 버튼을 눌러주세요. 입고 즉시 SMS/이메일로 알려드립니다."},
    {"category": "상품/재고", "q": "교환 절차", "a": "상품 수령 후 7일 이내, 사용 흔적이 없는 상태에서 교환 가능합니다. 마이페이지에서 신청해주세요."},
    {"category": "계정/로그인", "q": "비밀번호 재설정", "a": "로그인 화면의 '비밀번호 찾기'를 누르고 가입 이메일을 입력하면 재설정 링크가 발송됩니다."},
    {"category": "계정/로그인", "q": "회원 탈퇴", "a": "마이페이지 > 설정 > 회원탈퇴에서 진행 가능합니다. 탈퇴 후 30일 내 재가입은 동일 이메일로 불가합니다."},
    {"category": "기술 지원", "q": "앱 강제 종료", "a": "앱을 최신 버전으로 업데이트 후 캐시 삭제(설정 > 앱 관리)를 시도해보세요. 그래도 안 되면 재설치 권장."},
    {"category": "기술 지원", "q": "이미지 안 보임", "a": "Wi-Fi/데이터 연결 상태를 확인해주세요. 광고 차단 앱 사용 시 일부 이미지가 안 보일 수 있습니다."},
    {"category": "쿠폰/이벤트", "q": "신규 가입 쿠폰", "a": "가입 즉시 15% 할인 쿠폰이 마이페이지 > 쿠폰함에 자동 발급됩니다. 30일 내 사용해주세요."},
    {"category": "쿠폰/이벤트", "q": "진행중인 이벤트", "a": "메인 페이지 하단 '이벤트' 배너에서 모든 진행 이벤트를 확인할 수 있습니다."},
]

CAT_NAMES = list(CATEGORIES.keys())


def get_embeddings(model_key):
    if model_key in _EMB_CACHE:
        return _EMB_CACHE[model_key]
    model, _ = get_model(model_key)

    cat_emb_list = []
    for name in CAT_NAMES:
        emb = model.encode(CATEGORIES[name], normalize_embeddings=True)
        centroid = emb.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        cat_emb_list.append(centroid)
    cat_emb = np.array(cat_emb_list)

    faq_text = [f"{x['category']} {x['q']}" for x in FAQ]
    faq_emb = model.encode(faq_text, normalize_embeddings=True)

    _EMB_CACHE[model_key] = (cat_emb, faq_emb)
    return cat_emb, faq_emb


# 첫 모델 (v2) 미리 로드
print("[기본 모델 로딩...]")
t0 = time.time()
get_model("kor-static-v2")
get_embeddings("kor-static-v2")
print(f"[준비 완료] {(time.time()-t0)*1000:.0f}ms")


# -------- API --------
class QueryReq(BaseModel):
    text: str
    model_key: str = "kor-static-v2"
    top_k_faq: int = 3
    threshold: float = 0.25


@app.get("/api/models")
def list_models():
    return [
        {"key": k, **{kk: vv for kk, vv in v.items() if kk != "path"}}
        for k, v in MODELS.items()
    ]


@app.post("/api/classify")
def classify(req: QueryReq):
    t = time.time()
    model, model_label = get_model(req.model_key)
    cat_emb, faq_emb = get_embeddings(req.model_key)

    q_emb = model.encode([req.text], normalize_embeddings=True)[0]

    cat_scores = cat_emb @ q_emb
    cat_ranking = sorted(zip(CAT_NAMES, cat_scores), key=lambda x: -x[1])
    top_cat = cat_ranking[0]
    confidence = float(top_cat[1])

    faq_scores = faq_emb @ q_emb
    top_faq_idx = np.argsort(-faq_scores)[: req.top_k_faq]
    faqs = [
        {
            "category": FAQ[i]["category"],
            "q": FAQ[i]["q"],
            "a": FAQ[i]["a"],
            "score": float(faq_scores[i]),
        }
        for i in top_faq_idx
    ]

    elapsed_ms = (time.time() - t) * 1000
    return {
        "elapsed_ms": elapsed_ms,
        "model_label": model_label,
        "model_key": req.model_key,
        "category": top_cat[0] if confidence >= req.threshold else "분류 어려움 (상담사 연결)",
        "confidence": confidence,
        "category_ranking": [{"name": n, "score": float(s)} for n, s in cat_ranking],
        "top_faqs": faqs,
        "low_confidence": confidence < req.threshold,
    }


# -------- HTML --------
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>실시간 CS 도움말 검색 — 다중 모델 비교</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #f5f5f7; color: #1d1d1f; }
  h1 { font-size: 26px; margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }
  .controls { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
  select { padding: 10px 12px; font-size: 14px; border-radius: 8px; border: 1px solid #ccc;
           background: white; cursor: pointer; min-width: 280px; }
  .model-info { font-size: 12px; color: #666; }
  .badge { display: inline-block; padding: 2px 8px; background: #eee; border-radius: 8px;
           font-size: 11px; margin-left: 6px; }
  .input-box { background: white; padding: 20px; border-radius: 12px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 16px; }
  input[type=text] { width: 100%; padding: 14px; font-size: 16px;
                     border: 2px solid #ddd; border-radius: 10px; }
  input[type=text]:focus { border-color: #0066cc; outline: none; }
  .hint { color: #888; font-size: 12px; margin-top: 8px; }
  .preset { display: flex; gap: 6px; margin-top: 12px; flex-wrap: wrap; }
  .preset-btn { padding: 6px 12px; background: #eee; border: 1px solid #ddd;
                border-radius: 16px; font-size: 13px; cursor: pointer; }
  .preset-btn:hover { background: #ddd; }
  .result { background: white; padding: 20px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px; }
  .category-badge { display: inline-block; padding: 6px 14px;
                    background: #0066cc; color: white; border-radius: 14px;
                    font-weight: 600; font-size: 14px; }
  .category-badge.low { background: #ff6b35; }
  .confidence { color: #666; font-size: 12px; margin-left: 8px; }
  .cat-ranking { margin-top: 16px; }
  .cat-item { display: flex; align-items: center; padding: 6px 0;
              border-bottom: 1px solid #f0f0f0; font-size: 13px; }
  .cat-item .name { width: 120px; font-weight: 500; }
  .cat-item .bar { flex: 1; margin: 0 12px; background: #eee; height: 6px; border-radius: 3px; }
  .cat-item .bar > div { height: 100%; background: #0066cc; border-radius: 3px; }
  .cat-item .score { width: 60px; text-align: right; }
  .faq-item { background: #f8f9fb; padding: 14px; border-radius: 8px; margin: 10px 0;
              border-left: 4px solid #0066cc; }
  .faq-item .meta { font-size: 11px; color: #888; }
  .faq-item .q { font-weight: 600; margin: 4px 0; }
  .faq-item .a { color: #333; font-size: 14px; line-height: 1.5; }
  .faq-item .score { float: right; background: #0066cc; color: white;
                     padding: 2px 8px; border-radius: 10px; font-size: 11px; }
  .timing { color: #888; font-size: 12px; margin-top: 12px; text-align: right; }
  .loading { color: #0066cc; font-size: 13px; }
</style>
</head>
<body>
<h1>🎧 실시간 CS 도움말 검색</h1>
<div class="subtitle">사용자 발화 → 즉시 (< 5ms) 카테고리 분류 + Top-3 FAQ 답변. <strong>모델 선택해서 비교 가능.</strong></div>

<div class="controls">
  <label style="font-weight:600">모델:</label>
  <select id="model"></select>
  <span class="model-info" id="modelInfo"></span>
</div>

<div class="input-box">
  <input type="text" id="query" placeholder="질문을 입력하세요 — 입력하는 즉시 자동 검색됩니다" autocomplete="off" />
  <div class="hint">⚡ 타이핑하는 동안 실시간 분류 (300ms 디바운스)</div>
  <div class="preset">
    <button class="preset-btn" onclick="setQ('결제했는데 카드가 두 번 빠졌어요')">결제 2번</button>
    <button class="preset-btn" onclick="setQ('물건이 아직 안 왔어요 너무 늦어요')">배송 지연</button>
    <button class="preset-btn" onclick="setQ('비밀번호를 잊어버렸어요')">비번 찾기</button>
    <button class="preset-btn" onclick="setQ('이 상품 다른 색상도 있나요')">색상 문의</button>
    <button class="preset-btn" onclick="setQ('앱이 자꾸 꺼져요')">앱 오류</button>
    <button class="preset-btn" onclick="setQ('신규 가입 쿠폰 어디서 받나요')">쿠폰</button>
    <button class="preset-btn" onclick="setQ('오늘 점심 메뉴 추천해줘')">⚠️ 무관</button>
    <button class="preset-btn" onclick="setQ('I want to refund my order')">🌐 영어 환불</button>
  </div>
</div>

<div id="result"></div>

<script>
let timer = null;
let models = [];

async function loadModels() {
  const r = await fetch('/api/models');
  models = await r.json();
  const sel = document.getElementById('model');
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = m.label;
    sel.appendChild(opt);
  });
  updateModelInfo();
}
loadModels();

function updateModelInfo() {
  const key = document.getElementById('model').value;
  const m = models.find(x => x.key === key);
  if (!m) return;
  document.getElementById('modelInfo').innerHTML =
    `<span class="badge">${m.type}</span> <span class="badge">${m.dim}d</span> <span class="badge">${m.size_mb}MB</span>`;
}

document.getElementById('model').addEventListener('change', () => {
  updateModelInfo();
  runSearch();  // 모델 바꾸면 같은 쿼리로 재검색
});

function setQ(q) {
  document.getElementById('query').value = q;
  runSearch();
}

document.getElementById('query').addEventListener('input', () => {
  clearTimeout(timer);
  timer = setTimeout(runSearch, 300);
});

async function runSearch() {
  const text = document.getElementById('query').value.trim();
  const model_key = document.getElementById('model').value;
  if (!text) { document.getElementById('result').innerHTML = ''; return; }
  document.getElementById('result').innerHTML = '<div class="result loading">⏳ ' + model_key + ' 추론 중... (첫 호출 시 모델 로딩 ~10초)</div>';
  const r = await fetch('/api/classify', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, model_key})
  });
  const d = await r.json();
  render(d);
}

function render(d) {
  const maxScore = Math.max(...d.category_ranking.map(c => c.score), 0.01);
  let html = '<div class="result">';
  html += `<div style="float:right;color:#888;font-size:11px">모델: ${d.model_label}</div>`;
  const lowClass = d.low_confidence ? ' low' : '';
  html += `<div class="category-badge${lowClass}">${d.category}</div>`;
  html += `<span class="confidence">신뢰도 ${(d.confidence*100).toFixed(1)}%</span>`;
  html += '<div class="cat-ranking">';
  d.category_ranking.forEach(c => {
    const pct = Math.max(0, c.score / maxScore) * 100;
    html += `<div class="cat-item">
      <span class="name">${c.name}</span>
      <div class="bar"><div style="width:${pct}%"></div></div>
      <span class="score">${c.score.toFixed(3)}</span>
    </div>`;
  });
  html += '</div></div>';
  if (d.top_faqs && d.top_faqs.length) {
    html += '<div class="result"><h3 style="margin-top:0">📋 관련 도움말 Top-3</h3>';
    d.top_faqs.forEach(f => {
      html += `<div class="faq-item">
        <span class="score">${f.score.toFixed(3)}</span>
        <div class="meta">[${f.category}]</div>
        <div class="q">Q. ${f.q}</div>
        <div class="a">A. ${f.a}</div>
      </div>`;
    });
    html += '</div>';
  }
  html += `<div class="timing">⚡ ${d.elapsed_ms.toFixed(2)}ms</div>`;
  document.getElementById('result').innerHTML = html;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
