"""kor-static-embedding-512 테스트 웹앱.

기능:
  1. 의미 검색 (코퍼스 입력 → 쿼리 → Top-K)
  2. 문장 유사도 (두 문장 또는 여러 문장 매트릭스)
  3. 뉴스/문서 요약 (추출적 요약: 중심 임베딩 기반)
  4. 클러스터링 (KMeans)
  5. 한계 케이스 빠른 테스트 (부정문, 어순, 다의어)

실행:
  .venv/bin/uvicorn webapp:app --reload --port 8765
  → http://localhost:8765
"""

import time
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

app = FastAPI()

print("[모델 로딩...]")
t0 = time.time()
MODEL = SentenceTransformer("kekeappa/kor-static-embedding-512")
print(f"[로딩 완료] {(time.time()-t0)*1000:.0f}ms, dim={MODEL.get_embedding_dimension()}")


def split_sentences(text: str) -> List[str]:
    import re
    sents = re.split(r"(?<=[.!?。!?])\s+|\n+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 2]


# ---------- API ----------
class SearchReq(BaseModel):
    query: str
    corpus: List[str]
    top_k: int = 5


@app.post("/api/search")
def api_search(req: SearchReq):
    t = time.time()
    q_emb = MODEL.encode([req.query], normalize_embeddings=True)
    c_emb = MODEL.encode(req.corpus, normalize_embeddings=True, batch_size=64)
    scores = (q_emb @ c_emb.T)[0]
    top = np.argsort(-scores)[: req.top_k]
    elapsed_ms = (time.time() - t) * 1000
    return {
        "elapsed_ms": elapsed_ms,
        "results": [{"text": req.corpus[i], "score": float(scores[i])} for i in top],
    }


class SimReq(BaseModel):
    sentences: List[str]


@app.post("/api/similarity")
def api_similarity(req: SimReq):
    t = time.time()
    emb = MODEL.encode(req.sentences, normalize_embeddings=True)
    sim = (emb @ emb.T).tolist()
    elapsed_ms = (time.time() - t) * 1000
    return {"elapsed_ms": elapsed_ms, "matrix": sim, "sentences": req.sentences}


class SummarizeReq(BaseModel):
    text: str
    n_sentences: int = 3


@app.post("/api/summarize")
def api_summarize(req: SummarizeReq):
    """추출적 요약: 모든 문장 임베딩의 평균(=중심)과 가까운 문장 N개 선택."""
    t = time.time()
    sents = split_sentences(req.text)
    if len(sents) <= req.n_sentences:
        return {"summary": sents, "all_sents": sents, "elapsed_ms": 0,
                "note": "문장 수가 너무 적습니다."}
    emb = MODEL.encode(sents, normalize_embeddings=True, batch_size=64)
    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims = emb @ centroid
    # 중심과 가까운 N개 (원래 순서 유지)
    chosen_idx = sorted(np.argsort(-sims)[: req.n_sentences])
    elapsed_ms = (time.time() - t) * 1000
    return {
        "summary": [sents[i] for i in chosen_idx],
        "summary_idx": [int(i) for i in chosen_idx],
        "all_sents": sents,
        "scores": [float(sims[i]) for i in range(len(sents))],
        "elapsed_ms": elapsed_ms,
    }


class ClusterReq(BaseModel):
    sentences: List[str]
    n_clusters: int = 3


@app.post("/api/cluster")
def api_cluster(req: ClusterReq):
    t = time.time()
    emb = MODEL.encode(req.sentences, normalize_embeddings=True)
    k = min(req.n_clusters, len(req.sentences))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb)
    labels = km.labels_.tolist()
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(lab, []).append(req.sentences[i])
    elapsed_ms = (time.time() - t) * 1000
    return {
        "elapsed_ms": elapsed_ms,
        "groups": [{"id": int(k), "sentences": v} for k, v in sorted(groups.items())],
    }


# ---------- HTML ----------
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>kor-static-embedding-512 테스트</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f7; color: #1d1d1f; }
  h1 { font-size: 28px; margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 24px; font-size: 14px; }
  .tabs { display: flex; gap: 4px; border-bottom: 2px solid #ddd; margin-bottom: 24px; flex-wrap: wrap; }
  .tab { padding: 12px 20px; cursor: pointer; border: none; background: none; font-size: 15px;
         color: #666; border-bottom: 3px solid transparent; margin-bottom: -2px; }
  .tab.active { color: #0066cc; border-bottom-color: #0066cc; font-weight: 600; }
  .panel { display: none; background: white; padding: 24px; border-radius: 12px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .panel.active { display: block; }
  textarea, input[type=text], input[type=number] { width: 100%; padding: 10px; font-size: 14px;
           border: 1px solid #ccc; border-radius: 8px; font-family: inherit; }
  textarea { resize: vertical; min-height: 100px; }
  label { display: block; font-weight: 600; margin: 12px 0 6px; font-size: 14px; }
  button { padding: 10px 24px; background: #0066cc; color: white; border: none;
           border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 600; margin-top: 12px; }
  button:hover { background: #0052a3; }
  button:disabled { background: #999; cursor: not-allowed; }
  .result { margin-top: 20px; padding: 16px; background: #f8f8f8; border-radius: 8px;
            border-left: 4px solid #0066cc; }
  .result-item { padding: 10px 12px; margin: 8px 0; background: white; border-radius: 6px;
                 display: flex; gap: 12px; align-items: center; }
  .score { background: #0066cc; color: white; padding: 4px 10px; border-radius: 12px;
           font-size: 12px; font-weight: 600; min-width: 60px; text-align: center; }
  .timing { color: #888; font-size: 12px; margin-top: 8px; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { padding: 8px; text-align: center; border: 1px solid #eee; font-size: 13px; }
  th { background: #f0f0f0; }
  td.heatmap { font-weight: 600; }
  .preset-btn { padding: 6px 12px; margin: 4px 4px 0 0; background: #eee; color: #333;
                border: 1px solid #ddd; border-radius: 16px; font-size: 12px; cursor: pointer; }
  .preset-btn:hover { background: #ddd; }
  .cluster-group { margin: 12px 0; padding: 12px; background: white; border-radius: 8px;
                   border-left: 4px solid #0066cc; }
  .cluster-group h4 { margin: 0 0 8px; color: #0066cc; }
  .badge { display: inline-block; padding: 2px 8px; background: #f0f0f0; border-radius: 8px;
           font-size: 11px; margin-left: 8px; color: #666; }
  .summary-sent { padding: 6px 10px; margin: 4px 0; border-radius: 6px; }
  .summary-sent.chosen { background: #e8f4ff; border-left: 3px solid #0066cc; font-weight: 600; }
  .summary-sent.skipped { color: #999; }
  .info { background: #fff8dc; padding: 10px; border-radius: 6px; font-size: 13px; margin: 12px 0; }
</style>
</head>
<body>
<h1>🇰🇷 kor-static-embedding-512 테스트 콘솔</h1>
<div class="subtitle">한국어 Static Embedding (68MB · 512d · CPU 최적화) — <a href="https://huggingface.co/kekeappa/kor-static-embedding-512" target="_blank">HuggingFace 모델 카드</a></div>

<div class="tabs">
  <button class="tab active" onclick="switchTab('search')">🔍 의미 검색</button>
  <button class="tab" onclick="switchTab('similarity')">📊 유사도 매트릭스</button>
  <button class="tab" onclick="switchTab('summarize')">📰 뉴스 요약</button>
  <button class="tab" onclick="switchTab('cluster')">🗂️ 클러스터링</button>
  <button class="tab" onclick="switchTab('limit')">⚠️ 한계 테스트</button>
</div>

<!-- 1. 검색 -->
<div id="search" class="panel active">
  <label>코퍼스 (한 줄당 하나의 문서)</label>
  <textarea id="corpus" rows="10">김치찌개는 신김치와 돼지고기를 넣고 끓이는 한국 전통 음식이다.
된장찌개 만드는 법: 멸치 육수에 된장을 풀고 채소를 넣어 끓인다.
파이썬은 배우기 쉽고 활용도가 높은 프로그래밍 언어다.
리액트는 페이스북이 만든 사용자 인터페이스 라이브러리다.
도커는 컨테이너 기반 가상화 플랫폼이다.
git은 소스코드 버전 관리 시스템이다.
딥러닝은 다층 신경망을 사용하는 머신러닝 기법이다.
GPT는 OpenAI가 개발한 대규모 언어 모델이다.
RAG는 검색 증강 생성으로 외부 지식을 활용한다.
제주도는 한국 남쪽에 위치한 화산섬으로 관광지로 유명하다.
부산은 한국 제2의 도시이며 해운대 해수욕장이 유명하다.
유산소 운동은 심혈관 건강에 좋다.
근력 운동은 근육량을 늘리고 기초대사량을 높인다.
장마철에는 우산을 항상 챙겨야 한다.
오늘은 햇살이 따뜻하고 맑은 날씨다.</textarea>
  <label>쿼리</label>
  <input type="text" id="query" value="인공지능 모델 구조" />
  <div>
    <span class="preset-btn" onclick="setQuery('맛있는 한국 음식')">맛있는 한국 음식</span>
    <span class="preset-btn" onclick="setQuery('코드 버전 관리')">코드 버전 관리</span>
    <span class="preset-btn" onclick="setQuery('여행 가기 좋은 도시')">여행 가기 좋은 도시</span>
    <span class="preset-btn" onclick="setQuery('건강한 습관')">건강한 습관</span>
    <span class="preset-btn" onclick="setQuery('비 올 때 준비물')">비 올 때 준비물</span>
  </div>
  <label>Top-K</label>
  <input type="number" id="topk" value="5" min="1" max="20" />
  <button onclick="doSearch()">🔍 검색</button>
  <div id="search_result"></div>
</div>

<!-- 2. 유사도 -->
<div id="similarity" class="panel">
  <label>여러 문장 (한 줄당 하나)</label>
  <textarea id="sim_sentences" rows="6">오늘 날씨가 정말 좋네요.
햇살이 따뜻하고 기분 좋은 하루입니다.
파이썬으로 머신러닝을 배우고 있어요.
딥러닝 프레임워크를 공부 중입니다.
김치찌개에 라면 사리를 넣어 먹었다.</textarea>
  <button onclick="doSimilarity()">📊 유사도 매트릭스 계산</button>
  <div id="sim_result"></div>
</div>

<!-- 3. 요약 -->
<div id="summarize" class="panel">
  <label>긴 텍스트 / 뉴스 기사</label>
  <textarea id="article" rows="12">정부가 인공지능(AI) 산업 육성을 위해 향후 5년간 2조원을 투자한다고 발표했다. 이번 발표는 글로벌 AI 경쟁에서 한국이 뒤처지지 않기 위한 종합 전략의 일환이다. 과학기술정보통신부는 오늘 기자회견을 열고 구체적인 투자 계획을 공개했다.
주요 투자 분야는 대규모 언어 모델(LLM) 개발, AI 반도체, 그리고 산업별 응용 AI다. 특히 한국어에 특화된 거대 언어 모델 개발에 5000억원이 배정되어 가장 큰 비중을 차지했다.
정부는 또한 AI 인재 양성을 위해 매년 1만명의 전문 인력을 육성하겠다고 밝혔다. 대학과의 협력을 통해 AI 관련 학과를 신설하고 기업 연계 과정도 확대할 예정이다.
업계 반응은 긍정적이다. 한 AI 스타트업 대표는 "정부의 적극적인 지원으로 한국 AI 생태계가 한 단계 도약할 수 있을 것"이라고 말했다. 다만 일부 전문가들은 인프라 투자만으로는 부족하며 규제 완화도 함께 추진되어야 한다고 지적했다.
이번 발표 이후 관련 AI 주식이 일제히 상승세를 보였다. 시장 전문가들은 향후 6개월간 추가적인 정책 발표가 있을 것으로 전망하고 있다.</textarea>
  <label>요약 문장 수</label>
  <input type="number" id="n_sents" value="3" min="1" max="10" />
  <button onclick="doSummarize()">📰 요약</button>
  <div id="sum_result"></div>
  <div class="info">📌 알고리즘: 모든 문장을 임베딩 → 평균(중심) 벡터 계산 → 중심에 가장 가까운 N개 문장 선택 (추출적 요약).</div>
</div>

<!-- 4. 클러스터링 -->
<div id="cluster" class="panel">
  <label>문장 목록 (한 줄당 하나)</label>
  <textarea id="cluster_sentences" rows="10">김치찌개 만드는 법
된장찌개 끓이는 방법
비빔밥 맛있게 먹기
파이썬 프로그래밍 입문
자바스크립트 기초
리액트 컴포넌트 작성
딥러닝 학습 방법
신경망 구조 이해
GPT 활용 사례
제주도 여행 추천
부산 여행 코스
경주 역사 탐방</textarea>
  <label>클러스터 수 (K)</label>
  <input type="number" id="n_clusters" value="4" min="2" max="10" />
  <button onclick="doCluster()">🗂️ 자동 그룹화</button>
  <div id="cluster_result"></div>
</div>

<!-- 5. 한계 -->
<div id="limit" class="panel">
  <h3>Static Embedding 의 알려진 약점</h3>
  <button onclick="testCase(0)">1. 부정문 vs 긍정문</button>
  <button onclick="testCase(1)">2. 어순 차이</button>
  <button onclick="testCase(2)">3. 다의어 (동음이의)</button>
  <button onclick="testCase(3)">4. 반어/풍자</button>
  <div id="limit_result"></div>
</div>

<script>
const LIMIT_CASES = [
  ["나는 김치찌개를 좋아한다.", "나는 김치찌개를 좋아하지 않는다.", "나는 김치찌개를 싫어한다."],
  ["철수가 영희를 좋아한다.", "영희가 철수를 좋아한다.", "철수와 영희는 서로 좋아한다."],
  ["은행에서 돈을 인출했다.", "강변 은행에 앉아 책을 읽었다.", "ATM에서 현금을 뽑았다."],
  ["오늘 정말 재미있는 영화였다.", "오늘 정말 재미있는 영화였다... 라고 할 줄 알았지.", "정말 지루한 영화였다."],
];

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(name).classList.add('active');
}

function setQuery(q) { document.getElementById('query').value = q; doSearch(); }

async function doSearch() {
  const corpus = document.getElementById('corpus').value.split('\\n').filter(s => s.trim());
  const query = document.getElementById('query').value;
  const top_k = parseInt(document.getElementById('topk').value);
  const r = await fetch('/api/search', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query, corpus, top_k})});
  const d = await r.json();
  let html = '<div class="result"><strong>Top-' + top_k + ' 결과</strong>';
  d.results.forEach((it, i) => {
    html += '<div class="result-item"><span class="score">' + it.score.toFixed(4) + '</span><span>' + (i+1) + '. ' + it.text + '</span></div>';
  });
  html += '<div class="timing">⚡ ' + d.elapsed_ms.toFixed(1) + 'ms (모델 + 코퍼스 임베딩 포함)</div></div>';
  document.getElementById('search_result').innerHTML = html;
}

async function doSimilarity() {
  const sentences = document.getElementById('sim_sentences').value.split('\\n').filter(s => s.trim());
  const r = await fetch('/api/similarity', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sentences})});
  const d = await r.json();
  let html = '<div class="result"><strong>코사인 유사도 매트릭스</strong><table><tr><th></th>';
  d.sentences.forEach((s, i) => html += '<th>S'+(i+1)+'</th>');
  html += '</tr>';
  d.matrix.forEach((row, i) => {
    html += '<tr><th>S'+(i+1)+'</th>';
    row.forEach(v => {
      const intensity = Math.max(0, Math.min(1, (v + 0.2) / 1.2));
      const bg = 'rgba(0,102,204,' + intensity + ')';
      html += '<td class="heatmap" style="background:' + bg + ';color:' + (intensity > 0.4 ? 'white' : 'black') + '">' + v.toFixed(3) + '</td>';
    });
    html += '</tr>';
  });
  html += '</table><div style="margin-top:12px">';
  d.sentences.forEach((s, i) => html += '<div><strong>S'+(i+1)+':</strong> ' + s + '</div>');
  html += '</div><div class="timing">⚡ ' + d.elapsed_ms.toFixed(1) + 'ms</div></div>';
  document.getElementById('sim_result').innerHTML = html;
}

async function doSummarize() {
  const text = document.getElementById('article').value;
  const n_sentences = parseInt(document.getElementById('n_sents').value);
  const r = await fetch('/api/summarize', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, n_sentences})});
  const d = await r.json();
  let html = '<div class="result"><strong>📰 추출 요약 (' + d.summary.length + '문장):</strong>';
  d.summary.forEach((s, i) => html += '<div class="summary-sent chosen">• ' + s + '</div>');
  html += '<hr style="margin:16px 0"/><strong>전체 문장 (중심 유사도 기준):</strong>';
  d.all_sents.forEach((s, i) => {
    const chosen = d.summary_idx.includes(i);
    html += '<div class="summary-sent ' + (chosen ? 'chosen' : 'skipped') + '">[' + d.scores[i].toFixed(3) + '] ' + s + '</div>';
  });
  html += '<div class="timing">⚡ ' + d.elapsed_ms.toFixed(1) + 'ms · 총 ' + d.all_sents.length + '문장</div></div>';
  document.getElementById('sum_result').innerHTML = html;
}

async function doCluster() {
  const sentences = document.getElementById('cluster_sentences').value.split('\\n').filter(s => s.trim());
  const n_clusters = parseInt(document.getElementById('n_clusters').value);
  const r = await fetch('/api/cluster', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sentences, n_clusters})});
  const d = await r.json();
  let html = '<div class="result"><strong>🗂️ ' + d.groups.length + '개 그룹</strong>';
  d.groups.forEach((g, i) => {
    html += '<div class="cluster-group"><h4>그룹 ' + (i+1) + '<span class="badge">' + g.sentences.length + '개</span></h4>';
    g.sentences.forEach(s => html += '<div>• ' + s + '</div>');
    html += '</div>';
  });
  html += '<div class="timing">⚡ ' + d.elapsed_ms.toFixed(1) + 'ms</div></div>';
  document.getElementById('cluster_result').innerHTML = html;
}

async function testCase(idx) {
  const sentences = LIMIT_CASES[idx];
  const r = await fetch('/api/similarity', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({sentences})});
  const d = await r.json();
  const titles = ['부정문 vs 긍정문', '어순 차이', '다의어 (동음이의)', '반어/풍자'];
  const explains = [
    'S1(좋아함) ↔ S2(좋아하지 않음): 단어가 거의 같아 높게 나오는 게 약점. S2 ↔ S3(싫어함)는 의미는 같지만 단어가 다름.',
    'S1과 S2는 단어 구성이 같고 어순만 다름 → Static Embedding은 거의 동일하게 봄.',
    '동음이의어 "은행"이 다른 의미인데 단어가 같아 S1과 S2를 가깝게 보는 경향.',
    '반어법은 단어로는 긍정인데 의미는 부정 → Static Embedding이 잡지 못함.',
  ];
  let html = '<div class="result"><strong>' + titles[idx] + '</strong>';
  sentences.forEach((s, i) => html += '<div>S' + (i+1) + ': ' + s + '</div>');
  html += '<table style="margin-top:12px"><tr><th></th>';
  sentences.forEach((s, i) => html += '<th>S'+(i+1)+'</th>');
  html += '</tr>';
  d.matrix.forEach((row, i) => {
    html += '<tr><th>S'+(i+1)+'</th>';
    row.forEach(v => {
      const intensity = Math.max(0, Math.min(1, (v + 0.2) / 1.2));
      const bg = 'rgba(0,102,204,' + intensity + ')';
      html += '<td class="heatmap" style="background:' + bg + ';color:' + (intensity > 0.4 ? 'white' : 'black') + '">' + v.toFixed(3) + '</td>';
    });
    html += '</tr>';
  });
  html += '</table><div class="info">💡 ' + explains[idx] + '</div></div>';
  document.getElementById('limit_result').innerHTML = html;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
