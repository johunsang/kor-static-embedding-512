"""영어 STS-B 데이터를 NLLB-200으로 한국어 번역하여 학습 데이터 증강.

영어 STS-B: 5,749 train + 1,500 dev + 1,379 test (영어 원본)
→ KorSTS는 이미 이걸 번역한 셋이지만, 다른 번역기로 paraphrase 효과 기대
→ NLLB-200-distilled-600M로 재번역 → 추가 학습 데이터로 활용
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--output", default="/workspace/data/translated_sts.json")
    p.add_argument("--limit", type=int, default=0, help="0 = 전체")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    print(f"[NLLB 로드] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()

    print("[STS-B 영어 원본 로드]")
    ds = load_dataset("mteb/stsbenchmark-sts", split="train")
    print(f"  train rows: {len(ds)}")
    if args.limit:
        ds = ds.select(range(args.limit))
        print(f"  제한: {args.limit}")

    kor_tgt = tokenizer.convert_tokens_to_ids("kor_Hang")

    def translate(texts, max_length=128):
        with torch.no_grad():
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=max_length).to(device)
            outputs = model.generate(**inputs, forced_bos_token_id=kor_tgt,
                                     max_length=max_length, num_beams=2)
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = []
    total = len(ds)
    t0 = time.time()
    for i in range(0, total, args.batch_size):
        batch = ds.select(range(i, min(i + args.batch_size, total)))
        s1 = [x["sentence1"] for x in batch]
        s2 = [x["sentence2"] for x in batch]
        scores = [float(x["score"]) for x in batch]

        ko1 = translate(s1)
        ko2 = translate(s2)

        for a, b, sc in zip(ko1, ko2, scores):
            results.append({"sentence1": a, "sentence2": b, "score": sc})

        if i % (args.batch_size * 10) == 0:
            elapsed = time.time() - t0
            rate = (i + len(batch)) / max(elapsed, 0.01)
            eta = (total - i - len(batch)) / max(rate, 0.01)
            print(f"  [{i+len(batch)}/{total}]  {rate:.1f} pair/s  ETA {eta/60:.1f}분")

    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {args.output} ({len(results)} pair)")
    print("샘플 3개:")
    for r in results[:3]:
        print(f"  {r['score']:.1f}  s1={r['sentence1']}  s2={r['sentence2']}")


if __name__ == "__main__":
    main()
