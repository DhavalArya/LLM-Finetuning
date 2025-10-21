#!/usr/bin/env python3
"""
03_from_api_labels.py (mirror your API schema)
- Use your API's evaluation outputs as labels for SFT/QLoRA training.
- Ignore status/threshold/reason; drop non-numeric scores (e.g., "ERR").
- Skip sections with no source data in payload (prevents poisoning labels).
- Keep all five metrics per section: answer_relevancy, hallucination, summarization, toxicity, bias (0..1).
- Output targets in 0..1 scale, mirroring your API field names.
"""

import argparse, json, os, sys, random
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

SECTION_KEYS = ["overall_summary", "spend", "contracts", "recent_documents", "market_indices", "strategy"]
METRICS = ["answer_relevancy", "hallucination", "summarization", "toxicity", "bias"]

INSTRUCTION = (
    "Given a business JSON and the executive summary text, produce STRICT JSON with fields:\n"
    "{\n"
    '  "sections": { <section>: { "answer_relevancy":0..1, "hallucination":0..1, "summarization":0..1, "toxicity":0..1, "bias":0..1 }, ... },\n'
    '  "overall":  { "answer_relevancy":0..1, "hallucination":0..1, "summarization":0..1, "toxicity":0..1, "bias":0..1 }\n'
    "}\n"
    "Only include sections that are present in the business JSON (e.g., no market_indices if there are no indices/MI text)."
)

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def _load_index_by_tid(payloads_path: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    with open(payloads_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[payloads] skip line {i}: {e}\n")
                continue
            tid = str(obj.get("transaction_id") or f"row-{i}")
            idx[tid] = obj
    return idx

def _extract_sections_text(resp_response: Dict[str, Any]) -> Optional[str]:
    parts = []
    for sec in SECTION_KEYS:
        val = resp_response.get(sec)
        if isinstance(val, str) and val.strip():
            parts.append(f"[{sec}]\n{val.strip()}")
    return "\n\n".join(parts) if parts else None

def _sections_present_in_payload(payload: Dict[str, Any]) -> set:
    present = set()
    present.add("overall_summary")
    if isinstance(payload.get("spendData"), list) and payload["spendData"]:
        present.add("spend")
    if isinstance(payload.get("contractsData"), list) and payload["contractsData"]:
        present.add("contracts")
    has_alerts = isinstance(payload.get("marketIndicesAlerts"), list) and len(payload["marketIndicesAlerts"]) > 0
    emid = payload.get("externalMarketIntelligenceData") or {}
    has_mi_text = isinstance(emid, dict) and bool((emid.get("description") or "").strip())
    if has_alerts or has_mi_text:
        present.add("market_indices")
    si = payload.get("strategyInfo") or {}
    if isinstance(si, dict) and isinstance(si.get("strategiesData"), list) and si["strategiesData"]:
        present.add("strategy")
    if "recent_documents" in payload and payload.get("recent_documents"):
        present.add("recent_documents")
    return present

def _collect_scores(eval_list: List[Dict[str, Any]], allowed_sections: set) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Build per-section metric dict with 0..1 floats where available; ignore non-numeric.
    """
    per_sec: Dict[str, Dict[str, Optional[float]]] = {s: {m: None for m in METRICS} for s in allowed_sections}
    for e in eval_list:
        sec = e.get("section")
        if sec not in allowed_sections:
            continue
        metric = str(e.get("metric", "")).lower()
        if metric not in METRICS:
            continue
        sc = _safe_float(e.get("score"))
        if sc is None:
            continue
        if sec not in per_sec:
            per_sec[sec] = {m: None for m in METRICS}
        per_sec[sec][metric] = max(0.0, min(1.0, sc))
    return per_sec

def _compute_overall(per_sec: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """
    Per-metric mean of available per-section values.
    """
    overall: Dict[str, Optional[float]] = {}
    for m in METRICS:
        vals = []
        for sec in per_sec.keys():
            v = per_sec[sec].get(m)
            if v is not None:
                vals.append(v)
        overall[m] = _mean(vals)
    return overall

def _pack_sample(instruction: str, payload: Dict[str, Any], summary_text: str, target: Dict[str, Any]) -> Dict[str, str]:
    return {
        "instruction": instruction,
        "input": json.dumps(payload, ensure_ascii=False) + "\n\n" + summary_text,
        "output": json.dumps(target, ensure_ascii=False)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payloads", required=True, help="payloads.jsonl used to call API")
    ap.add_argument("--api-out", required=True, help="summaries.eval.jsonl (evaluate=true)")
    ap.add_argument("--sft-out", required=True, help="training JSONL output path")
    ap.add_argument("--val-out", default="", help="optional validation JSONL output path")
    ap.add_argument("--val-frac", type=float, default=0.1, help="validation fraction if --val-out is set")
    args = ap.parse_args()

    tid2payload = _load_index_by_tid(args.payloads)
    samples = []

    with open(args.api_out, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[api-out] skip line {i}: {e}\n")
                continue
            if rec.get("status") != "ok":
                continue

            tid = str(rec.get("transaction_id") or f"row-{i}")
            payload = tid2payload.get(tid)
            if not payload:
                sys.stderr.write(f"[warn] missing payload for {tid}; skipping\n")
                continue

            resp = rec.get("response", {}) if isinstance(rec.get("response", {}), dict) else {}
            rr = resp.get("response", {}) if isinstance(resp, dict) else {}
            eval_list = rr.get("evaluation") if isinstance(rr, dict) else None
            if not isinstance(eval_list, list):
                sys.stderr.write(f"[warn] no evaluation list for {tid}; skipping\n")
                continue

            summary_text = _extract_sections_text(rr)
            if not summary_text:
                sys.stderr.write(f"[warn] no summary text for {tid}; skipping\n")
                continue

            allowed_sections = _sections_present_in_payload(payload)
            if not allowed_sections:
                sys.stderr.write(f"[warn] no valid sections (from payload) for {tid}; skipping\n")
                continue

            per_sec = _collect_scores(eval_list, allowed_sections)
            # prune sections that remained all None (no numeric scores at all)
            per_sec = {s: m for s, m in per_sec.items() if any(v is not None for v in m.values())}
            if not per_sec:
                sys.stderr.write(f"[warn] no numeric section scores for {tid}; skipping\n")
                continue

            overall = _compute_overall(per_sec)
            target = {"sections": per_sec, "overall": overall}

            samples.append(_pack_sample(INSTRUCTION, payload, summary_text, target))

    if not samples:
        sys.stderr.write("[error] No usable samples found. Check the input structure.\n")
        sys.exit(1)

    random.shuffle(samples)
    n = len(samples)
    n_val = int(n * args.val_frac) if args.val_out else 0
    val = samples[:n_val]
    trn = samples[n_val:]

    os.makedirs(os.path.dirname(args.sft_out) or ".", exist_ok=True)
    with open(args.sft_out, "w", encoding="utf-8") as w:
        for ex in trn:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    if args.val_out:
        os.makedirs(os.path.dirname(args.val_out) or ".", exist_ok=True)
        with open(args.val_out, "w", encoding="utf-8") as w:
            for ex in val:
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(trn)} train samples to {args.sft_out}")
    if args.val_out:
        print(f"✅ Wrote {len(val)} val samples to {args.val_out}")

if __name__ == "__main__":
    main()
