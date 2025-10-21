#!/usr/bin/env python3
import argparse
import asyncio
import aiohttp
import json
import os
import random
import sys
from typing import Any, Dict, Optional, Set, Tuple

def load_seen_transaction_ids(out_path: str) -> Set[str]:
    seen = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    tid = str(rec.get("transaction_id", ""))
                    if tid:
                        seen.add(tid)
                except Exception:
                    continue
    return seen

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[skip] invalid JSON line {i}: {e}\n")

class APIClient:
    def __init__(self, url: str, concurrency: int = 3, max_retries: int = 5,
                 timeout_eval: float = 190, timeout_noeval: float = 20, force_evaluate: Optional[bool] = None,
                 headers: Optional[Dict[str,str]] = None):
        self.url = url
        self.semaphore = asyncio.Semaphore(max(1, min(3, concurrency)))
        self.max_retries = max_retries
        self.timeout_eval = timeout_eval
        self.timeout_noeval = timeout_noeval
        self.force_evaluate = force_evaluate
        self.headers = headers or {}
        self.ok_count = 0
        self.err_count = 0

    async def _send_once(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.force_evaluate is not None:
            payload = dict(payload)
            payload["evaluate_response"] = bool(self.force_evaluate)
        eval_flag = payload.get("evaluate_response", False)
        timeout = self.timeout_eval if eval_flag else self.timeout_noeval

        async with session.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status in (429, 500, 502, 503, 504):
                txt = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=resp.status,
                    message=f"{resp.status} body={txt[:300]}"
                )
            if resp.status < 200 or resp.status >= 300:
                return {"error": f"HTTP {resp.status}", "body": await resp.text()}
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return await resp.json(content_type=None)
            txt = await resp.text()
            try:
                return json.loads(txt)
            except Exception:
                return {"raw": txt}

    async def _send_with_retry(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        print(f"➡️ Sending transaction_id={payload.get('transaction_id')}")
        for attempt in range(1, self.max_retries + 1):
            print(f"  ↳ Attempt {attempt}")
            try:
                async with self.semaphore:
                    res = await self._send_once(session, payload)
                print(f"  ✅ transaction_id={payload.get('transaction_id')} success")
                return True, res
            except Exception as e:
                print(f"  ⚠️ transaction_id={payload.get('transaction_id')} error: {e}")
                if attempt >= self.max_retries:
                    return False, {"error": f"{type(e).__name__}: {e}", "attempts": attempt}
                # exponential backoff with jitter
                is_eval = payload.get("evaluate_response") if self.force_evaluate is None else self.force_evaluate
                base = 2.0 if is_eval else 1.0
                delay = min(base * (2 ** (attempt - 1)), 30.0 if is_eval else 8.0)
                delay *= random.uniform(0.5, 1.5)
                await asyncio.sleep(delay)
        return False, {"error": "max retries exceeded"}

    async def run(self, in_path: str, out_path: str):
        seen = load_seen_transaction_ids(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        total, skipped = 0, 0

        print(f"🚀 Starting processing. Input: {in_path}, Output: {out_path}, Already seen: {len(seen)}")

        out_f = open(out_path, "a", encoding="utf-8")
        async with aiohttp.ClientSession() as session:
            tasks = []
            for payload in iter_jsonl(in_path):
                total += 1
                tid = str(payload.get("transaction_id", f"row-{total}"))
                if tid in seen:
                    skipped += 1
                    continue
                tasks.append(self._handle_one(session, payload, tid, out_f))
                if len(tasks) >= 100:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    # count any task-level exceptions
                    for r in results:
                        if isinstance(r, Exception):
                            print(f"[error] task exception: {r}", file=sys.stderr)
                            self.err_count += 1
                    tasks = []
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        print(f"[error] task exception: {r}", file=sys.stderr)
                        self.err_count += 1
        out_f.close()
        print(f"✅ Done. total={total}, skipped(resume)={skipped}, ok={self.ok_count}, failed={self.err_count}. Output → {out_path}")

    async def _handle_one(self, session, payload, tid: str, out_f):
        try:
            ok, res = await self._send_with_retry(session, payload)
            record = {"transaction_id": tid, "status": "ok" if ok else "error"}
            if ok:
                record["response"] = res
                self.ok_count += 1
            else:
                record["error"] = res
                self.err_count += 1
        except Exception as e:
            # absolute safety net: never crash the batch
            record = {"transaction_id": tid, "status": "error", "error": {"fatal": str(e)}}
            self.err_count += 1
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        print(f"[{record['status']}] transaction_id={tid} written to the jsonl file.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--url", default="http://localhost:4000/leo-ds-icm-execsummary/getSummary")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--timeout-eval", type=float, default=300)
    ap.add_argument("--timeout-noeval", type=float, default=20)
    ap.add_argument("--force-evaluate", choices=["true", "false"], help="Force evaluate_response for all payloads")
    ap.add_argument("--auth-header", action="append", help="Custom header 'Key: Value' (repeatable)")
    args = ap.parse_args()

    headers = {
        "Authorization": "Bearer YOUR_API_KEY_HERE",
        "Content-Type": "application/json",
        "OCP-APIM-Subscription-Key": "YOUR_SUBSCRIPTION_KEY_HERE",
    }
    for kv in (args.auth_header or []):
        if ":" in kv:
            k, v = kv.split(":", 1)
            headers[k.strip()] = v.strip()

    force_eval = None
    if args.force_evaluate:
        force_eval = args.force_evaluate.lower() == "true"

    client = APIClient(
        url=args.url,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        timeout_eval=args.timeout_eval,
        timeout_noeval=args.timeout_noeval,
        force_evaluate=force_eval,
        headers=headers,
    )

    asyncio.run(client.run(args.in_path, args.out_path))

if __name__ == "__main__":
    main()
