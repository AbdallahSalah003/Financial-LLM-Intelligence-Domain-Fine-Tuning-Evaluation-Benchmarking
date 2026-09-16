"""
FinancialIntelligenceBenchmarking is to compare: HuggingFace (base + LoRA) vs KServe + vLLM

Per request:  input_tokens, output_tokens, total_tokens,
              ttft, generation_latency, total_latency,
              output_tokens_per_sec, total_tokens_per_sec

Per run:      mean, median, p50, p90, p95, p99, min, max, std
              + system throughput and requests/sec

    bench = FinancialIntelligenceBenchmarking(prompts, max_new_tokens=512)
    hf = bench.benchmark_huggingface("Qwen/Qwen2.5-1.5B-Instruct",
                                     "abdallahsalah0/qwen25-1.5b-arafinnews-lora")
    ks = bench.benchmark_kserve_vllm("http://my-svc.default.example.com",
                                     "arafinnews", concurrency=8)
    print(bench.compare(hf, ks))

`prompts` is a list of chat message lists -> Build them with build_prompts()
"""

from __future__ import annotations
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a professional NLP data parser.\n"
    "Follow the provided `Task` by the user and the `Output Scheme` to generate "
    "the `Output JSON`.\nDo not generate any introduction or conclusion."
)

TASK = (
    "Your task is to translate and extract financial events and metrics.\n"
    "You will be provided by an Arabic article associated with an Output Scheme.\n"
    "Generate the ouptut in the same input text language.\n"
    "You have to extract JSON details from text according the Output Scheme details.\n"
    "Extract details as mentioned in text."
)


def build_prompts(jsonl_path: str | Path, schema_json: str = "{}", n: int | None = None):
    """Build benchmark prompts from the notebook's synthetic/test.jsonl"""
    prompts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join([
                    f"# Article Title: {r.get('title_ar', '')}",
                    "# Article: ", r.get("article", ""),
                    "# Task:", TASK,
                    "# Output Scheme:", schema_json,
                ])},
            ])
            if n and len(prompts) >= n:
                break
    return prompts


#  statistics 

def _pct(xs: list[float], q: float) -> float:
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats(values) -> dict:
    """mean, median, p50, p90, p95, p99, min, max, std."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return dict.fromkeys(
            ["mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std"], 0.0)
    return {
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p50": _pct(xs, 50), "p90": _pct(xs, 90),
        "p95": _pct(xs, 95), "p99": _pct(xs, 99),
        "min": xs[0], "max": xs[-1],
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def _derive(r: dict) -> dict:
    """Fill total_tokens and the two throughput figures."""
    r["total_tokens"] = r["input_tokens"] + r["output_tokens"]
    gen, tot = r.get("generation_latency"), r["total_latency"]
    r["output_tokens_per_sec"] = r["output_tokens"] / gen if gen and r["output_tokens"] > 1 else None
    r["total_tokens_per_sec"] = r["total_tokens"] / tot if tot else None
    return r


class Result:
    """Records from one backend run"""

    METRICS = ["input_tokens", "output_tokens", "total_tokens", "ttft",
               "generation_latency", "total_latency",
               "output_tokens_per_sec", "total_tokens_per_sec"]

    def __init__(self, backend, model, concurrency, wall_time, records):
        self.backend, self.model = backend, model
        self.concurrency, self.wall_time = concurrency, wall_time
        self.records = records

    @property
    def ok(self):
        return [r for r in self.records if r["success"]]

    def summary(self) -> dict:
        ok = self.ok
        w = self.wall_time or 1
        return {
            "backend": self.backend, "model": self.model,
            "concurrency": self.concurrency,
            "requests": len(self.records), "successful": len(ok),
            "success_rate": len(ok) / len(self.records) if self.records else 0.0,
            "wall_time_s": self.wall_time,
            "requests_per_sec": len(ok) / w,
            "system_output_tokens_per_sec": sum(r["output_tokens"] for r in ok) / w,
            "metrics": {m: stats(r[m] for r in ok) for m in self.METRICS},
        }

    def table(self) -> str:
        s = self.summary()
        keys = ["mean", "median", "p50", "p90", "p95", "p99", "min", "max", "std"]
        lines = [
            f"### {self.backend} — {self.model} (concurrency={self.concurrency})", "",
            f"requests={s['requests']}  success={s['success_rate']:.0%}  "
            f"wall={s['wall_time_s']:.1f}s  rps={s['requests_per_sec']:.2f}  "
            f"system_out_tok/s={s['system_output_tokens_per_sec']:.1f}", "",
            "| Metric | " + " | ".join(k.capitalize() for k in keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        for m in self.METRICS:
            fmt = "{:.0f}" if m.endswith("tokens") else "{:.3f}"
            lines.append(f"| {m} | " + " | ".join(fmt.format(s["metrics"][m][k]) for k in keys) + " |")
        return "\n".join(lines)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(), "records": self.records}, f, indent=2)


# benchmark 

class FinancialIntelligenceBenchmarking:
    """Same timing logic on both backends, so the numbers are comparable

    TTFT is stamped on the first chunk that carries real generated text 
    empty / role-only streaming chunks are ignored
    """

    def __init__(self, prompts, max_new_tokens=512, temperature=0.0,
                 warmup=2, timeout=300):
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.warmup = warmup
        self.timeout = timeout


    def benchmark_huggingface(self, model_id, adapter_id=None, dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=getattr(torch, dtype))
        if adapter_id:
            model.load_adapter(adapter_id)
        model.eval()

        def run(messages):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok([text], return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            threading.Thread(target=model.generate, kwargs=dict(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, streamer=streamer), daemon=True).start()

            pieces, first, last = [], None, None
            for chunk in streamer:
                if not chunk:                       
                    continue
                now = time.perf_counter()
                first = first or now
                last = now
                pieces.append(chunk)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            out = "".join(pieces)
            return _derive({
                "input_tokens": int(inputs.input_ids.shape[-1]),
                "output_tokens": len(tok(out, add_special_tokens=False).input_ids),
                "ttft": first - t0 if first else None,
                "generation_latency": last - first if first else None,
                "total_latency": end - t0,
                "success": True, "error": None,
            })

        name = "huggingface+lora" if adapter_id else "huggingface"
        return self._drive(run, name, model_id, concurrency=1)


    def benchmark_kserve_vllm(self, base_url, model_name, api_key="EMPTY",
                              host_header=None, path="/openai/v1/chat/completions",
                              concurrency=1):
        """`path` is the KServe vLLM OpenAI route.
        For a plain `vllm serve` endpoint pass path="/v1/chat/completions".
        """
        import requests

        url = base_url.rstrip("/") + path
        headers = {"Authorization": f"Bearer {api_key}"}
        if host_header:
            headers["Host"] = host_header
        local = threading.local()

        def run(messages):
            if not hasattr(local, "s"):
                local.s = requests.Session()
            payload = {"model": model_name, "messages": messages,
                       "max_tokens": self.max_new_tokens,
                       "temperature": self.temperature, "stream": True,
                       "stream_options": {"include_usage": True}}

            t0 = time.perf_counter()
            resp = local.s.post(url, headers=headers, json=payload,
                                stream=True, timeout=self.timeout)
            resp.raise_for_status()

            first = last = None
            chunks, usage = 0, None
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                now = time.perf_counter()               
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or usage
                choices = obj.get("choices") or [{}]
                if not (choices[0].get("delta") or {}).get("content"):
                    continue                            
                first = first or now
                last = now
                chunks += 1
            end = time.perf_counter()

            usage = usage or {}
            return _derive({
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", chunks)),
                "ttft": first - t0 if first else None,
                "generation_latency": last - first if first else None,
                "total_latency": end - t0,
                "success": True, "error": None,
            })

        return self._drive(run, "kserve+vllm", model_name, concurrency)

    #  driver

    def _drive(self, run, backend, model, concurrency):
        def safe(messages):
            t0 = time.perf_counter()
            try:
                return run(messages)
            except Exception as e:
                return _derive({"input_tokens": 0, "output_tokens": 0, "ttft": None,
                                "generation_latency": None,
                                "total_latency": time.perf_counter() - t0,
                                "success": False, "error": f"{type(e).__name__}: {e}"})

        for p in self.prompts[:self.warmup]:            # warmup, discarded
            safe(p)

        t0 = time.perf_counter()
        if concurrency <= 1:
            records = [safe(p) for p in self.prompts]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                records = list(pool.map(safe, self.prompts))
        wall = time.perf_counter() - t0

        failed = [r for r in records if not r["success"]]
        if failed:
            print(f"[{backend}] {len(failed)}/{len(records)} failed — {failed[0]['error']}")
        return Result(backend, model, concurrency, wall, records)

    #  comparison 

    @staticmethod
    def compare(base: Result, other: Result) -> str:
        a, b = base.summary(), other.summary()
        rows = [
            ("TTFT p50 (s)", a["metrics"]["ttft"]["p50"], b["metrics"]["ttft"]["p50"], False),
            ("TTFT p95 (s)", a["metrics"]["ttft"]["p95"], b["metrics"]["ttft"]["p95"], False),
            ("Latency p50 (s)", a["metrics"]["total_latency"]["p50"],
             b["metrics"]["total_latency"]["p50"], False),
            ("Latency p95 (s)", a["metrics"]["total_latency"]["p95"],
             b["metrics"]["total_latency"]["p95"], False),
            ("Output tok/s (median)", a["metrics"]["output_tokens_per_sec"]["median"],
             b["metrics"]["output_tokens_per_sec"]["median"], True),
            ("System output tok/s", a["system_output_tokens_per_sec"],
             b["system_output_tokens_per_sec"], True),
            ("Requests/sec", a["requests_per_sec"], b["requests_per_sec"], True),
            ("Success rate", a["success_rate"], b["success_rate"], True),
        ]
        lines = ["# Benchmark comparison", "",
                 f"| Metric | {a['backend']} | {b['backend']} | speedup |",
                 "|---|---:|---:|---:|"]
        for label, x, y, higher in rows:
            sp = (y / x if higher else x / y) if x > 0 and y > 0 else 0
            lines.append(f"| {label} | {x:.3f} | {y:.3f} | {sp:.2f}x |" if sp
                         else f"| {label} | {x:.3f} | {y:.3f} | n/a |")
        lines += ["", base.table(), "", other.table(), "",
                  "_`output tok/s` is per-request decode speed; `system output tok/s` "
                  "is aggregate throughput and is the number that matters under concurrency._"]
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="synthetic/test.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="abdallahsalah0/qwen25-1.5b-arafinnews-lora")
    ap.add_argument("--kserve-url", required=True)
    ap.add_argument("--kserve-model", default="arafinnews")
    ap.add_argument("--host-header")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    bench = FinancialIntelligenceBenchmarking(build_prompts(args.data, n=args.n))
    hf = bench.benchmark_huggingface(args.model, args.adapter)
    ks = bench.benchmark_kserve_vllm(args.kserve_url, args.kserve_model,
                                     host_header=args.host_header,
                                     concurrency=args.concurrency)
    print(bench.compare(hf, ks))