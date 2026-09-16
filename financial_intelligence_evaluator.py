from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


# Normalization

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# longest first "billion" must win over "bn" and مليارات contains مليار
_SCALES = sorted(
    [("trillion", 1e12), ("تريليون", 1e12), ("billion", 1e9), ("مليارات", 1e9),
     ("مليار", 1e9), ("bn", 1e9), ("million", 1e6), ("ملايين", 1e6),
     ("مليون", 1e6), ("mn", 1e6), ("thousand", 1e3), ("ألف", 1e3), ("الف", 1e3)],
    key=lambda x: -len(x[0]))

_CURRENCY = {
    "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD", "دولار": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "يورو": "EUR",
    "£": "GBP", "gbp": "GBP", "استرليني": "GBP",
    "sar": "SAR", "ريال": "SAR", "aed": "AED", "درهم": "AED",
    "egp": "EGP", "جنيه": "EGP", "kwd": "KWD", "دينار": "KWD",
    "jpy": "JPY", "ين": "JPY", "cny": "CNY", "يوان": "CNY",
}

_SUFFIXES = {"inc", "ltd", "llc", "plc", "co", "corp", "company", "group",
             "holding", "holdings", "the", "شركة", "مجموعة"}


def norm(s) -> str:
    """lowercase, unify Arabic letters & drop punctuation & corporate suffixes"""
    if s is None:
        return ""
    s = str(s).translate(_DIGITS).casefold()
    s = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", s)         
    for src, dst in (("أإآٱ", "ا"), ("ى", "ي"), ("ة", "ه")):
        for ch in src:
            s = s.replace(ch, dst)
    s = re.sub(r"[^\w\s%]", " ", s)
    return " ".join(w for w in s.split() if w not in _SUFFIXES)


def num(s) -> str | None:
    """'1.84 مليار' -> '1840000000' ; '131%' -> '131' ; None if no number."""
    if s is None:
        return None
    t = str(s).translate(_DIGITS).casefold()
    m = re.search(r"-?\d[\d,]*\.?\d*", t)
    if not m:
        return None
    try:
        val = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    for word, mult in _SCALES:
        if word in t:
            val *= mult
            break
    return f"{val:.4f}".rstrip("0").rstrip(".")


def currency(s) -> str | None:
    if s is None:
        return None
    t = norm(s)
    for token in t.split() + [t]:
        if token in _CURRENCY:
            return _CURRENCY[token]
    return t.upper() or None


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score, 0-100. Stable on short text, unlike BLEU."""
    h, r = re.sub(r"\s+", "", hyp or ""), re.sub(r"\s+", "", ref or "")
    if not h or not r:
        return 0.0
    ps, rs = [], []
    for n in range(1, max_n + 1):
        ch = Counter(h[i:i + n] for i in range(len(h) - n + 1))
        cr = Counter(r[i:i + n] for i in range(len(r) - n + 1))
        hit = sum((ch & cr).values())
        ps.append(hit / max(sum(ch.values()), 1))
        rs.append(hit / max(sum(cr.values()), 1))
    p, q = sum(ps) / max_n, sum(rs) / max_n
    return 0.0 if p + q == 0 else 100 * (1 + beta**2) * p * q / (beta**2 * p + q)


# Parsing

_FIELDS = {"title_ar": str, "title_en": str, "translation": str,
           "companies": list, "people": list, "countries": list, "locations": list,
           "financial_events": list, "financial_metrics": list, "sentiment": str}


def parse(raw: str) -> dict | None:
    """Pull the JSON object out of a fenced / truncated model output."""
    if not raw or not raw.strip():
        return None
    m = (re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
         or re.search(r"```(?:json)?\s*(.*)$", raw, re.DOTALL))
    text = m.group(1) if m else (raw[raw.find("{"):] if "{" in raw else "")
    if not text.strip():
        return None
    try:
        obj = json.loads(text)
    except Exception:
        if repair_json is None:
            return None
        try:
            obj = repair_json(text, return_objects=True)
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def is_valid(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if any(k not in obj or not isinstance(obj[k], t) for k, t in _FIELDS.items()):
        return False
    return obj["sentiment"].strip().lower() in {"positive", "negative", "neutral"}


# Matching

def f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def match_entities(pred: list[str], gold: list[str], thr: float = 0.9):
    """Greedy 1:1 fuzzy match, so small surface drift still gets credit."""
    pred = [x for x in dict.fromkeys(pred) if x]
    gold = [x for x in dict.fromkeys(gold) if x]
    used, tp = [False] * len(gold), 0
    for p in pred:
        best, bi = 0.0, -1
        for i, g in enumerate(gold):
            if not used[i]:
                s = 1.0 if p == g else SequenceMatcher(None, p, g).ratio()
                if s > best:
                    best, bi = s, i
        if bi >= 0 and best >= thr:
            used[bi], tp = True, tp + 1
    return tp, len(pred) - tp, len(gold) - tp


def match_tuples(pred: list, gold: list):
    cp, cg = Counter(pred), Counter(gold)
    tp = sum((cp & cg).values())
    return tp, sum(cp.values()) - tp, sum(cg.values()) - tp


# Evaluator

class FinancialIntelligenceEvaluator:
    METRICS = ["schema_valid", "entity_f1", "event_f1", "metric_f1", "translation_chrf"]
    ENTITY_FIELDS = ("companies", "people", "countries", "locations")

    @staticmethod
    def _load(src) -> list[dict]:
        if isinstance(src, (str, Path)):
            return [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        return list(src)

    @staticmethod
    def _raw(rec: dict) -> str:
        for k in ("raw_output", "output", "prediction", "response"):
            if rec.get(k):
                v = rec[k]
                return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return ""

    def evaluate(self, gold, predictions, name: str = "model") -> dict:
        rows = self._load(gold)
        preds = {str(r["id"]): self._raw(r) for r in self._load(predictions)}

        valid, chrfs = 0, []
        ent, evt, met = Counter(), Counter(), Counter()

        for g in rows:
            p = parse(preds.get(str(g["id"]), ""))
            valid += is_valid(p)
            p = p or {}

            for fld in self.ENTITY_FIELDS:
                pv = [norm(x) for x in (p.get(fld) or []) if isinstance(x, (str, int, float))]
                gv = [norm(x) for x in (g.get(fld) or [])]
                tp, fp, fn = match_entities(pv, gv)
                ent.update(tp=tp, fp=fp, fn=fn)

            tp, fp, fn = match_tuples(
                [self._event(e) for e in (p.get("financial_events") or []) if isinstance(e, dict)],
                [self._event(e) for e in (g.get("financial_events") or []) if isinstance(e, dict)])
            evt.update(tp=tp, fp=fp, fn=fn)

            tp, fp, fn = match_tuples(
                [self._metric(x) for x in (p.get("financial_metrics") or []) if isinstance(x, dict)],
                [self._metric(x) for x in (g.get("financial_metrics") or []) if isinstance(x, dict)])
            met.update(tp=tp, fp=fp, fn=fn)

            chrfs.append(chrf(str(p.get("translation") or ""), str(g.get("translation") or "")))

        n = len(rows) or 1
        return {
            "name": name,
            "n_samples": len(rows),
            "schema_valid": round(valid / n, 4),
            "entity_f1": round(f1(ent["tp"], ent["fp"], ent["fn"]), 4),
            "event_f1": round(f1(evt["tp"], evt["fp"], evt["fn"]), 4),
            "metric_f1": round(f1(met["tp"], met["fp"], met["fn"]), 4),
            "translation_chrf": round(sum(chrfs) / n, 2),
        }

    @staticmethod
    def _event(e: dict):
        return (norm(e.get("event_type")), norm(e.get("company")) or None,
                num(e.get("percentage")))

    @staticmethod
    def _metric(x: dict):
        return (norm(x.get("metric")), num(x.get("value")), currency(x.get("currency")))

    # Reporting

    def report(self, r: dict) -> str:
        lines = [f"# {r['name']} (n={r['n_samples']})", "",
                 "| Metric | Value |", "|---|---:|"]
        for k in self.METRICS:
            lines.append(f"| {k} | {r[k]:.2f} |" if k.endswith("chrf")
                         else f"| {k} | {r[k]:.4f} |")
        return "\n".join(lines)

    def compare(self, base: dict, ft: dict) -> str:
        lines = [f"# {base['name']} vs {ft['name']}", "",
                 "| Metric | Base | Fine-tuned | Δ | Rel. % |", "|---|---:|---:|---:|---:|"]
        for k in self.METRICS:
            b, f = base[k], ft[k]
            rel = f"{100 * (f - b) / abs(b):+.1f}" if abs(b) > 1e-9 else "n/a"
            fmt = "{:.2f}" if k.endswith("chrf") else "{:.4f}"
            lines.append(f"| {k} | {fmt.format(b)} | {fmt.format(f)} | {f - b:+.4f} | {rel} |")
        lines.append("\n_Rel. % is meaningless when the base score is ~0; read Δ there._")
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--base-pred")
    args = ap.parse_args()

    ev = FinancialIntelligenceEvaluator()
    ft = ev.evaluate(args.gold, args.pred, "fine-tuned")
    print(ev.report(ft))
    if args.base_pred:
        print("\n" + ev.compare(ev.evaluate(args.gold, args.base_pred, "base"), ft))
