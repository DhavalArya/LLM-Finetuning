#!/usr/bin/env python3
"""
01_make_payloads.py
Generate synthetic payloads by perturbing a seed payload, including edge cases.
Outputs JSONL with one payload per line.

Usage:
  python scripts/01_make_payloads.py \
      --seed data/seed.json \
      --out data/payloads.jsonl \
      --n 500 \
      --edge-cases \
      --extreme-rate 0.15 \
      --rng-seed 42
"""

import argparse
import copy
import json
import math
import random
from datetime import datetime, timedelta

FX = {
    ("USD","EUR"): 0.92,
    ("EUR","USD"): 1.09,
    ("USD","INR"): 84.0,
    ("INR","USD"): 1/84.0,
    ("EUR","INR"): 91.0,
    ("INR","EUR"): 1/91.0,
}
CURRENCIES = ["USD","EUR","INR"]

def clamp(x, lo, hi): return max(lo, min(hi, x))

def rand_scale(mu=0.0, sigma=0.15):
    g = random.gauss(mu, sigma)
    return clamp(math.exp(g), 0.5, 1.6)

def shift_date(s, days=(-90, 120)):
    try:
        dt = datetime.strptime(s, "%m/%d/%Y")
    except Exception:
        return s
    d = random.randint(days[0], days[1])
    return (dt + timedelta(days=d)).strftime("%m/%d/%Y")

def convert_currency(val, src, dst):
    if src == dst: return val
    rate = FX.get((src, dst), 1.0)
    return val * rate

def fix_monthwise_sum(spend_year):
    months = spend_year.get("FYMonthWiseSpendData", [])
    s = sum(max(0.0, float(m.get("spend", 0.0)))
            for m in months
            if str(m.get("year")) == str(spend_year.get("year")))
    spend_year["totalFYSpend"] = round(s * random.uniform(0.985, 1.015), 2)

def perturb_spend_block(block, extreme=False):
    y = copy.deepcopy(block)
    if "FYBudget" in y:
        if extreme and random.random() < 0.25:
            y["FYBudget"] = 0.0
        else:
            y["FYBudget"] = round(max(0.0, float(y["FYBudget"]) * rand_scale()), 2)
    if "FYBudgetYTD" in y:
        y["FYBudgetYTD"] = round(max(0.0, float(y["FYBudgetYTD"]) * rand_scale()), 2)
    months = y.get("FYMonthWiseSpendData", [])
    for m in months:
        if extreme and random.random() < 0.05:
            try:
                m["year"] = str(int(y.get("year", m.get("year", "2025"))) + random.choice([-2,-1,1]))
            except Exception:
                pass
        base = max(0.0, float(m.get("spend", 0.0)))
        scale = rand_scale(sigma=0.25 if extreme else 0.12)
        m["spend"] = round(base * scale, 2)
        m["month"] = int(clamp(int(m.get("month", 1)), 1, 12))
    fix_monthwise_sum(y)
    if "FYPercent" in y:
        try:
            y["FYPercent"] = round(float(y["FYPercent"]) * rand_scale(0.0, 0.2), 4)
        except Exception:
            y["FYPercent"] = 0.0
    return y

def perturb_savings(s, extreme=False):
    o = copy.deepcopy(s)
    for row in o:
        if random.random() < 0.9:
            row["sumToShow"] = max(0, int(float(row.get("sumToShow", 0)) * rand_scale(sigma=0.25 if extreme else 0.12)))
        if "actual" in row:
            if extreme and random.random() < 0.2:
                row["actualFlag"] = False
                row["projection"] = row.get("projection", 0) + int(abs(random.gauss(0, 500000)))
                row["actual"] = 0
            else:
                row["actualFlag"] = True
                row["actual"] = max(0, int(float(row.get("actual", 0)) * rand_scale()))
    return o

def perturb_contracts(c_list, extreme=False, flip_currency=True):
    res = []
    for c in c_list:
        cc = copy.deepcopy(c)
        if flip_currency and random.random() < (0.5 if extreme else 0.25):
            dst = random.choice(CURRENCIES)
            src = cc.get("currency", {}).get("code", "USD")
            cc["currency"] = {"code": dst, "name": {"USD":"US Dollar","EUR":"Euro","INR":"Indian Rupee"}.get(dst, dst)}
            cc["value"] = round(convert_currency(float(cc.get("value", 0.0)), src, dst), 2)
        if "startDate" in cc:
            cc["startDate"] = shift_date(cc["startDate"], days=(-180, 180 if extreme else 90))
        if "endDate" in cc:
            cc["endDate"] = shift_date(cc["endDate"], days=(-180, 240 if extreme else 120))
        if random.random() < (0.2 if extreme else 0.1):
            cc["value"] = 0.0
        else:
            cc["value"] = round(max(0.0, float(cc.get("value", 0.0)) * rand_scale(sigma=0.35 if extreme else 0.18)), 2)
        pe = cc.get("priceEvolution", {})
        def j(s):
            if not s: return s
            try:
                sign = -1 if "-" in s else 1
                num = float(s.replace("%","").replace("+","").replace("-",""))
                num = num * clamp(random.gauss(1.0, 0.25 if extreme else 0.12), 0.4, 1.8)
                return f"{'+' if sign>0 else '-'}{abs(round(num, 1))}%"
            except Exception:
                return s
        for k in list(pe.keys()):
            pe[k] = j(pe[k])
        cc["priceEvolution"] = pe
        res.append(cc)
    return res

def perturb_alerts(alerts, extreme=False):
    out = []
    for a in alerts:
        aa = copy.deepcopy(a)
        if random.random() < (0.25 if extreme else 0.12):
            if "Increased" in aa["alertStatement"]:
                aa["alertStatement"] = aa["alertStatement"].replace("Increased", "Decreased")
            elif "Decreased" in aa["alertStatement"]:
                aa["alertStatement"] = aa["alertStatement"].replace("Decreased", "Increased")
        import re
        def repl(m):
            num = float(m.group(1))
            num = max(0.1, num * rand_scale(sigma=0.3 if extreme else 0.15))
            return f"{round(num, 2)}"
        aa["alertStatement"] = re.sub(r"([0-9]+(?:\.[0-9]+)?)%", lambda m: repl(m) + "%", aa["alertStatement"])
        out.append(aa)
    if extreme and len(out) > 0:
        if random.random() < 0.15: out = out[:-1]
        if random.random() < 0.15: out.append(copy.deepcopy(random.choice(out)))
    return out

def perturb_summary_cards(cards, extreme=False):
    c = copy.deepcopy(cards)
    for k in list(c.keys()):
        if random.random() < (0.25 if extreme else 0.12):
            c[k] = 0
        else:
            try:
                c[k] = int(max(0, int(c[k]) * random.uniform(0.5, 1.6)))
            except Exception:
                pass
    return c

def perturb_strategy_info(si, extreme=False):
    o = copy.deepcopy(si)
    for k in ["sumOfPotentialSavings","sumOfRealizedSaving","percentChangeStrategySaving"]:
        if k in o:
            try:
                base = float(o[k])
                if k == "percentChangeStrategySaving":
                    o[k] = round(base * clamp(random.gauss(1.0, 0.35 if extreme else 0.2), 0.3, 1.8), 4)
                else:
                    o[k] = round(max(0.0, base * rand_scale(sigma=0.35 if extreme else 0.2)), 2)
            except Exception:
                pass
    for st in o.get("strategiesData", []):
        sd = st.get("strategyDetail", {})
        if "executionStartDate" in sd:
            sd["executionStartDate"] = shift_date(sd["executionStartDate"], days=(-90, 120))
        if "executionEndDate" in sd:
            sd["executionEndDate"] = shift_date(sd["executionEndDate"], days=(-90, 180))
        if "addressableSpend" in sd:
            sd["addressableSpend"] = round(max(0.0, float(sd["addressableSpend"]) * rand_scale()), 2)
        if "percentSavings" in sd:
            try:
                sd["percentSavings"] = round(max(0.0, float(sd["percentSavings"]) * clamp(random.gauss(1.0, 0.25), 0.5, 1.8)), 2)
            except Exception:
                pass
        if "savingsPotential" in sd:
            sd["savingsPotential"] = round(max(0.0, float(sd["savingsPotential"]) * rand_scale()), 2)
        st["strategyDetail"] = sd
        for pr in st.get("projects", []):
            if "savings" in pr and isinstance(pr["savings"], (int,float)):
                pr["savings"] = round(max(0.0, float(pr["savings"]) * rand_scale()), 2)
    return o

def introduce_edge_cases(payload):
    p = copy.deepcopy(payload)
    if "summaryCardsData" in p and random.random() < 0.3:
        for k in p["summaryCardsData"].keys():
            if random.random() < 0.5:
                p["summaryCardsData"][k] = 0
    if "externalMarketIntelligenceData" in p and random.random() < 0.25:
        em = p["externalMarketIntelligenceData"]
        for k in ["description","priceOutlook","currentRisk"]:
            if random.random() < 0.4:
                em[k] = ""
    # Instead of deleting keys, blank them as empty lists to keep schema stable
    if random.random() < 0.1:
        p["marketIndicesAlerts"] = []
    if random.random() < 0.1:
        p["savingsData"] = []
    return p

def harmonize_currency(payload, target=None):
    p = copy.deepcopy(payload)
    cur = p.get("currency", "USD")
    dst = target or random.choice(CURRENCIES)
    if dst != cur: p["currency"] = dst
    return p

def generate_variant(seed, extreme=False, flip_currency=True, edge_cases=False):
    x = copy.deepcopy(seed)
    if flip_currency and random.random() < (0.35 if extreme else 0.2):
        x = harmonize_currency(x)
    x["spendData"] = [perturb_spend_block(b, extreme=extreme) for b in seed.get("spendData", [])]
    if "savingsData" in seed:
        x["savingsData"] = perturb_savings(seed["savingsData"], extreme=extreme)
    if "contractsData" in seed:
        x["contractsData"] = perturb_contracts(seed["contractsData"], extreme=extreme, flip_currency=True)
    if "summaryCardsData" in seed:
        x["summaryCardsData"] = perturb_summary_cards(seed["summaryCardsData"], extreme=extreme)
    if "externalMarketIntelligenceData" in seed:
        em = copy.deepcopy(seed["externalMarketIntelligenceData"])
        if random.random() < 0.4: em["priceOutlook"] = random.choice(["Stable","Rising","Falling","Volatile","Balanced"])
        if random.random() < 0.35: em["currentRisk"] = random.choice(["Stable","Elevated","High","Moderate"])
        x["externalMarketIntelligenceData"] = em
    if "marketIndicesAlerts" in seed:
        x["marketIndicesAlerts"] = perturb_alerts(seed["marketIndicesAlerts"], extreme=extreme)
    if "strategyInfo" in seed:
        x["strategyInfo"] = perturb_strategy_info(seed["strategyInfo"], extreme=extreme)
    if edge_cases:
        x = introduce_edge_cases(x)

    # Fresh IDs
    import uuid
    x["transaction_id"] = str(uuid.uuid4())
    x["documentCode"] = str(uuid.uuid4())

    # ---- enforce required fields with safe defaults ----
    if "spendData" not in x or not isinstance(x["spendData"], list): x["spendData"] = []
    if "savingsData" not in x or not isinstance(x["savingsData"], list): x["savingsData"] = []
    if "contractsData" not in x or not isinstance(x["contractsData"], list): x["contractsData"] = []
    if "marketIndicesAlerts" not in x or not isinstance(x["marketIndicesAlerts"], list): x["marketIndicesAlerts"] = []
    if "summaryCardsData" not in x or not isinstance(x["summaryCardsData"], dict): x["summaryCardsData"] = {}

    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, help="Path to seed payload JSON")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--n", type=int, default=500, help="Number of variants to generate")
    ap.add_argument("--extreme-rate", type=float, default=0.15, help="Fraction with extreme perturbations")
    ap.add_argument("--edge-cases", action="store_true", help="Introduce zeros/empty/missing keys occasionally")
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.rng_seed)
    with open(args.seed, "r", encoding="utf-8") as f:
        seed = json.load(f)

    total = args.n
    extreme_n = int(total * args.extreme_rate)
    with open(args.out, "w", encoding="utf-8") as w:
        for i in range(total):
            extreme = (i < extreme_n)
            v = generate_variant(seed, extreme=extreme, edge_cases=args.edge_cases, flip_currency=True)
            w.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"Wrote {total} payloads to {args.out}")

if __name__ == "__main__":
    main()
