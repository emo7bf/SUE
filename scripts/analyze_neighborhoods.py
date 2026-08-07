"""One-shot analyses answering three questions about the corpus:

1. TEL vs KLA: do TEL's reports cover the same ESG topics as KLA's and
   additionally carry financial-disclosure content? Regex-count both
   sides per-company.
2. GMM cluster composition: cross-tab the 2-component GMM assignment
   against company. Where does KLA (green) land relative to TEL
   (purple)? If KLA's content is a subset of TEL's, the two firms should
   have similar tabular/prose ratios and share GMM cluster membership.
3. Company-diverse embedding neighborhoods: for K=5, find the
   neighborhoods that span the most distinct companies. Dump a few of
   them so we can eyeball whether nearby chunks from different firms are
   discussing the same topic (scope emissions, water withdrawal,
   workforce demographics, ...).

Run:  python sue_release/scripts/analyze_neighborhoods.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "assets" / "chunks.parquet"
EMB    = ROOT / "assets" / "embeddings.npy"

df = pd.read_parquet(CHUNKS)
X  = np.load(EMB).astype(np.float32)
print(f"loaded {len(df):,} chunks, embeddings shape {X.shape}")
print("companies:", sorted(df["company"].unique()))
print()

# ------------------------------------------------------------------ 1
# Regex topic-panel counts per company.  Case-insensitive; a chunk counts
# for a topic if any of that topic's phrases appears in its text.
ESG_TOPICS = {
    "scope_emissions":  r"\bscope\s*[123]\b|market[- ]based|location[- ]based",
    "ghg_totals":       r"greenhouse gas|\bghg\b|\bco2e?\b|carbon (?:dioxide|footprint|neutrality|intensity)",
    "water":            r"water withdrawal|water consumption|water recycled|water intensity|freshwater",
    "energy":           r"renewable energy|energy consumption|energy intensity|kilowatt|megawatt|\bmwh\b|\bkwh\b|\bgwh\b",
    "waste":            r"waste diverted|hazardous waste|landfill|recycling rate",
    "workforce":        r"workforce|headcount|full[- ]time equivalent|voluntary turnover|women in (?:leadership|management)",
    "safety":           r"recordable (?:injury|incident)|lost[- ]time|\btrir\b|\bltir\b|fatalities?",
    "diversity":        r"diversity, equity|\bdei\b|underrepresented|racial|gender pay",
    "governance":       r"board of directors|board (?:composition|independence|oversight)|governance framework",
    "supply_chain":     r"supplier code|supply chain (?:audit|due diligence|risk)|conflict minerals|responsible sourcing",
    "ethics":           r"code of (?:ethics|conduct|business)|anti[- ]corruption|whistleblower|human rights",
    "materiality":      r"materiality (?:assessment|matrix)|double materiality|priority topics",
}

FIN_TOPICS = {
    "revenue":          r"\bnet (?:sales|revenue)|total (?:sales|revenue)|\brevenue for (?:the |fiscal )",
    "gross_profit":     r"gross profit|cost of (?:sales|goods sold)|gross margin",
    "operating_income": r"operating (?:income|profit|loss)|operating margin|\bebit\b",
    "net_income":       r"net income|net earnings|earnings per share|\beps\b|diluted (?:earnings|share)",
    "cash_flow":        r"cash flows? from (?:operating|investing|financing)|free cash flow|capital expenditures",
    "balance_sheet":    r"total (?:assets|liabilities)|shareholders'? equity|retained earnings|goodwill",
    "financing":        r"share (?:buyback|repurchase)|dividends? per share|debt securities|credit facility",
    "audit_opinion":    r"opinion of the auditor|independent auditor'?s report|internal control over financial reporting",
    "risk_factors":     r"risk factors?|forward[- ]looking statements|material adverse (?:effect|impact)",
    "mda":              r"management(?:'s|\u2019s)? discussion and analysis|md&a",
    "segments":         r"segment (?:results|revenue|performance)|reportable segments",
    "currency":         r"[\u00a5\u00a3\u20ac\$]\s?\d[\d,\.]*\s?(?:million|billion|thousand)|yen\b",
}

def count_topics(topics: dict, sub: pd.DataFrame) -> dict:
    out = {}
    for name, pat in topics.items():
        rx = re.compile(pat, re.IGNORECASE)
        out[name] = int(sub["text"].str.contains(rx, na=False).sum())
    return out

by_company = df.groupby("company")
esg_table = pd.DataFrame({c: count_topics(ESG_TOPICS, g) for c, g in by_company})
fin_table = pd.DataFrame({c: count_topics(FIN_TOPICS, g) for c, g in by_company})

# Normalise to per-1000-chunks so TEL's larger corpus doesn't crowd the
# comparison.  Also emit raw counts.
chunk_counts = by_company.size()
print("Chunk counts per company:")
print(chunk_counts.to_string())
print()

def per_1k(table: pd.DataFrame) -> pd.DataFrame:
    return (table.div(chunk_counts, axis=1) * 1000).round(1)

print("=" * 72)
print("ANALYSIS 1a  \u2014  ESG topic hits per 1,000 chunks (per company)")
print("=" * 72)
print(per_1k(esg_table).to_string())
print()

print("=" * 72)
print("ANALYSIS 1b  \u2014  Financial-disclosure hits per 1,000 chunks (per company)")
print("=" * 72)
print(per_1k(fin_table).to_string())
print()

# Focused TEL vs KLA comparison.
if "TEL" in chunk_counts.index and "KLA" in chunk_counts.index:
    esg_p1k = per_1k(esg_table)[["KLA", "TEL"]]
    fin_p1k = per_1k(fin_table)[["KLA", "TEL"]]
    esg_p1k["TEL/KLA"] = (esg_p1k["TEL"] / esg_p1k["KLA"].replace(0, np.nan)).round(2)
    fin_p1k["TEL/KLA"] = (fin_p1k["TEL"] / fin_p1k["KLA"].replace(0, np.nan)).round(2)
    print("=" * 72)
    print("ANALYSIS 1c  \u2014  TEL vs KLA, side by side (per 1,000 chunks)")
    print("=" * 72)
    print("ESG topics:")
    print(esg_p1k.to_string())
    print("\nFinancial-disclosure topics:")
    print(fin_p1k.to_string())
    print()

# ------------------------------------------------------------------ 2
# Fit 2-component GMM in top-20 PC subspace and cross-tab against company.
print("=" * 72)
print("ANALYSIS 2  \u2014  GMM (2 components, top-20 PCs) cross-tab \u00d7 company")
print("=" * 72)
Z20 = PCA(n_components=20, random_state=0).fit_transform(X)
gmm = GaussianMixture(n_components=2, covariance_type="full",
                      random_state=0, n_init=3).fit(Z20)
cluster = gmm.predict(Z20)
# Label the cluster whose members have higher digit density as "tabular".
dd = df["text"].str.count(r"\d") / df["text"].str.len().clip(lower=1)
mean_dd = [dd[cluster == k].mean() for k in (0, 1)]
tabular_k = int(np.argmax(mean_dd))
label = np.where(cluster == tabular_k, "tabular", "prose")
print(f"cluster {tabular_k} identified as TABULAR "
      f"(mean digit-density {mean_dd[tabular_k]:.3f} vs "
      f"{mean_dd[1 - tabular_k]:.3f})")

ct = pd.crosstab(df["company"], pd.Series(label, name="cluster"),
                 margins=True, margins_name="total")
ct_pct = pd.crosstab(df["company"], pd.Series(label, name="cluster"),
                     normalize="index").mul(100).round(1)
print("\nRaw counts:")
print(ct.to_string())
print("\nPer-company row percentages:")
print(ct_pct.to_string())
print()

# KLA specifically: what fraction of its chunks sit in the tabular cluster
# vs how often those chunks contain financial-disclosure language.
for co in ("KLA", "TEL"):
    m = (df["company"] == co) & (label == "tabular")
    n = int(m.sum())
    if n == 0:
        continue
    print(f"[{co}] tabular-cluster chunks: {n}")
    fin_rx = re.compile(r"|".join(FIN_TOPICS.values()), re.IGNORECASE)
    esg_rx = re.compile(r"|".join(ESG_TOPICS.values()), re.IGNORECASE)
    fh = int(df.loc[m, "text"].str.contains(fin_rx, na=False).sum())
    eh = int(df.loc[m, "text"].str.contains(esg_rx, na=False).sum())
    print(f"       of which contain financial language: {fh} ({100*fh/n:.1f}%)")
    print(f"       of which contain ESG language:       {eh} ({100*eh/n:.1f}%)")
print()

# ------------------------------------------------------------------ 3
# Find K=5 neighborhoods that span the most distinct companies.  For each,
# dump a short summary of the chunks so we can eyeball their content.
print("=" * 72)
print("ANALYSIS 3  \u2014  K=5 neighborhoods spanning the most companies")
print("=" * 72)
K = 5
nn = NearestNeighbors(n_neighbors=K, metric="cosine").fit(X)
_, idx = nn.kneighbors(X)  # (n, K)  -- idx[i,0] == i

# Company diversity of each neighborhood (0..4 distinct companies besides self).
co = df["company"].to_numpy()
div = np.array([len(set(co[nbrs])) for nbrs in idx])
print(f"neighborhood-diversity histogram (# distinct companies in K=5):")
uniq, cnt = np.unique(div, return_counts=True)
for u, c in zip(uniq, cnt):
    print(f"  {u} companies : {c:>5} neighborhoods ({100*c/len(div):.1f}%)")
print()

# Pull the top max-diversity neighborhoods; among them, prefer ones whose
# center chunk contains a categorical keyword we can label a-priori.
CATEGORICAL_PROBES = [
    ("scope_emissions", r"\bscope\s*[123]\b"),
    ("water",           r"water (?:withdrawal|consumption|recycled)"),
    ("workforce",       r"(?:workforce|headcount|full[- ]time equivalent|voluntary turnover)"),
    ("safety",          r"(?:recordable|lost[- ]time|trir|ltir|fatalit)"),
    ("board",           r"board of directors|board (?:composition|independence)"),
    ("energy",          r"renewable energy|energy (?:consumption|intensity)|\bmwh\b|\bkwh\b"),
    ("supply_chain",    r"supplier code|conflict minerals|responsible sourcing"),
]

max_div = int(div.max())
candidates = np.where(div == max_div)[0]
print(f"neighborhoods hitting max diversity of {max_div}: {len(candidates)}")

# Bucket by categorical probe on the center chunk.
buckets: dict[str, list[int]] = {name: [] for name, _ in CATEGORICAL_PROBES}
for i in candidates:
    for name, pat in CATEGORICAL_PROBES:
        if re.search(pat, df.iloc[i]["text"], re.IGNORECASE):
            buckets[name].append(i)
            break

for name, ids in buckets.items():
    if not ids:
        continue
    print(f"\n--- neighborhood category: {name}  ({len(ids)} candidates) ---")
    # Show up to 2 example neighborhoods.
    for center in ids[:2]:
        print(f"\n  CENTER: [{co[center]}] {df.iloc[center]['doc']} chunk {df.iloc[center]['chunk_id']}")
        text = re.sub(r"\s+", " ", df.iloc[center]["text"]).strip()
        print(f"    \u25b8 {text[:260]}\u2026")
        for rank, j in enumerate(idx[center][1:], start=1):
            t = re.sub(r"\s+", " ", df.iloc[j]["text"]).strip()
            print(f"    #{rank} [{co[j]}] {df.iloc[j]['doc']} c{df.iloc[j]['chunk_id']}: {t[:200]}\u2026")

print("\n" + "=" * 72)
print("done.")
