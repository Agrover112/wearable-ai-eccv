#!/usr/bin/env python
"""EgoLongQA data analysis — Steps 1-5 (text/labels only, no video, no models).

Reads the egolongqa/val annotations (parquet export) and writes:
  outputs/eda/report.md
  outputs/eda/tables/*.csv
"""
import os, re
from pathlib import Path
import pandas as pd
from collections import Counter
from huggingface_hub import hf_hub_download

OUT = Path("outputs/eda"); (OUT / "tables").mkdir(parents=True, exist_ok=True)
lines = []  # report.md accumulator
def emit(s=""):
    print(s); lines.append(s)

def main():
    pq = hf_hub_download("facebook/wearable-ai", "egolongqa/val/0000.parquet",
                         repo_type="dataset", revision="refs/convert/parquet",
                         token=os.environ.get("HF_TOKEN") or True)
    df = pd.read_parquet(pq)
    emit(f"# EgoLongQA — EDA (Steps 1-5)\n")
    emit(f"**Rows:** {len(df)}  |  **Columns:** {', '.join(df.columns)}\n")

    # 1. STRUCTURE
    emit("## 1. Structure")
    qpv = df.groupby("video_path").size()
    emit(f"- Unique videos: **{df.video_path.nunique()}**")
    emit(f"- Questions per video: min {qpv.min()}, max {qpv.max()}, mean {qpv.mean():.2f}")
    cat = df.category.value_counts()
    cat.to_csv(OUT / "tables/category_counts.csv", header=["count"])
    emit(f"- Categories: **{df.category.nunique()}**\n")
    emit("| category | count | % |\n|---|---:|---:|")
    for c, n in cat.items():
        emit(f"| {c} | {n} | {100*n/len(df):.1f}% |")
    emit("")

    # 2. ANSWER BALANCE
    emit("## 2. MCQ answer balance")
    lb = df.mcq_answer.str.strip().str.upper().value_counts().sort_index()
    lb.to_csv(OUT / "tables/mcq_letter_bias.csv", header=["count"])
    emit("| letter | count | % |\n|---|---:|---:|")
    for l, n in lb.items():
        emit(f"| {l} | {n} | {100*n/len(df):.1f}% |")
    emit(f"\n- Most common: **{lb.idxmax()}** ({100*lb.max()/len(df):.1f}%) — uniform would be 25%")
    n_opts = df.mcq_options.apply(lambda s: len(re.findall(r"\b[A-D]\.\s", str(s))))
    emit(f"- Options per question: {n_opts.min()}-{n_opts.max()} (mode {n_opts.mode().iloc[0]})\n")

    # 3. TEXT STATS
    emit("## 3. Text stats (characters)")
    emit("| field | mean | min | max |\n|---|---:|---:|---:|")
    for c in ["question", "answer", "mcq_options"]:
        s = df[c].astype(str).str.len()
        emit(f"| {c} | {s.mean():.0f} | {s.min()} | {s.max()} |")
    fw = df.question.str.strip().str.split().str[0].str.lower().value_counts().head(8)
    emit("\n- Question first words: " + ", ".join(f"{w} ({n})" for w, n in fw.items()) + "\n")

    # 4. TEMPORAL REASONING
    emit("## 4. Temporal-reasoning share")
    cues = ["after","before","first","then","next","while","during","earlier","last","finally","once","prior"]
    pat = re.compile(r"\b(" + "|".join(cues) + r")\b", re.I)         # capturing, for findall
    pat_search = re.compile(r"\b(?:" + "|".join(cues) + r")\b", re.I)  # non-capturing, for contains
    temporal = df.question.str.contains(pat_search)
    emit(f"- Temporal questions: **{temporal.sum()} / {len(df)} = {100*temporal.mean():.1f}%**")
    hits = Counter()
    for q in df.question: hits.update(m.lower() for m in pat.findall(q))
    emit("- Top cues: " + ", ".join(f"{w} ({n})" for w, n in hits.most_common(6)) + "\n")

    # 5. SHORTCUT CHECK — where does the correct option fall by length?
    emit("## 5. Shortcut check — option length")
    def parse(opts):
        parts = re.split(r"(?=[A-D]\.\s)", str(opts).strip())
        d = {}
        for p in parts:
            p = p.strip()
            if len(p) >= 2 and p[0] in "ABCD" and p[1] == ".":
                d[p[0]] = p[2:].strip()
        return d
    LAB = {1:"longest",2:"2nd longest",3:"3rd longest",4:"shortest"}
    ranks, short_letters = [], []
    for _, r in df.iterrows():
        d = parse(r.mcq_options); gold = str(r.mcq_answer).strip().upper()
        if len(d) < 2 or gold not in d: continue
        order = sorted(d, key=lambda k: len(d[k]), reverse=True)  # longest first
        rk = order.index(gold) + 1
        ranks.append(rk)
        if rk == len(d): short_letters.append(gold)  # correct == shortest
    rc = pd.Series(ranks).value_counts().sort_index(); n = len(ranks)
    emit("Where the **correct** answer falls when options are sorted by length:\n")
    emit("| correct option is… | count | % |\n|---|---:|---:|")
    for k in sorted(rc.index):
        emit(f"| {LAB.get(k,k)} | {rc[k]} | {100*rc[k]/n:.1f}% |")
    short_pct = 100*rc.get(4,0)/n
    emit(f"\n- ⚠️ **Reverse length shortcut:** the correct answer is the **shortest** option "
         f"**{short_pct:.1f}%** of the time (vs 25% by chance).")
    emit(f"  - A blind 'pick the shortest option' baseline scores ~{short_pct:.0f}%.")
    emit(f"  - (An earlier 'is it the *longest*?' check missed this — the bias runs toward short.)")
    # letter breakdown of shortest-correct → is it just the C bias?
    sl = pd.Series(short_letters).value_counts()
    c_share = 100*sl.get("C",0)/max(len(short_letters),1)
    emit(f"- Of shortest-correct answers, **{sl.get('C',0)}/{len(short_letters)} = {c_share:.1f}% are 'C'** "
         f"≈ C's overall {100*(df.mcq_answer.str.strip().str.upper()=='C').mean():.1f}% rate.")
    emit(f"  - ⇒ the **length** and **letter (C)** shortcuts are ~independent — two separate leaks.\n")

    emit("## Summary — exploitable shortcuts (video-blind)")
    emit("| shortcut | blind-baseline score |\n|---|---:|")
    emit(f"| Always answer **C** | {100*(df.mcq_answer.str.strip().str.upper()=='C').mean():.1f}% |")
    emit(f"| Always pick **shortest** option | {short_pct:.1f}% |")
    emit("(random chance = 25%)\n")

    (OUT / "report.md").write_text("\n".join(lines))
    print(f"\n>> Wrote {OUT/'report.md'} + tables/")

if __name__ == "__main__":
    main()
