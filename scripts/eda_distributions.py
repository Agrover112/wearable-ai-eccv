#!/usr/bin/env python
"""EgoLongQA — distributions behind the Step 3-5 summaries.

Publication-quality (ECCV / LNCS) via scripts/pubstyle.py:
  - Okabe-Ito colorblind-safe palette, serif fonts, no chartjunk
  - every figure saved as PDF (vector) + PNG
  - every table saved as CSV + LaTeX booktabs

Produces (under outputs/eda/):
  figures/dist_question_length.{pdf,png}      (single column)
  figures/dist_first_words.{pdf,png}          (single column)
  figures/dist_temporal_cues.{pdf,png}        (single column)
  figures/dist_option_length_rank.{pdf,png}   (single column)
  figures/eda_distributions.{pdf,png}         (2x2 combined, double column)
  tables/dist_*.{csv,tex}
Also prints the bucketed counts.
"""
import os, re, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, COL, DBL

OUT = Path("outputs/eda"); (OUT/"figures").mkdir(parents=True, exist_ok=True); (OUT/"tables").mkdir(exist_ok=True)

def load():
    pq = hf_hub_download("facebook/wearable-ai","egolongqa/val/0000.parquet",
        repo_type="dataset",revision="refs/convert/parquet",token=os.environ.get("HF_TOKEN") or True)
    return pd.read_parquet(pq)

def parse(opts):
    d={}
    for p in re.split(r"(?=[A-D]\.\s)", str(opts).strip()):
        p=p.strip()
        if len(p)>=2 and p[0] in "ABCD" and p[1]==".": d[p[0]]=p[2:].strip()
    return d

# ---- individual panel renderers (reused for standalone + combined) ----------
def panel_qlen(ax, df):
    qlen=df.question.str.len()
    bins=list(range(50,375,25))
    ax.hist(qlen,bins=bins,color=CB[0],edgecolor="white",linewidth=0.5)
    ax.axvline(qlen.mean(),color=CB[3],ls="--",lw=1.2,label=f"mean {qlen.mean():.0f}")
    ax.set(title="Question length",xlabel="characters",ylabel="# questions"); ax.legend()
    return qlen

def panel_firstwords(ax, fw):
    top=fw.head(10)[::-1]
    ax.barh(top.index,top.values,color=CB[2])
    ax.set(title="Question first word (top 10)",xlabel="# questions")

def panel_temporal(ax, ser):
    t=ser[::-1]
    ax.barh(t.index,t.values,color=CB[3])
    ax.set(title="Temporal cue frequency",xlabel="# occurrences")

def panel_optrank(ax, rc, n, LABELS):
    ax.bar([LABELS[i] for i in rc.index],rc.values,color=CB[9])
    ax.axhline(n/4,color="0.4",ls="--",lw=1.0,label="no bias (25%)")
    ax.set(title="Correct answer: longest or shortest option?",
           xlabel="correct option's length (vs other 3)",ylabel="# questions"); ax.legend()
    ax.tick_params(axis="x",labelrotation=15)

def main():
    set_style()
    df=load()

    # (1) QUESTION LENGTH ---------------------------------------------------
    qlen=df.question.str.len(); bins=list(range(50,375,25))
    hist=pd.cut(qlen,bins=bins).value_counts().sort_index()
    tbl=pd.DataFrame({"length_range":[f"{int(iv.left)}-{int(iv.right)}" for iv in hist.index],
                      "count":hist.values})
    tbl.to_csv(OUT/"tables/dist_question_length.csv",index=False)
    df_to_booktabs(tbl,OUT/"tables/dist_question_length.tex",float_fmt="%.0f",
        caption="EgoLongQA question length distribution (characters).",label="tab:qlen")
    print("=== 1. QUESTION LENGTH (chars) ===")
    for iv,nn in hist.items(): print(f"  {int(iv.left):>3}-{int(iv.right):<3}: {nn}")
    print(f"  mean {qlen.mean():.0f} | median {qlen.median():.0f} | p90 {qlen.quantile(.9):.0f}")

    # (2) FIRST WORDS -------------------------------------------------------
    fw=df.question.str.strip().str.split().str[0].str.lower().value_counts()
    tbl=fw.head(15).rename_axis("first_word").reset_index(name="count")
    tbl.to_csv(OUT/"tables/dist_first_words.csv",index=False)
    df_to_booktabs(tbl,OUT/"tables/dist_first_words.tex",float_fmt="%.0f",
        caption="Most frequent question-opening words.",label="tab:firstwords")
    print("\n=== 2. FIRST WORDS (top 12) ===")
    for w,nn in fw.head(12).items(): print(f"  {w:<10}{nn}")

    # (3) TEMPORAL CUES -----------------------------------------------------
    cues=["after","before","first","then","next","while","during","earlier","last","finally","once","prior"]
    pat=re.compile(r"\b("+"|".join(cues)+r")\b",re.I)
    hits=Counter()
    for q in df.question: hits.update(m.lower() for m in pat.findall(q))
    ser=pd.Series(dict(hits)).sort_values(ascending=False)
    tbl=ser.rename_axis("cue").reset_index(name="count")
    tbl.to_csv(OUT/"tables/dist_temporal_cues.csv",index=False)
    df_to_booktabs(tbl,OUT/"tables/dist_temporal_cues.tex",float_fmt="%.0f",
        caption="Temporal-ordering cue frequency in questions.",label="tab:temporal")
    print("\n=== 3. TEMPORAL CUES ===")
    for w,nn in ser.items(): print(f"  {w:<10}{nn}")

    # (4) OPTION-LENGTH RANK OF CORRECT ANSWER ------------------------------
    ranks=[]
    for _,r in df.iterrows():
        d=parse(r.mcq_options); g=str(r.mcq_answer).strip().upper()
        if len(d)<2 or g not in d: continue
        order=sorted(d,key=lambda k:len(d[k]),reverse=True)  # longest first
        ranks.append(order.index(g)+1)
    LABELS={1:"longest",2:"2nd longest",3:"3rd longest",4:"shortest"}
    rc=pd.Series(ranks).value_counts().sort_index()
    tbl=pd.DataFrame({"correct_option_length":[LABELS[i] for i in rc.index],
                      "count":rc.values,"percent":[100*v/len(ranks) for v in rc.values]})
    tbl.to_csv(OUT/"tables/dist_option_length_rank.csv",index=False)
    df_to_booktabs(tbl,OUT/"tables/dist_option_length_rank.tex",float_fmt="%.1f",
        caption="Is the correct MCQ option the longest or shortest? "
                "Uniform (25\\% each) implies no length shortcut.",label="tab:optlen")
    print("\n=== 4. CORRECT ANSWER — is it the longest or shortest option? ===")
    for k,nn in rc.items(): print(f"  {LABELS[k]:<12}: {nn}  ({100*nn/len(ranks):.1f}%)")
    print(f"  (uniform over 4 options => 25% each; SHORTEST = {100*rc.get(4,0)/len(ranks):.1f}%)")

    # ---- standalone single-column figures ---------------------------------
    fig,ax=plt.subplots(figsize=(COL,COL*0.8)); panel_qlen(ax,df)
    savefig(fig,OUT/"figures/dist_question_length")
    fig,ax=plt.subplots(figsize=(COL,COL*0.8)); panel_firstwords(ax,fw)
    savefig(fig,OUT/"figures/dist_first_words")
    fig,ax=plt.subplots(figsize=(COL,COL*0.8)); panel_temporal(ax,ser)
    savefig(fig,OUT/"figures/dist_temporal_cues")
    fig,ax=plt.subplots(figsize=(COL,COL*0.85)); panel_optrank(ax,rc,len(ranks),LABELS)
    savefig(fig,OUT/"figures/dist_option_length_rank")

    # ---- combined 2x2 (double column) -------------------------------------
    fig,axes=plt.subplots(2,2,figsize=(DBL,DBL*0.72))
    panel_qlen(axes[0,0],df); panel_firstwords(axes[0,1],fw)
    panel_temporal(axes[1,0],ser); panel_optrank(axes[1,1],rc,len(ranks),LABELS)
    fig.tight_layout()
    savefig(fig,OUT/"figures/eda_distributions")
    print(f"\n>> Wrote figures/dist_*.{{pdf,png}} + eda_distributions.{{pdf,png}} + tables/dist_*.{{csv,tex}}")

if __name__=="__main__":
    main()
