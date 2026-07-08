#!/usr/bin/env python
"""EgoLongQA — distributions behind the Step 3-5 summaries.

Produces (under outputs/eda/):
  figures/dist_question_length.png
  figures/dist_first_words.png
  figures/dist_temporal_cues.png
  figures/dist_option_length_rank.png
  figures/eda_distributions.png            (2x2 combined)
  tables/dist_*.csv
Also prints the bucketed counts.
"""
import os, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from huggingface_hub import hf_hub_download

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

def main():
    df=load()
    fig,axes=plt.subplots(2,2,figsize=(14,10)); fig.suptitle("EgoLongQA — text distributions (n=700)",fontsize=14)

    # (1) QUESTION LENGTH (chars) -------------------------------------------
    qlen=df.question.str.len()
    bins=list(range(50,375,25))
    hist=pd.cut(qlen,bins=bins).value_counts().sort_index()
    hist.to_csv(OUT/"tables/dist_question_length.csv",header=["count"])
    print("=== 1. QUESTION LENGTH (chars) ===")
    for iv,n in hist.items(): print(f"  {int(iv.left):>3}-{int(iv.right):<3}: {n}")
    print(f"  mean {qlen.mean():.0f} | median {qlen.median():.0f} | p90 {qlen.quantile(.9):.0f}")
    ax=axes[0,0]; ax.hist(qlen,bins=bins,color="#4C72B0",edgecolor="white")
    ax.axvline(qlen.mean(),color="crimson",ls="--",label=f"mean {qlen.mean():.0f}")
    ax.set(title="Question length",xlabel="characters",ylabel="# questions"); ax.legend()

    # (2) FIRST WORDS -------------------------------------------------------
    fw=df.question.str.strip().str.split().str[0].str.lower().value_counts()
    fw.head(15).to_csv(OUT/"tables/dist_first_words.csv",header=["count"])
    print("\n=== 2. FIRST WORDS (top 12) ===")
    for w,n in fw.head(12).items(): print(f"  {w:<10}{n}")
    top=fw.head(10)[::-1]
    ax=axes[0,1]; ax.barh(top.index,top.values,color="#55A868")
    ax.set(title="Question first word (top 10)",xlabel="# questions")

    # (3) TEMPORAL CUES -----------------------------------------------------
    cues=["after","before","first","then","next","while","during","earlier","last","finally","once","prior"]
    pat=re.compile(r"\b("+"|".join(cues)+r")\b",re.I)
    hits=Counter()
    for q in df.question: hits.update(m.lower() for m in pat.findall(q))
    ser=pd.Series(dict(hits)).sort_values(ascending=False)
    ser.to_csv(OUT/"tables/dist_temporal_cues.csv",header=["count"])
    print("\n=== 3. TEMPORAL CUES ===")
    for w,n in ser.items(): print(f"  {w:<10}{n}")
    t=ser[::-1]
    ax=axes[1,0]; ax.barh(t.index,t.values,color="#C44E52")
    ax.set(title="Temporal cue frequency",xlabel="# occurrences")

    # (4) OPTION-LENGTH RANK OF CORRECT ANSWER ------------------------------
    # rank 1 = correct option is the LONGEST. Uniform => no length shortcut.
    ranks=[]
    for _,r in df.iterrows():
        d=parse(r.mcq_options); g=str(r.mcq_answer).strip().upper()
        if len(d)<2 or g not in d: continue
        order=sorted(d,key=lambda k:len(d[k]),reverse=True)  # longest first
        ranks.append(order.index(g)+1)
    LABELS={1:"longest",2:"2nd longest",3:"3rd longest",4:"shortest"}
    rc=pd.Series(ranks).value_counts().sort_index()
    rc_lab=rc.rename(index=LABELS); rc_lab.index.name="correct_option_length"
    rc_lab.to_csv(OUT/"tables/dist_option_length_rank.csv",header=["count"])
    print("\n=== 4. CORRECT ANSWER — is it the longest or shortest option? ===")
    for k,n in rc.items(): print(f"  {LABELS[k]:<12}: {n}  ({100*n/len(ranks):.1f}%)")
    print(f"  (uniform over 4 options => 25% each; SHORTEST = {100*rc.get(4,0)/len(ranks):.1f}%)")
    ax=axes[1,1]; ax.bar([LABELS[i] for i in rc.index],rc.values,color="#8172B3")
    ax.axhline(len(ranks)/4,color="gray",ls="--",label="no bias (25%)")
    ax.set(title="Is the CORRECT answer the longest or shortest option?",
           xlabel="correct option's length (vs the other 3)",ylabel="# questions"); ax.legend()
    ax.tick_params(axis="x",labelrotation=15)

    # save
    for ax_name,ax in [("question_length",axes[0,0]),("first_words",axes[0,1]),
                       ("temporal_cues",axes[1,0]),("option_length_rank",axes[1,1])]:
        pass
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(OUT/"figures/eda_distributions.png",dpi=110)
    print(f"\n>> Wrote {OUT}/figures/eda_distributions.png + tables/dist_*.csv")

if __name__=="__main__":
    main()
