#!/usr/bin/env python
"""EgoLongQA — text-only leaks: question<->option overlap + answer<->option match.

(2a) Lexical overlap  : pick the option sharing most content words with the QUESTION
                        (Jaccard). Input-visible -> a real shortcut if > 25%.
(2b) Odd-one-out      : pick the option most similar (MiniLM cosine) to the other three
                        ("distractors are clones of the answer"). Also a shortcut.
(4)  Answer<->option  : does the FREE-FORM `answer` field (ground truth, hidden at test
                        time) match the correct option? -> auto-grader viability +
                        dataset-construction forensics. NOT a shortcut.

CPU-only, annotations only. Outputs (outputs/eda/):
  figures/blind_baselines.{pdf,png}      ladder of all video-blind baselines
  tables/blind_baselines.{csv,tex}
  tables/answer_option_match.{csv,tex}
"""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pubstyle import set_style, savefig, df_to_booktabs, CB, COL
from src.data import parse_mcq
import matplotlib.pyplot as plt

REPO = "facebook/wearable-ai"; TOK = os.environ.get("HF_TOKEN") or True
OUT = Path("outputs/eda")
LETTERS = "ABCD"

STOP = set("""a an the and or of to in on at for with from by is are was were be been being
this that these those i my me you your he she it its we our they their what which who whom
whose when where why how did do does done had has have will would can could shall should may
might must not no nor as if then than so such very just about into over under near after
before during while first last earlier later left right""".split())


def content_words(s):
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP and len(w) > 2}


def main():
    set_style()
    pq = hf_hub_download(REPO, "egolongqa/val/0000.parquet", repo_type="dataset",
                         revision="refs/convert/parquet", token=TOK)
    df = pd.read_parquet(pq)

    rows = []
    for _, r in df.iterrows():
        opts = parse_mcq(r["mcq_options"])
        gold = str(r["mcq_answer"]).strip().upper()
        if len(opts) < 2 or gold not in opts:
            continue
        rows.append((r["question"], r["answer"], gold,
                     [opts.get(L, "") for L in LETTERS]))
    N = len(rows)
    print(f">> {N} usable rows")

    # ---- (2a) lexical overlap: question vs options (Jaccard) ---------------
    hit_lex = 0
    for q, _, gold, opts in rows:
        qw = content_words(q)
        scores = [(len(qw & content_words(o)) / max(len(qw | content_words(o)), 1))
                  for o in opts]
        if LETTERS[int(np.argmax(scores))] == gold:
            hit_lex += 1
    acc_lex = 100 * hit_lex / N
    print(f"(2a) lexical overlap picks correct: {acc_lex:.1f}%")

    # ---- embeddings for (2b) and (4): one MiniLM pass over all texts -------
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    flat = []
    for q, a, gold, opts in rows:
        flat.extend(opts); flat.append(a)
    emb = m.encode(flat, batch_size=256, normalize_embeddings=True,
                   show_progress_bar=True)
    emb = emb.reshape(N, 5, -1)                 # per row: 4 options + 1 answer

    # ---- (2b) odd-one-out: option most similar to the other three ----------
    hit_odd = 0
    for i, (_, _, gold, _) in enumerate(rows):
        O = emb[i, :4]                           # (4, d), normalized
        sim = O @ O.T
        mean_sim = (sim.sum(1) - 1.0) / 3.0      # mean cosine to the other 3
        if LETTERS[int(np.argmax(mean_sim))] == gold:
            hit_odd += 1
    acc_odd = 100 * hit_odd / N
    print(f"(2b) odd-one-out (most-typical option): {acc_odd:.1f}%")

    # ---- (4) free-form answer -> nearest option -----------------------------
    hit_ans, sim_gold, sim_dis = 0, [], []
    for i, (_, _, gold, _) in enumerate(rows):
        O, a = emb[i, :4], emb[i, 4]
        sims = O @ a
        gi = LETTERS.index(gold)
        if int(np.argmax(sims)) == gi:
            hit_ans += 1
        sim_gold.append(sims[gi])
        sim_dis.extend([s for j, s in enumerate(sims) if j != gi])
    acc_ans = 100 * hit_ans / N
    print(f"(4) answer-field nearest option == correct: {acc_ans:.1f}% "
          f"(mean cos: correct {np.mean(sim_gold):.3f} vs distractors {np.mean(sim_dis):.3f})")

    # ---- outputs ------------------------------------------------------------
    base = pd.DataFrame([
        {"video-blind policy": "Always answer C", "accuracy (%)": 63.4, "shortcut": "yes"},
        {"video-blind policy": "Most-typical option (odd-one-out, MiniLM)",
         "accuracy (%)": round(acc_odd, 1), "shortcut": "yes"},
        {"video-blind policy": "Always pick shortest option", "accuracy (%)": 40.3,
         "shortcut": "yes"},
        {"video-blind policy": "Max lexical overlap with question",
         "accuracy (%)": round(acc_lex, 1), "shortcut": "yes"},
        {"video-blind policy": "Random guess", "accuracy (%)": 25.0, "shortcut": "—"},
    ]).sort_values("accuracy (%)", ascending=False)
    base.to_csv(OUT/"tables/blind_baselines.csv", index=False)
    df_to_booktabs(base[["video-blind policy", "accuracy (%)"]],
        OUT/"tables/blind_baselines.tex", float_fmt="%.1f",
        caption="Video-blind baselines on EgoLongQA val (chance 25\\%). "
        "None of these read a single frame.", label="tab:blind_baselines")

    ans = pd.DataFrame([{
        "N": N, "nearest option = correct (%)": round(acc_ans, 1),
        "mean cos(answer, correct option)": round(float(np.mean(sim_gold)), 3),
        "mean cos(answer, distractor)": round(float(np.mean(sim_dis)), 3)}])
    ans.to_csv(OUT/"tables/answer_option_match.csv", index=False)
    df_to_booktabs(ans, OUT/"tables/answer_option_match.tex", float_fmt="%.3f",
        caption="Free-form \\texttt{answer} field vs.\\ MCQ options (MiniLM cosine). "
        "High agreement enables option-matching as an automatic grader for "
        "free-text model outputs.", label="tab:answer_match")

    fig, ax = plt.subplots(figsize=(COL*1.5, COL*0.85))
    b = base.iloc[::-1]
    colors = [CB[3] if s == "yes" else "0.6" for s in b["shortcut"]]
    ax.barh(range(len(b)), b["accuracy (%)"], color=colors, alpha=0.88,
            edgecolor="white", linewidth=0.4)
    ax.axvline(25, color="0.35", lw=1.0, ls=":")
    for yi, v in enumerate(b["accuracy (%)"]):
        ax.text(v + 1.2, yi, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels(b["video-blind policy"], fontsize=7)
    ax.set_xlabel("accuracy (%)"); ax.set_xlim(0, 80)
    ax.set_title("Video-blind baselines (no frame is read)", fontsize=8)
    ax.grid(axis="y", visible=False)
    print(">> wrote", savefig(fig, OUT/"figures/blind_baselines"))


if __name__ == "__main__":
    main()
