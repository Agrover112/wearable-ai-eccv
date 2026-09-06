# Temporal Suppression Experiments (2026-08-08)

All first-pass experiments use balanced fold 0 (140 questions). This fold was
constructed jointly over category, temporal operator, answer label, duration,
endpoint correctness, and endpoint/option-quota agreement. It is therefore a
safer development screen than the original fixed dev140 subset.

## Experiments

1. **Atomic hypothesis verifier.** Endpoint, option-quota, and uncertainty are
   the only candidate generators. Qwen3.5-27B rewrites each distinct candidate
   answer as visible claims, retrieves candidate-specific evidence, checks every
   part for support or contradiction, and may perform one additional retrieval.
   It keeps the current probability-fusion answer unless one candidate is
   supported and every competing candidate is contradicted.

2. **Wrong-time contrast.** For BEFORE and AFTER questions, frames are separated
   into the requested and opposite sides of the nearest retrieved pivot. Qwen
   scores all four options under both packs. The tested contrast subtracts an
   option's wrong-side support from its requested-side support, preventing a
   visually plausible event from winning merely because it occurs elsewhere.

3. **Hierarchical segment retrieval.** Qwen3.5-9B first scores coarse temporal
   windows, searches individual frames only inside promising windows, and saves
   48 local plus 16 global frames. A separate dependent job answers from those
   frames with Qwen3.5-27B.

4. **Targeted object occurrence ledger.** Grounding DINO detects only concepts
   extracted from the question over 48 candidate times, so distractor wording
   cannot create detector targets. The prompt
   receives a chronological ledger of first, strongest, and last detections,
   together with the selected visual frames. This tests repeated-object,
   identity, attribute, and OCR-like failures without densely cataloguing every
   object in every frame.

5. **Option-permutation calibration.** The same option-quota evidence is scored
   four times under cyclic option rotations. Scores are mapped back to semantic
   answers before averaging, so an answer cannot benefit consistently from being
   displayed as a preferred option letter.

6. **Uncertainty as a third evidence view.** Cached endpoint and option-quota
   rankings are retained. Only a third Qwen3.5-27B ranking over the already
   selected uncertainty frames is added. Mean probability, median probability,
   reciprocal-rank fusion, entropy-weighted fusion, and conservative
   uncertainty tie-breaking are evaluated without fitting labels.

## Run Order

The atomic verifier, wrong-time contrast, targeted object ledger,
option-permutation, uncertainty third-view, and hierarchical selection jobs are
independent. The 27B hierarchical answer job must run only after hierarchical
selection completes.

Recommended priority on limited GPUs:

1. uncertainty third-view;
2. wrong-time contrast;
3. atomic hypothesis verifier;
4. hierarchical selection, followed by hierarchical answer;
5. targeted object ledger;
6. option-permutation calibration.

Only methods that improve balanced fold 0 without a large category regression
should proceed to a second balanced fold. No routing threshold is to be fitted
on the 700-row validation labels.
