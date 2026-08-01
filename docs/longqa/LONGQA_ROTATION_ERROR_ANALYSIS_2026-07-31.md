# EgoLongQA rotation-pivot error analysis

## Scope

The current best pipeline scores 565/700 (80.71%). It starts from the fixed
majority of Qwen3.5 pivot, Qwen3.5 uniform, and the Qwen3 verifier, then resolves
their 213 disagreements using Qwen3.5 pivot-view scores averaged across four
cyclic option placements.

## Where the 135 errors arise

| Failure source | Errors | Share of errors | Interpretation |
| --- | ---: | ---: | --- |
| Correct candidate available but not selected | 64 | 47.4% | Selection/verifier failure |
| Missing from the three deployed candidates but present in another run | 10 | 7.4% | Candidate-set coverage failure |
| Missing from all six strong runs | 61 | 45.2% | Evidence or model-understanding failure |

The deployed three-candidate oracle is 629/700 (89.86%), while the six-run
oracle is 639/700 (91.29%). Selection accuracy conditional on the correct
candidate being available is 565/629 (89.83%).

## Highest-impact slices

| Slice | Questions | Accuracy | Errors | Missing candidate | Selection errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-time ordering | 205 | 74.63% | 52 | 27 | 25 |
| OCR or named detail | 224 | 84.38% | 35 | 17 | 18 |
| Spatial location | 171 | 84.80% | 26 | 15 | 11 |
| Color or appearance | 44 | 75.00% | 11 | 5 | 6 |
| Shopping | 72 | 69.44% | 22 | 11 | 11 |
| Daily Activities | 127 | 79.53% | 26 | 19 | 7 |
| Sightseeing | 103 | 77.67% | 23 | 7 | 16 |

`BEFORE` questions score 74.07%. `LAST` scores 57.14%, although that slice has
only seven questions. The large cross-time slice, rather than OCR or generic
object identity, is the clearest systematic weakness.

## Sampling versus reasoning

The evidence does not support a single explanation.

1. **Wrong event instance or insufficient temporal structure.** Cross-time
   questions account for 52 errors. Typical failures require connecting the
   first and last product, distinguishing repeated encounters, or carrying an
   object identity across distant moments. More pixels do not directly solve
   these cases.
2. **Verifier selection.** The correct candidate exists in 64 failed examples.
   On final errors, Qwen3.5 uniform alone is correct in 26 cases while pivot is
   wrong; pivot alone is correct in only six. The pivot view can retrieve a
   visually relevant but temporally incorrect occurrence and then overrule a
   correct global-view answer.
3. **Upstream model/evidence failure.** All six strong systems miss 61 answers.
   These require a manual evidence-presence audit: if the answer is visible in
   the supplied frames, the failure is visual interpretation or temporal
   binding; if absent, it is frame selection.
4. **Fine detail is important but not the global bottleneck.** OCR/name,
   spatial, and fine-object questions score approximately 84%. Shopping remains
   difficult because it combines fine details with repeated products and event
   order. A universal resolution increase is unlikely to be cost-effective.

## Recommended next work

1. Freeze 565/700 as the primary reproducible pipeline.
2. Audit a stratified sample of 30 errors: 15 missing from all six systems and
   15 selection failures. Mark whether decisive evidence is visibly present in
   pivot and uniform packs. This directly measures sampling versus reasoning.
3. If larger-model compute is available, run a 32B model only on the 213 fixed
   disagreements using the existing pivot and uniform evidence. It must answer
   from all four options, not only the current candidates, so it can recover
   candidate-missing cases.
4. Use HieraMamba or InternVideo segments as additional evidence proposals for
   the evidence-absent portion, rather than replacing Qwen as the answerer.
5. Avoid another broad resolution or generic frame-count sweep. Any new visual
   intervention should target repeated-event, first/last, and before/after
   questions and retain a global timeline alongside local evidence.

Machine-readable per-question labels and aggregate statistics are stored under
`analysis/egolongqa/rotation_pivot_error_audit_2026-07-31/`.
