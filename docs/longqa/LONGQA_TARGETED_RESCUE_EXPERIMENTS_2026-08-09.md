# LongQA Targeted Rescue Experiments - 2026-08-09

## Fixed reference

All experiments first run on balanced fold 0. The frozen reference is the
endpoint/option-quota mean-probability fusion result, which scores `124/140` on
this fold and `618/700` overall. Non-target questions always retain this answer.

## Experiments

| Experiment | Controlled comparison | Promotion gate |
|---|---|---|
| Prompt ablation | Same endpoint-inclusive 64 frames; four answer instructions | Improve `124/140` without more than one regression |
| OCR ablation | Same full frames and detail crops; crops alone versus crops plus transcription | At least three fixes and at most one regression |
| Occurrence strips | Question-only detections grouped into distinct temporal occurrences, with before/center/after frames | At least three fixes and at most one regression |
| Density ablation | Same pivot source and prompt; raw 48, temporal deduplication, or visual deduplication | Best arm must exceed raw 48 and the reference |
| Support ablation | Identical endpoint frames; whole-option versus explicit-clause support | Conservative or strict policy must exceed `124/140` |
| Rescue router | OCR is one evidence family, occurrence strips are another, and clause support confirms overrides | Must improve without fold-specific threshold fitting |

The OCR arms are deliberately not counted as two independent votes because
they use the same visual crops. The router never introduces a synthetic third
candidate. It retains the primary answer unless an ordinary specialist answer
is confirmed by a different verification mechanism.

## Run order

The following three arrays and the occurrence job are independent and may run
simultaneously:

```bash
sbatch slurm_longqa_qwen35_27b_prompt_ablation_fold0_array.sh
sbatch slurm_longqa_qwen35_27b_ocr_ablation_fold0_array.sh
sbatch slurm_longqa_qwen35_27b_density_ablation_fold0_array.sh
sbatch slurm_longqa_qwen35_27b_occurrence_strips_fold0.sh
```

The support ablation is also independent, but it is the most expensive because
it scores multiple candidates per targeted question:

```bash
sbatch slurm_longqa_qwen35_27b_clause_support_fold0_array.sh
```

Run the CPU router only after both OCR tasks, occurrence strips, and the
`clauses` support task have completed:

```bash
sbatch slurm_longqa_targeted_rescue_router_fold0.sh
```

Do not run fold 1 or a full-set expansion until an arm passes its stated gate.
