# Wearable AI LongQA Run Log

## Current Primary Pipeline

The current primary submission pipeline is
`qwen35_rotation_averaged_temporal_pivot`, recorded in
`configs/egolongqa_primary_pipeline.json`. It scores **565/700 (80.71%)** and
uses the fixed three-run majority except on disagreements, where candidate
answers are rescored on temporal-pivot frames under four cyclic option
rotations. This rule was selected on dev140 and independently confirmed on the
disjoint val560 split before promotion.

The temporal-pivot proof pack has a newly identified retrieval limitation:
SigLIP2 accepts only 64 text positions, while the original selector supplied a
single question-and-options string. On the 700 validation questions, 642
retrieval strings are truncated; option D is completely absent in 608, option C
in 540, and option B in 372. This affects frame retrieval only, not the final
Qwen prompt. Future proof packs must use independently encoded, length-safe
queries before they can replace the current validated primary output.

## Summary Table

| Run ID | Date | Task | Model | Backend | GPU | Samples | Frames sampled | Image budget | Runtime | Accuracy | Correct | Context fill | Output |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| `qwen2_5_vl_7b_hf_default32_full_2026-07-02` | 2026-07-02 | EgoLongQA val | Qwen/Qwen2.5-VL-7B-Instruct | HF | 1x H100 NVL | 700 | 4 | HF default | ~38 min | 0.5300 | 371/700 | n/a | `runs/egolongqa/qwen2_5_vl_7b_hf_default32_full_2026-07-02/` |
| `qwen3_vl_8b_vllm_default32_full_2026-07-02` | 2026-07-02 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 700 | 4 | 50,176 px (~224x224) | ~26 min | 0.5300 | 371/700 | n/a | `runs/egolongqa/qwen3_vl_8b_vllm_default32_full_2026-07-02/` |
| `qwen2_5_vl_7b_hf_32frames_full_2026-07-03` | 2026-07-03 | EgoLongQA val | Qwen/Qwen2.5-VL-7B-Instruct | HF | 1x H100 NVL | 700 | 32 | HF default | 5h25m41s | 0.6543 | 458/700 | mean 51,494 tok; p95 86,285; max 65.93% of 131K | `runs/egolongqa/qwen2_5_vl_7b_hf_32frames_full_2026-07-03/` |
| `qwen3_vl_8b_vllm_32frames_full_2026-07-03` | 2026-07-03 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 700 | 32 | 50,176 px (~224x224) | 3h40m27s | 0.6429 | 450/700 | mean 1,743 tok; p95 1,835; max 12.20% of 16K | `runs/egolongqa/qwen3_vl_8b_vllm_32frames_full_2026-07-03/` |
| `qwen3_vl_8b_vllm_32frames_px200704_2026-07-04` | 2026-07-04 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 700 | 32 | 200,704 px (~448x448) | 3h42m19s | **0.6629** | **464/700** | mean 6,336 tok; p95 6,443; max 20.16% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_32frames_px200704_2026-07-04/` |
| `qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06` | 2026-07-06 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP grounding | 1x H100 NVL | 700 | 32 selected from 128 | 200,704 px (~448x448) | 9h28m56s | **0.7000** | **490/700** | mean 6,048 tok; p95 6,366; max 20.16% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06/` |
| `qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07` | 2026-07-07 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 700 | 64 | 200,704 px (~448x448) | 7h41m32s | **0.7071** | **495/700** | mean 12,524 tok; p95 12,651; max 39.11% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07/` |
| `qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07` | 2026-07-07 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + timeline scaffold | 1x H100 NVL | 700 | 64 summary + 32 answer | 200,704 px (~448x448) | 11h30m05s | 0.6643 | 465/700 | 1,400 calls; mean 9,525 tok; p95 12,661; max 39.22% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07/` |
| `qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07` | 2026-07-07 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP grounding | 1x H100 NVL | 270/700 grounding only | up to 40 selected from 256 | 200,704 px (~448x448) | cancelled after ~11h30m | n/a | n/a | no generation | partial cache in `runs/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07/` |
| `qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08` | 2026-07-08 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP hybrid | 1x H100 NVL | 700 | 64 hybrid selected from 128 | 200,704 px (~448x448) | 12h02m10s | **0.7114** | **498/700** | mean 12,524 tok; p95 12,651; max 39.11% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08/` |
| `qwen3_vl_8b_vllm_uniform128_px50176_2026-07-08` | 2026-07-08 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 668/700 predictions | 128 | 50,176 px (~224x224) | timed out at 14h | partial 0.6976 | 466/668 | no final context summary | partial archive in `runs/egolongqa/qwen3_vl_8b_vllm_uniform128_px50176_2026-07-08/` |
| `qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-08` | 2026-07-08 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP grounding | 1x H100 NVL | 380/700 grounding only | up to 40 selected from 256 | 200,704 px (~448x448) | killed after ~16h | n/a | n/a | no generation | partial cache in `runs/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-08/` |
| `qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined_dev_2026-07-09` | 2026-07-09 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 140 | 64 | 200,704 px (~448x448) | 1h40m28s | 0.7429 | 104/140 | mean 12,646 tok; p95 12,760; max 39.42% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined_dev_2026-07-09/` |
| `qwen3_vl_8b_vllm_grounded_per_option_union_nms10_prompt_combined_dev_2026-07-09` | 2026-07-09 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP per-option retrieval | 1x H100 NVL | 140 grounding; 1 prediction | intended 64 selected from 128 | 200,704 px (~448x448) | failed after ~3h14m | n/a | n/a | no final summary | invalid working artifacts removed; fresh rerun required |
| `qwen3_vl_8b_vllm_cft_question_options_nms10_prompt_combined_dev_2026-07-09` | 2026-07-09 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP coarse-to-fine | 1x H100 NVL | 140 | 45-57 selected from 128 | 200,704 px (~448x448) | 4h24m10s | 0.7429 | 104/140 | mean 10,518 tok; p95 11,139; max 34.63% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_cft_question_options_nms10_prompt_combined_dev_2026-07-09/` |
| `qwen3_vl_8b_vllm_video_blind_prompt_baseline_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM, video blind | 1x H100 NVL | 140 | 0 | n/a | 3m08s | 0.5214 | 73/140 | mean 145 tok; p95 226; max 4.68% of 8K | `runs/egolongqa/qwen3_vl_8b_vllm_video_blind_prompt_baseline_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_uniform32_px451584_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 140 | 32 | 451,584 px (~672x672) | 49m24s | **0.7571** | **106/140** | mean 14,017 tok; p95 14,114; max 43.55% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform32_px451584_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_openqa_minilm_uniform64_px200704_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM open QA + MiniLM option matching | 1x H100 NVL | 140 | 64 | 200,704 px (~448x448) | 1h37m33s | 0.5286 | 74/140 | mean 12,463 tok; p95 12,519; max 38.26% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_openqa_minilm_uniform64_px200704_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_c128_top24_anc8_px200704_prompt_baseline_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP2 grounding | 1x H100 NVL | 140 | 26-32 selected from 128 | 200,704 px (~448x448) | 3h42m42s | 0.7214 | 101/140 | mean 6,055 tok; p95 6,362; max 19.67% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_c128_top24_anc8_px200704_prompt_baseline_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_uniform64_px451584_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 140 | 64 | 451,584 px (~672x672) | 1h43m54s | **0.7857** | **110/140** | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform64_px451584_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_uniform48_px313600_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 140 | 48 | 313,600 px (~560x560) | 1h16m58s | 0.6857 | 96/140 | mean 14,639 tok; p95 14,722; max 45.41% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform48_px313600_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_eventlet8x3_anc32_bnd8_final64_px200704_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP2 eventlets | 1x H100 NVL | 140 | 64 selected from 128 | 200,704 px (~448x448) | 1h34m46s | 0.7429 | 104/140 | mean 12,528 tok; p95 12,642; max 39.06% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_eventlet8x3_anc32_bnd8_final64_px200704_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_option_contrast4x3_anc16_final64_px200704_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + option-contrastive SigLIP2 eventlets | 1x H100 NVL | 140 | 64 selected from 128 | 200,704 px (~448x448) | 1h35m26s | 0.7000 | 98/140 | mean 12,528 tok; p95 12,642; max 39.06% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_option_contrast4x3_anc16_final64_px200704_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px200704_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + temporal-pivot proof pack | 1x H100 NVL | 140 | 64 selected from 128 | 200,704 px (~448x448) | 1h35m01s | **0.7786** | **109/140** | mean 12,528 tok; p95 12,642; max 39.06% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px200704_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_structured_anc24_final64_px200704_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + structured temporal-pivot proof pack | 1x H100 NVL | 140 | same 64-frame packs as unstructured pivot | 200,704 px (~448x448) | 55m09s | 0.7429 | 104/140 | mean 13,315 tok; p95 13,441; max 41.51% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_structured_anc24_final64_px200704_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_uniform64_px451584_full_2026-07-11` | 2026-07-11/12 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 700 | 64 | 451,584 px (~672x672) | 7h58m21s | **0.7343** | **514/700** | mean 27,884 tok; p95 28,011; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform64_px451584_full_2026-07-11/` |
| `qwen3_vl_8b_vllm_uniform96_px451584_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM | 1x H100 NVL | 140 | 96 | 451,584 px (~672x672) | 2h34m37s | 0.7286 | 102/140 | mean 41,760 tok; p95 41,890; max 64.16% of 65K | `runs/egolongqa/qwen3_vl_8b_vllm_uniform96_px451584_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + temporal-pivot proof pack | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 1h37m49s | **0.7929** | **111/140** | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px200704_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + coverage-filled temporal pivot | 1x H100 NVL | 140 | 64 selected from 128 | 200,704 px (~448x448) | 1h33m00s | 0.7571 | 106/140 | mean 12,528 tok; p95 12,642; max 39.06% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px200704_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px451584_dev_2026-07-11` | 2026-07-11 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + coverage-filled temporal pivot | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 59m10s | 0.7714 | 108/140 | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px451584_dev_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px451584_full_2026-07-11` | 2026-07-11/12 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + coverage-filled temporal pivot | 1x H100 NVL | 700 | 64 selected from 128 | 451,584 px (~672x672) | 10h28m40s | **0.7557** | **529/700** | mean 27,884 tok; p95 28,011; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px451584_full_2026-07-11/` |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12` | 2026-07-12 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + temporal-pivot proof pack | 1x H100 NVL | 700 | 64 selected from 128 | 451,584 px (~672x672) | 8h11m30s | **0.7543** | **528/700** | mean 27,884 tok; p95 28,011; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12/` |
| `qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_dev_2026-07-12` | 2026-07-12 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + exact operator router | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 1h00m02s | **0.7929** | **111/140** | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_dev_2026-07-12/` |
| `qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_full_2026-07-12` | 2026-07-12/13 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + exact operator router | 1x H100 NVL | 700 | 64 selected from 128 | 451,584 px (~672x672) | 7h48m06s | **0.7557** | **529/700** | mean 27,884 tok; p95 28,011; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_full_2026-07-12/` |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12` | 2026-07-12 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + pivot/uniform disagreement verifier | 1x H100 NVL | 140 (19 verifier calls) | 64 on disagreements | 451,584 px (~672x672) | 12m56s | **0.7929** | **111/140** | verifier calls: mean 27,967 tok; p95 28,033; max 57.44% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12/` |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12` | 2026-07-12 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + pivot/uniform disagreement verifier | 1x H100 NVL | 700 (122 verifier calls) | 64 on disagreements | 451,584 px (~672x672) | 1h22m07s | **0.7700** | **539/700** | verifier calls: mean 27,942 tok; p95 28,033; max 57.44% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12/` |
| `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13` | 2026-07-13 | EgoLongQA dev140 | yanziang/InternVideo3-8B-Instruct | HF/PyTorch | 1x H100 | 140 | 64 | 451,584 px per-frame cap | 1h42m20s | 0.6429 | 90/140 | mean 14,266 tok; p95 14,363; max 5.54% of 262K | `runs/egolongqa/internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13/` |
| `qwen3_vl_8b_vllm_siglip2_qca_s16_b64_px451584_dev_2026-07-13` | 2026-07-13/14 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + QCA proof pack | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 1h46m42s | 0.7429 | 104/140 | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_qca_s16_b64_px451584_dev_2026-07-13/` |
| `qwen3_vl_8b_vllm_siglip2_qca_global_router_pivot24_s16_b64_px451584_dev_2026-07-13` | 2026-07-13/14 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + QCA/pivot router | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 1h42m26s | 0.7786 | 109/140 | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_qca_global_router_pivot24_s16_b64_px451584_dev_2026-07-13/` |
| `qwen3_vl_8b_vllm_siglip2_multi_event_router_c3x2_px451584_dev_2026-07-13` | 2026-07-13/14 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + multi-event/pivot router | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 1h41m11s | 0.7857 | 110/140 | mean 27,888 tok; p95 28,002; max 57.29% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_multi_event_router_c3x2_px451584_dev_2026-07-13/` |
| `qwen3_vl_8b_vllm_support_contradiction_verifier_global_after_before_last_px451584_dev_2026-07-13` | 2026-07-13 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + gated support/contradiction verifier | 1x H100 NVL | 140 (17 verifier calls) | 64 on disagreements | 451,584 px (~672x672) | 12m43s | **0.7929** | **111/140** | verifier calls: mean 27,975 tok; p95 28,011; max 57.00% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_support_contradiction_verifier_global_after_before_last_px451584_dev_2026-07-13/` |
| `qwen3_vl_8b_vllm_pairwise_order_swap_verifier_px451584_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + candidate-constrained order-swap verifier | 1x H100 NVL | 140 (38 verifier calls) | 64 on disagreements | 451,584 px (~672x672) | 11m56s | 0.7786 | 109/140 | verifier calls: mean 27,897 tok; p95 56.81%; max 56.82% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pairwise_order_swap_verifier_px451584_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_siglip2_qframe_c128_h4m8l32_mixedres_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP2 Q-Frame mixed resolution | 1x H100 NVL | 140 | 44 selected from 128 | 4x451,584 + 8x200,704 + 32x50,176 px | 36m04s | 0.6857 | 96/140 | mean 5,026 tok; p95 15.61%; max 16.09% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_qframe_c128_h4m8l32_mixedres_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_pivot_letterlogp_blindw05_px451584_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + visual/blind option-letter likelihood | 1x H100 NVL | 140 (280 scoring calls) | 64 pivot + blind branch | 451,584 px (~672x672) | 57m34s | 0.6429 | 90/140 | calls: mean 14,015 tok; p95 56.91%; max 57.28% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_letterlogp_blindw05_px451584_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_pivot_qgate_lite_cap16_narr12_px200704_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + lightweight narrative gate | 1x H100 NVL | 140 (280 generation calls) | 16 caption anchors + 64 answer frames | 200,704 px (~448x448) | 1h59m24s | 0.7000 | 98/140 | calls: mean 8,122 tok; p95 39.64%; max 40.10% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_qgate_lite_cap16_narr12_px200704_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_siglip2_adaq_c256_f64_px200704_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP2 AdaQ | 1x H100 NVL | 140 | 64 selected from 256 | 200,704 px (~448x448) | 4h10m40s | 0.6857 | 96/140 | mean 12,528 tok; p95 38.58%; max 39.06% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_adaq_c256_f64_px200704_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_siglip2_focus_c256_a16_z025_f64_px200704_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + SigLIP2 FOCUS | 1x H100 NVL | 140 | 64 selected from 256 | 200,704 px (~448x448) | 4h30m35s | 0.6643 | 93/140 | mean 12,528 tok; p95 38.58%; max 39.06% of 32K | `runs/egolongqa/qwen3_vl_8b_vllm_siglip2_focus_c256_a16_z025_f64_px200704_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_pivot_letterlogp_blindwm01_px451584_val560_2026-07-15` | 2026-07-15/16 | EgoLongQA held-out val560 | Qwen/Qwen3-VL-8B-Instruct | vLLM + locked visual/blind option likelihood | 1x H100 NVL | 560 (1,120 scoring calls) | 64 pivot + blind branch | 451,584 px (~672x672) | 6h06m47s | 0.7482 | 419/560 | calls: mean 14,015 tok; p95 56.94%; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_letterlogp_blindwm01_px451584_val560_2026-07-15/` |
| `qwen3_vl_8b_vllm_candidate_logp_verifier_blindwm01_px451584_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + candidate-restricted pivot/uniform likelihood | 1x H100 NVL | 140 (19 disagreements; 57 scoring calls) | separate 64-frame pivot and uniform packs | 451,584 px (~672x672) | 23m56s | 0.7643 | 107/140 | calls: mean 18,650 tok; p95 56.85%; max 56.89% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_candidate_logp_verifier_blindwm01_px451584_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_option_permutation_verifier_px451584_dev_2026-07-15` | 2026-07-15 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + candidate-restricted option-permutation verifier | 1x H100 NVL | 140 (19 disagreements; 76 generation calls) | 64-frame verifier pack | 451,584 px (~672x672) | 13m20s | 0.7786 | 109/140 | calls: mean 27,911 tok; p95 56.86%; max 56.90% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_option_permutation_verifier_px451584_dev_2026-07-15/` |
| `qwen3_vl_8b_vllm_event_ledger_f16_r8_final64_px451584_dev20_2026-07-16` | 2026-07-16 | EgoLongQA dev20 | Qwen/Qwen3-VL-8B-Instruct | vLLM + schema event ledger | 1x H100 NVL | 0/20 | intended 16 ledger + 64 answer | 200,704 px ledger; 451,584 px answer | failed during first ledger batch | n/a | n/a | first JSON response truncated at 256 tokens | failed attempt archived; rerun required |
| `qwen3_vl_8b_vllm_semantic_candidate_logp_pivot_uniform_dev_2026-07-16` | 2026-07-16 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM full-answer likelihood | 1x H100 NVL | 0/140 | intended 19 disagreement scorings over separate 64-frame contexts | 451,584 px (~672x672) | failed on first scoring call | n/a | n/a | prompt-logprob logits caused 4.43 GiB OOM | failed attempt archived; rerun required |
| `qwen3_vl_8b_vllm_event_ledger_groundingdino_final64_px451584_dev20_2026-07-16` | 2026-07-16 | EgoLongQA dev20 | Grounding DINO + Qwen3-VL | detector augmentation | 1x H100 NVL | 0/20 | depends on completed event ledger | n/a | exited before model startup | n/a | n/a | no completed `LEDGER_RUN` supplied | no artifacts; rerun only after ledger succeeds |
| `qwen3_vl_8b_vllm_event_ledger_f16_r8_final64_px451584_dev20_2026-07-17` | 2026-07-17 | EgoLongQA dev20 | Qwen/Qwen3-VL-8B-Instruct | vLLM + schema event ledger | 1x H100 NVL | 20 | 16 ledger frames + 64 answer frames | 200,704 px ledger; 451,584 px answer | 33m46s | 0.7000 | 14/20 | 343 calls; mean 1,979 tok; p95 28,720; max 59.75% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_event_ledger_f16_r8_final64_px451584_dev20_2026-07-17/` |
| `qwen3_vl_8b_vllm_semantic_candidate_logp_pivot_uniform_dev_2026-07-17` | 2026-07-17 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM full-answer likelihood | 1x H100 NVL | 140 (19 disagreements; 114 scoring calls) | separate 64-frame pivot and uniform contexts | 451,584 px (~672x672) | 32m47s | 0.7857 | 110/140 | mean 27,847 tok; p95 27,874; max 56.71% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_semantic_candidate_logp_pivot_uniform_dev_2026-07-17/` |
| `qwen3_vl_8b_vllm_event_ledger_groundingdino_final64_px451584_dev20_2026-07-17` | 2026-07-17 | EgoLongQA dev20 | Grounding DINO + Qwen3-VL | detector augmentation | 1x H100 NVL | 0/20 | intended 16 detection frames + 64 answer frames | n/a | two failed startup attempts | n/a | n/a | mixed FP32/BF16 operations inside detector | current code loads detector fully in FP32; rerun required |
| `qwen3_vl_8b_vllm_uniform64_answer_cot_px451584_dev_2026-07-21` | 2026-07-21/22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + answer-stage temporal rationale | 1x H100 NVL | 140 generated; invalid evaluation | 64 uniform | 451,584 px (~672x672) | 1h51m41s | invalid (raw 0.5286) | raw 74/140 | 140 calls; mean 27,912 tok; p95 28,026; max 57.34% of 49K | 48/140 responses lacked the required final-answer marker before the 192-token cap; archived as failed attempt and rerun required |
| `qwen3_vl_8b_vllm_tcot_single_c128_sel64_px451584_dev_2026-07-21` | 2026-07-21/22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + single-step Qwen TCoT | 1x H100 NVL | 140 generated; invalid evaluation | mean 15.3 selected from 128 candidates | 50,176 px selector; 451,584 px answer | 3h43m04s | invalid (raw 0.3000) | raw 42/140 | 286 calls; mean 6,715 tok; p95 7,242; max 51.67% of 49K | 52/140 answers lacked a final marker; valid selector cache retained, answer-only rerun required |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel64_px451584_dev_2026-07-21` | 2026-07-21/22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + four-segment Qwen TCoT | 1x H100 NVL | 140 generated; invalid evaluation | mean 28.1 selected from 256 candidates | 50,176 px selector; 451,584 px answer | 6h54m52s | invalid (raw 0.4857) | raw 68/140 | 704 calls; mean 5,223 tok; p95 14,074; max 56.92% of 49K | 40/140 answers lacked a final marker; selector cache complete, answer-only rerun required |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel48_u16_px451584_dev_2026-07-21` | 2026-07-21/22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + four-segment Qwen TCoT and uniform coverage | 1x H100 NVL | 140 generated; invalid evaluation | mean 43.9 final; selected neighborhoods + 16 uniform | 50,176 px selector; 451,584 px answer | 6h30m22s | invalid (raw 0.4929) | raw 69/140 | 595 calls; mean 7,158 tok; p95 21,045; max 57.09% of 49K | 43/140 answers lacked a final marker; concurrent cache race changed selections on 22 rows, matched-cache rerun required |
| `qwen3_vl_8b_vllm_uniform64_answer_cot_px451584_dev_2026-07-22` | 2026-07-22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + answer-stage temporal rationale with strict retry | 1x H100 NVL | 140 | 64 uniform | 451,584 px (~672x672) | 2h01m56s | **0.7857** | **110/140** | 280 calls; mean 27,820 tok; p95 27,973; max 57.29% of 49K | all 140 rationale calls required answer-only retry; result ties uniform64 and adds no usable answer-stage gain |
| `qwen3_vl_8b_vllm_tcot_single_c128_sel64_px451584_dev_2026-07-22` | 2026-07-22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + cached single-step Qwen TCoT | 1x H100 NVL | 140 | mean 15.3 selected from 128 candidates | 50,176 px selector; 451,584 px answer | 26m25s | 0.5857 | 82/140 | 140 answer calls; mean 6,763 tok; p95 8,775; max 51.62% of 49K | 140/140 selector cache hits; complete valid archive |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel64_px451584_dev_2026-07-22` | 2026-07-22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + cached four-segment Qwen TCoT | 1x H100 NVL | 140 | mean 28.1 selected from 256 candidates | 50,176 px selector; 451,584 px answer | 46m34s | 0.6786 | 95/140 | 140 answer calls; mean 12,335 tok; p95 19,045; max 56.87% of 49K | 140/140 selector cache hits; complete valid archive |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel48_u16_px451584_dev_2026-07-22` | 2026-07-22 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + cached four-segment Qwen TCoT and uniform coverage | 1x H100 NVL | 40/140 valid prefix | mean 43.6 final over prefix; selected neighborhoods + 16 uniform | 50,176 px selector; 451,584 px answer | interrupted after row 40 | partial 0.7000 | 28/40 | no final context summary | transient OpenCV metadata timeout at row 41; resumable prefix archived and working state retained |
| `qwen3_vl_8b_vllm_object_labels_pivot64_px451584_dev_2026-07-26` | 2026-07-26 | EgoLongQA dev140 | Grounding DINO + Qwen/Qwen3-VL-8B-Instruct | detector labels over temporal-pivot frames + vLLM | 1x H100 NVL | 140 | 64 pivot frames; detections on 32 frames | 451,584 px (~672x672) | 2h59m07s | 0.6857 | 96/140 | mean 27,913 tok; p95 28,027; max 57.34% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_object_labels_pivot64_px451584_dev_2026-07-26/` |
| `qwen3_vl_8b_thinking_vllm_uniform64_px451584_dev_2026-07-26` | 2026-07-26 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Thinking | vLLM reasoning + strict-retry attempt | 1x H100 NVL | 140 generated; invalid evaluation | 64 uniform | 451,584 px (~672x672) | 2h20m33s resumed run | invalid (raw 0.5929) | raw 83/140 | 155 calls; mean 28,357 tok; p95 30,136; max 61.54% of 49K | 26/140 outputs still lacked a final-answer marker after retry; archived for protocol debugging, not fair comparison |
| `qwen3_vl_8b_thinking_budget2048_vllm_uniform64_px451584_dev_2026-07-26_failed_job49270165` | 2026-07-26 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Thinking | vLLM bounded-reasoning attempt | 1x H100 NVL | 0/140 | 64 uniform | 451,584 px (~672x672) | failed on first request | n/a | n/a | no successful calls | vLLM required an explicit server-side reasoning configuration for `thinking_token_budget`; failed artifacts archived separately |
| `qwen3_vl_8b_thinking_budget2048_vllm_uniform64_px451584_dev_2026-07-26` | 2026-07-26/27 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Thinking | vLLM bounded reasoning with final-answer reserve | 1x H100 NVL | 140 | 64 uniform | 451,584 px (~672x672) | 2h28m13s | 0.6714 | 94/140 | 143 calls; mean 28,063 tok; p95 28,157; max 62.05% of 49K | 140/140 valid final markers; three answer-only retries; complete archive |
| `qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26` | 2026-07-26 | EgoLongQA dev140 | Grounding DINO + Qwen/Qwen3-VL-8B-Instruct | pivot frames + paired object crops | 1x H100 NVL | 140 | 56 pivot context + up to 8 object crops; 64 final | 451,584 px (~672x672) | 1h51m05s | 0.7786 | 109/140 | mean 26,157 tok; p95 27,924; max 57.06% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26/` |
| `qwen3_vl_8b_vllm_object_panels_f56_p8_px451584_dev_2026-07-26` | 2026-07-26 | EgoLongQA dev140 | Grounding DINO + Qwen/Qwen3-VL-8B-Instruct | pivot frames + object evidence panels | 1x H100 NVL | 140 | 56 pivot context + up to 8 panels; 64 final | 451,584 px (~672x672) | 1h49m59s | 0.7643 | 107/140 | mean 27,996 tok; p95 28,106; max 57.50% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_object_panels_f56_p8_px451584_dev_2026-07-26/` |
| `qwen3_vl_8b_vllm_object_ledger_pivot64_px451584_dev_2026-07-26` | 2026-07-26 | EgoLongQA dev140 | Grounding DINO + Qwen/Qwen3-VL-8B-Instruct | detector-derived object ledger + pivot frames | 1x H100 NVL | 140 | 64 pivot frames with text ledger | 451,584 px (~672x672) | 2h04m32s | 0.7286 | 102/140 | mean 28,408 tok; p95 28,601; max 58.55% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_object_ledger_pivot64_px451584_dev_2026-07-26/` |
| `qwen3_vl_8b_vllm_ug_segmented_c64_s16x2_u32_final64_px451584_dev20_2026-07-27` | 2026-07-27 | EgoLongQA temporal dev20 | Qwen/Qwen3-VL-8B-Instruct | vLLM + segment-balanced intrinsic uncertainty | 1x H100 NVL | 20 | 32 low-entropy segment frames + uniform fill; 64 final | 50,176 px scoring; 451,584 px answer | 37m57s | 0.7500 | 15/20 | 1,300 calls; mean 642 tok; p95 436; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_segmented_c64_s16x2_u32_final64_px451584_dev20_2026-07-27/` |
| `qwen3_vl_8b_vllm_ug_temporal_pivot_c128_anc24_final64_px451584_dev20_2026-07-27` | 2026-07-27 | EgoLongQA temporal dev20 | Qwen/Qwen3-VL-8B-Instruct | vLLM + uncertainty temporal pivot | 1x H100 NVL | 20 | 128 target-entropy candidates + 128 pivot-presence scores; 64 final | 50,176 px scoring; 451,584 px answer | 1h27m37s | **0.8000** | **16/20** | 5,140 calls; mean 273 tok; p95 431; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_temporal_pivot_c128_anc24_final64_px451584_dev20_2026-07-27/` |
| `qwen3_vl_8b_vllm_ug_c128_s16x4_dynamic_tcot_s4_sel48_u16_px451584_dev20_2026-07-27` | 2026-07-27 | EgoLongQA temporal dev20 | Qwen/Qwen3-VL-8B-Instruct | cached uncertainty shortlist + dynamic TCoT | 1x H100 NVL | 20 | 64-frame uncertainty shortlist; 21-54 final frames after TCoT + 16 uniform | 50,176 px selector; 451,584 px answer | 1h06m50s | 0.7500 | 15/20 | 100 calls; mean 4,339 tok; p95 19,667; max 48.49% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_c128_s16x4_dynamic_tcot_s4_sel48_u16_px451584_dev20_2026-07-27/` |
| `qwen3_vl_8b_vllm_ug_window_c128_w9_st3_s16_u16_final64_px451584_dev20_2026-07-29` | 2026-07-29 | EgoLongQA temporal dev20 | Qwen/Qwen3-VL-8B-Instruct | vLLM + overlapping-window intrinsic uncertainty | 1x H100 NVL | 20 | low-entropy 9-frame windows at stride 3 + 16 uniform frames; 64 final | 50,176 px scoring; 451,584 px answer | 59m56s | 0.7500 | 15/20 | 840 calls; mean 1,264 tok; max 57.32% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_window_c128_w9_st3_s16_u16_final64_px451584_dev20_2026-07-29/` |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev140prefix20_2026-07-29` | 2026-07-29 | EgoLongQA dev140 prefix20 | Qwen/Qwen3-VL-8B-Instruct | vLLM + uncertainty-ranked detector crops | 1x H100 NVL | 20 | 56 pivot context frames + 8 uncertainty-ranked object crops | 50,176 px crop scoring; 451,584 px answer | 22m53s | **0.8000** | **16/20** | 492 calls; mean 1,286 tok; max 56.87% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev140prefix20_2026-07-29/` |
| `qwen35_9b_vllm_uniform64_px451584_smoke5_2026-07-29` | 2026-07-29 | EgoLongQA dev140 prefix smoke | Qwen/Qwen3.5-9B | vLLM + Triton GDN prefill | 1x H100 NVL | 5 | 64 uniform | 451,584 px (~672x672) | 6m33s, including 3m25s warmup | 1.0000 | 5/5 | mean 27,983 tok; p95/max 28,049; max 57.07% of 49K | `runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_smoke5_2026-07-29/` |
| `qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29` | 2026-07-29 | EgoLongQA dev140 | Qwen/Qwen3.5-9B | vLLM + Triton GDN prefill | 1x H100 NVL | 140 | 64 uniform | 451,584 px (~672x672) | 1h42m19s | **0.8000** | **112/140** | mean 27,960 tok; p95 28,074; max 57.44% of 49K | `runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29/` |
| `qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29` | 2026-07-29 | EgoLongQA dev140 | Qwen/Qwen3.5-9B | vLLM + exact Qwen3-VL temporal-pivot frames | 1x H100 NVL | 140 | 64 selected from 128 | 451,584 px (~672x672) | 1h42m43s | **0.8071** | **113/140** | mean 27,960 tok; p95 28,074; max 57.44% of 49K | `runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29/` |
| `qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29` | 2026-07-29 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | entropy router over pivot, uniform, and crop-pair answers | 1x H100 NVL | 140; 33 routed disagreements | 3x64-frame uncertainty scoring on disagreements | 451,584 px (~672x672) | 1h15m09s | **0.8214** | **115/140** | 99 scoring calls; mean 27,296 tok; max 56.97% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29/` |
| `offline_majority_ug_router_qwen35_pivot_qwen35_uniform_2026-07-29` | 2026-07-29 | EgoLongQA dev140 | Qwen3-VL + Qwen3.5 fixed ensemble | offline majority vote | none | 140 | reuses three completed runs | n/a | <1 min | **0.8429** | **118/140** | no model calls | `runs/egolongqa/offline_majority_ug_router_qwen35_pivot_qwen35_uniform_2026-07-29/` |
| `qwen35_9b_vllm_router_pivot_candidate_pair_verifier_px451584_dev_2026-07-29` | 2026-07-29 | EgoLongQA dev140 | Qwen/Qwen3.5-9B | candidate-constrained verifier over Qwen3.5 pivot/router disagreements | 1x H100 NVL | 140; 31 verifier calls | 64 verifier frames | 451,584 px (~672x672) | 17m15s | **0.8214** | **115/140** | calls: mean 27,905 tok; p95 28,000; max 57.04% of 49K | `runs/egolongqa/qwen35_9b_vllm_router_pivot_candidate_pair_verifier_px451584_dev_2026-07-29/` |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev_2026-07-29` | 2026-07-29/30 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | uncertainty-ranked detector crops | 1x H100 NVL | 140 | 56 pivot frames + 8 crops selected from 24 | 50,176 px scoring; 451,584 px answer | 2h29m24s | 0.7571 | 106/140 | 2,825 calls; mean 1,518 tok; max 56.99% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev_2026-07-29/` |
| `qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29` | 2026-07-29/30 | EgoLongQA val | Qwen/Qwen3.5-9B | vLLM + fixed cached temporal-pivot evidence | 1x H100 NVL | 700 | exact cached 64-frame packs selected from 128 | 451,584 px (~672x672) | 7h53m11s | **0.7671** | **537/700** | mean 27,956 tok; p95 28,083; max 57.47% of 49K | `runs/egolongqa/qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29/` |
| `qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-30` | 2026-07-30 | EgoLongQA dev140 | Qwen/Qwen3.5-9B | original pivot/uniform disagreement verifier | 1x H100 NVL | 140; 19 verifier calls | 64 verifier frames | 451,584 px (~672x672) | 16m36s | **0.8071** | **113/140** | calls: mean 28,039 tok; p95 28,077; max 57.16% of 49K | `runs/egolongqa/qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-30/` |
| `qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-30` | 2026-07-30 | EgoLongQA val | Qwen/Qwen3.5-9B | original pivot/uniform disagreement verifier | 1x H100 NVL | 700; 122 verifier calls | 64 verifier frames | 451,584 px (~672x672) | 1h23m59s | **0.7700** | **539/700** | calls: mean 28,014 tok; p95 28,106; max 57.58% of 49K | `runs/egolongqa/qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-30/` |

## Job Status Snapshot (2026-07-11 12:49 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_openqa_minilm_uniform64_px200704_dev_2026-07-11` | 49020024 | Completed | 140/140; accuracy 0.5286; final artifacts archived |
| `qwen3_vl_8b_vllm_uniform32_px451584_dev_2026-07-11` | 49020027 | Completed | 140/140; accuracy 0.7571; final artifacts archived |
| `qwen3_vl_8b_vllm_video_blind_prompt_baseline_dev_2026-07-11` | 49020023 | Completed | 140/140; accuracy 0.5214; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_c128_top24_anc8_px200704_prompt_baseline_dev_2026-07-11` | 49020061 | Completed | proper rerun after compatibility fix; 140/140; accuracy 0.7214; final artifacts archived |

The original SigLIP2 job `49020026` failed before producing a grounding row because Transformers returned `BaseModelOutputWithPooling` instead of a bare tensor. The compatibility fix was applied and job `49020061` completed the proper rerun. No further rerun is needed for this configuration.

All six proof-pack/resolution jobs submitted later on 2026-07-11 completed with 140 predictions, final evaluation summaries, diagnostics, and archived Slurm/vLLM logs. No proof-pack reruns are required for artifact completeness.

The six next-stage jobs also completed: both full runs contain 700 predictions and all four dev runs contain 140. The full revised-pivot run populated the remaining SigLIP2 c128 feature cache; 140/700 rows were cache hits during that job and the other 560 were encoded and stored.

## Job Status Snapshot (2026-07-13 01:44 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12` | 49050528 | Completed | 700/700; accuracy 0.7543; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_dev_2026-07-12` | 49052006 | Completed | 140/140; accuracy 0.7929; final artifacts archived |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12` | 49052008 | Completed | 140/140; 19 verifier calls; accuracy 0.7929; final artifacts archived |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12` | 49052012 | Completed | 700/700; 122 verifier calls; accuracy 0.7700; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_full_2026-07-12` | 49052007 | Completed | 700/700; accuracy 0.7557; final artifacts archived |
| `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13` | 49058687 | Completed | 140/140; accuracy 0.6429; runtime 1h42m20s; final predictions, evaluation, and diagnostics archived |

The full verifier is the new best completed full-validation result. Relative to
the original pivot (`528/700`), it fixes 24 errors and regresses 13 correct
answers for a net gain of 11. It selects the pivot answer 76 times, the uniform
answer 38 times, and a third answer 8 times across the 122 disagreements. The
largest operator-level gain is on `GLOBAL` questions (`+9` net); `AFTER`,
`BEFORE`, and `LAST` contribute another `+4`, while `FIRST` and `STATE_CHANGE`
lose one each. The candidate oracle is `575/700`, leaving 36 correct answers
between the verifier and perfect candidate selection.

The dev verifier ties its primary pivot at `111/140`, so the full-set gain also
confirms that dev140 is too noisy to rank small routing changes by accuracy
alone. The original and coverage-filled full pivots are effectively tied
(`528` versus `529`), despite a larger dev gap; the extra generic coverage did
not produce a meaningful full-set improvement.

The independently executed operator router also reaches `529/700`. Its 493
temporal-pivot routes solve 372 rows (`0.7546`) and its 207 exact-uniform
`GLOBAL` routes solve 157 (`0.7585`). It differs from the original-pivot answers
on 38 rows and has a `+17/-16` crossover, so routing yields only one net answer.
It differs from uniform64 on 84 rows with a stronger `+45/-30` crossover. The
result validates operator specialization, but deterministic first-pass routing
does not replace disagreement verification: the verifier is still ten answers
better (`539` versus `529`).

## Job Status Snapshot (2026-07-14 00:23 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_siglip2_qca_s16_b64_px451584_dev_2026-07-13` | 49078726 | Completed | 140/140; accuracy 0.7429; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_qca_global_router_pivot24_s16_b64_px451584_dev_2026-07-13` | 49078727 | Completed | 140/140; accuracy 0.7786; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_multi_event_router_c3x2_px451584_dev_2026-07-13` | 49078729 | Completed | 140/140; accuracy 0.7857; final artifacts archived |
| `qwen3_vl_8b_vllm_support_contradiction_verifier_global_after_before_last_px451584_dev_2026-07-13` | 49078730 | Completed | 140/140; 17 verifier calls and 2 gated disagreements; accuracy 0.7929; final artifacts archived |

## Job Status Snapshot (2026-07-15 22:27 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_pairwise_order_swap_verifier_px451584_dev_2026-07-15` | 49180946 | Completed | 140/140; 38 verifier calls; accuracy 0.7786; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_qframe_c128_h4m8l32_mixedres_dev_2026-07-15` | 49180962 | Completed | 140/140; accuracy 0.6857; final artifacts archived |
| `qwen3_vl_8b_vllm_pivot_letterlogp_blindw05_px451584_dev_2026-07-15` | 49181553 | Completed rerun | 140/140; accuracy 0.6429; final artifacts archived; original job 49180964 failed before inference because `psutil` was unavailable on the launcher path |
| `qwen3_vl_8b_vllm_pivot_qgate_lite_cap16_narr12_px200704_dev_2026-07-15` | 49181554 | Completed rerun | 140/140; accuracy 0.7000; final artifacts archived; original job 49180971 failed before inference for the same launcher-path issue |
| `qwen3_vl_8b_vllm_siglip2_adaq_c256_f64_px200704_dev_2026-07-15` | 49180952 | Completed | 140/140; accuracy 0.6857; diagnostics and final artifacts archived manually after the submitted wrapper hit its stale post-run quoting bug |
| `qwen3_vl_8b_vllm_siglip2_focus_c256_a16_z025_f64_px200704_dev_2026-07-15` | 49180976 | Completed | 140/140; accuracy 0.6643; diagnostics and final artifacts archived manually after the same stale post-run quoting bug |

The local environment does not expose `squeue`, so the earlier running state was
inferred from monotonically growing prediction files and fresh vLLM server logs.
Both adaptive jobs completed normally at the inference/evaluation level. Their
submitted shell processes retained an older wrapper body and encountered
`unexpected EOF` only during post-run archival; complete working outputs were
used to regenerate diagnostics and reconstruct the canonical archives, so no GPU
rerun is required. The current wrapper passes `bash -n`, preserves the user-site
package path, enables unbuffered output, and uses process-specific cache temporary
files.

## Job Status Snapshot (2026-07-16 18:35 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_pivot_letterlogp_blindwm01_px451584_val560_2026-07-15` | 49183656 | Completed | 560/560 held-out rows; accuracy 0.7482; diagnostics and final artifacts archived |
| `qwen3_vl_8b_vllm_candidate_logp_verifier_blindwm01_px451584_dev_2026-07-15` | 49183657 | Completed | 140/140; 19 disagreement decisions; accuracy 0.7643; final artifacts archived |
| `qwen3_vl_8b_vllm_option_permutation_verifier_px451584_dev_2026-07-15` | 49183658 | Completed | 140/140; 19 disagreement decisions and 76 rotated calls; accuracy 0.7786; final artifacts archived |

The `.err` files contain only expected subset-evaluation warnings caused by
matching 140 or 560 keyed predictions against the 700-row annotation file.
There are no tracebacks, OOMs, missing predictions, or failed runs. These jobs
also use sample-key-aligned proof-pack lookup; the archived pivot inputs contain
one unique video per row, so this robustness correction does not retroactively
change the earlier pivot/verifier results.

## Job Status Snapshot (2026-07-17 11:16 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_event_ledger_f16_r8_final64_px451584_dev20_2026-07-16` | 49196308 | Failed; rerun required | First structured frame record reached the 256-token generation cap and returned unterminated JSON; 0/20 predictions and no scratch ledger records |
| `qwen3_vl_8b_vllm_semantic_candidate_logp_pivot_uniform_dev_2026-07-16` | 49196309 | Failed; rerun required | First pivot-context prompt-logprob request OOMed while allocating 4.43 GiB for logits; 0/140 predictions and 0/19 score records |
| `qwen3_vl_8b_vllm_event_ledger_groundingdino_final64_px451584_dev20_2026-07-16` | 49196311 | Not started | Launcher exited before model startup because the prerequisite ledger archive did not exist and `LEDGER_RUN` was unset |

The event schema now caps salient lists at six short phrases and retries only
failed structured requests at 512 tokens. The semantic-likelihood launcher now
uses 1,024-token chunked prefills and reserves 15% of GPU memory, addressing the
temporary full-vocabulary logit allocation rather than changing frame identity
or image resolution. The detector launcher now discovers the newest completed
ledger archive automatically. All three 2026-07-16 attempts remain invalid for
accuracy comparison and should not be interpreted as model results.

## Job Status Snapshot (2026-07-17 12:12 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_event_ledger_f16_r8_final64_px451584_dev20_2026-07-17` | 49200468 | Completed | 20/20; accuracy 0.7000 (14/20); complete ledger, predictions, diagnostics, and logs archived |
| `qwen3_vl_8b_vllm_semantic_candidate_logp_pivot_uniform_dev_2026-07-17` | 49200471 | Completed | 140/140; 19 disagreement score rows; accuracy 0.7857 (110/140); complete artifacts archived |
| `qwen3_vl_8b_vllm_event_ledger_groundingdino_final64_px451584_dev20_2026-07-17` | 49200868 | Failed; rerun required | Grounding DINO loaded, then failed on its first frame because FP32 pixels were passed to BF16 convolution weights; 0 detections and 0 predictions |
| `qwen3_vl_8b_vllm_event_ledger_groundingdino_final64_px451584_dev20_2026-07-17` | 49201187 | Failed; rerun required | Pixel casting passed the visual backbone, but Grounding DINO's text-enhancer still produced FP32 activations against BF16 weights; current code now loads the small detector fully in FP32 |

The event ledger does not improve this PoC slice. Uniform64 scores `17/20` and
the parent pivot scores `15/20`; the ledger scores `14/20`, adds no unique fix
over either baseline, and regresses one pivot-correct and three uniform-correct
rows. It also ties the unusually strong always-C shortcut on this 14-C/20 set,
so this formulation should not be scaled without a qualitatively different
narrative representation.

Semantic full-answer likelihood ties uniform64 at `110/140` and trails pivot by
one. Relative to pivot it changes 11 answers with a `+5/-6` crossover. On the 19
actual disagreements, pivot-context scores select `9` correctly, uniform-context
scores select `8`, and the consensus/fallback rule selects `9`, versus `10` for
retaining pivot. Separate complete-answer likelihoods are valid and auditable,
but they do not provide a promotion-worthy routing signal under this setup.

The detector failures have no experimental meaning. Casting processor inputs
fixed the first mismatch but exposed an internal FP32 text branch against BF16
weights. The small detector is now loaded fully in FP32, and the deprecated
loading argument was replaced. Its scratch cache is still empty, so the
corrected job will start cleanly and is the only one of these configurations
requiring a rerun.

## Job Status Snapshot (2026-07-22 09:00 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_uniform64_answer_cot_px451584_dev_2026-07-21` | 49237950 | Completed generation; invalid evaluation; rerun required | 140/140 predictions, but 48 responses reached the 192-token cap without `Final Answer:`; raw parser score 74/140 is not comparable |
| `qwen3_vl_8b_vllm_tcot_single_c128_sel64_px451584_dev_2026-07-21` | 49237951 | Completed selection/generation; invalid evaluation; answer rerun required | 140 selector records are reusable; 52 answers lacked a final marker; raw 42/140 is invalid |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel64_px451584_dev_2026-07-21` | 49237952 | Completed selection/generation; invalid evaluation; answer rerun required | 140 selector records and complete shared cache; 40 answers lacked a final marker; raw 68/140 is invalid |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel48_u16_px451584_dev_2026-07-21` | 49237953 | Completed selection/generation; invalid evaluation; matched-cache rerun required | 43 answers lacked a final marker; only 27 selector rows were cache hits because both dynamic jobs ran concurrently; 22/140 selected sets differ from job 49237952 |

The failure is in the answer protocol rather than model startup, frame extraction,
or structured frame selection. Across the four runs, 40-52 responses omitted the
required final marker because the rationale consumed the entire 192-token output
budget. `normalize_answer()` then sometimes treated a letter occurring in the
unfinished prose as the prediction. The raw reported accuracies (`0.3000` to
`0.5286`) are therefore excluded from the dev140 fair-comparison table.

The selector artifacts remain informative. Single-step TCoT selected a median of
12 seed IDs and saturated its 12-ID limit on 113/140 rows, but consecutive choices
collapsed to only 15.3 final frames on average after deduplication. Dynamic TCoT
selected a median of 21 seeds and produced 28.1 selected frames on average; 407/560
segment calls saturated their six-ID limit, while 77/560 correctly returned no
evidence. The coverage version increased the final mean to 43.9 frames. One sample
used the all-empty uniform fallback.

For a weak, selection-biased diagnostic only, rows that did finish with an explicit
marker scored `71/92` for uniform answer-CoT, `39/88` for single-step, `66/100` for
dynamic selected-only, and `66/97` for dynamic coverage. These are not valid
accuracy estimates, but they suggest that the answer-CoT prompt itself is unlikely
to beat the established `110/140` answer-only uniform baseline and that single-step
selection is particularly lossy.

The runner now uses the established answer-only MCQ prompt for TCoT selection
experiments, raises the answer-CoT budget to 512 tokens, and performs a strict
answer-only retry whenever `Final Answer:` is absent. The answer fingerprint was
bumped, so reruns restart predictions while reusing the completed selector caches.
The two dynamic reruns will therefore consume identical cached selections and form
a valid selected-only versus selected-plus-uniform ablation.

## Job Status Snapshot (2026-07-22 15:30 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_uniform64_answer_cot_px451584_dev_2026-07-22` | 49239709 | Completed | 140/140; valid strict-retry answers; 110/140; final artifacts archived |
| `qwen3_vl_8b_vllm_tcot_single_c128_sel64_px451584_dev_2026-07-22` | 49239710 | Completed | 140/140; 140 selector cache hits; 82/140; final artifacts archived |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel64_px451584_dev_2026-07-22` | 49240980 | Completed | 140/140; 140 selector cache hits; 95/140; final artifacts archived |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel48_u16_px451584_dev_2026-07-22` | 49240982 | Partial; resume required | 40 valid predictions and matched selections; stopped before row 41 on transient OpenCV metadata timeout |

The answer-CoT control reaches the same `110/140` accuracy as direct uniform64,
but every one of its 140 rationale calls still omitted the final marker even with
512 output tokens and therefore triggered the strict answer-only retry. Its final
letters differ from the original uniform run on only two rows, both wrong in both
runs. Explicit answer-stage rationale adds cost without measurable benefit and
should be retired for this model.

Single-step TCoT falls to `82/140`, changing 54 uniform answers with a `+8/-36`
crossover. Dynamic segmentation is better at `95/140`, but still changes 30
uniform answers with only `+6/-21`; its union oracle with uniform reaches
`116/140`, below the existing pivot/uniform oracle of `120/140`. Direct Qwen frame
selection is therefore not competitive with uniform64 or the temporal pivot in
its present low-resolution selector form.

The deficit is visible on the temporal slices that TCoT was intended to help.
Dynamic TCoT scores `6/11` on explicit recurrence questions versus `8/11` for
both uniform and pivot; `21/35` on first/last versus `26/35`; and `44/67` on
before/after versus `53/67` uniform and `54/67` pivot. Questions receiving more
than 32 dynamic frames score only `15/26` (`0.5769`), compared with `48/68`
(`0.7059`) for 25-32 frames, suggesting that broad selector output reflects
uncertainty and introduces distractors rather than useful coverage.

The coverage prefix is directionally better than selected-only on the same first
40 rows: `28/40` versus `25/40`, with a `+5/-2` crossover, but still trails
uniform's `29/40`. The first 40 selected sets are byte-equivalent at the semantic
ID level and all are cache hits, so this prefix comparison is matched. It remains
non-final until the 100-row suffix is resumed. The runner now retries video
metadata reads three times with bounded backoff; no selection recomputation is
needed.

## Job Status Snapshot (2026-07-26 20:03 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_object_labels_pivot64_px451584_dev_2026-07-26` | 49268048 | Completed | 140/140; accuracy 0.6857; detections, predictions, evaluation, diagnostics, and logs archived |
| `qwen3_vl_8b_thinking_vllm_uniform64_px451584_dev_2026-07-26` | 49268291 / 49268387 | Completed generation; invalid evaluation | first attempt stopped after 11 cached rows on the old vLLM reasoning schema; resumed run reached 140/140, but 26 outputs remained unfinished after strict retry |
| `qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26` | 49269887 | Running | 140 detection rows cached; vLLM server started; prediction file created but still empty at 20:03 CEST |
| `qwen3_vl_8b_vllm_object_panels_f56_p8_px451584_dev_2026-07-26` | 49269888 | Running | reused all 140 cached detection rows; vLLM server startup in progress; no predictions yet at 20:03 CEST |
| `qwen3_vl_8b_vllm_object_ledger_dev` | n/a | Scheduled/pending | no Slurm log or output directory yet; local environment does not expose `squeue`/`sacct`, so the job ID cannot be recovered here |

The object-label experiment is valid but substantially worse than both parent
views. Relative to the high-resolution temporal pivot, it changes 34 answers
with only five fixes and twenty regressions, falling from `111/140` to `96/140`.
Relative to uniform64 it changes 36 answers with six fixes and twenty
regressions. The detector fired very densely: every row had detections, with a
mean of 29.5 of 32 detection frames and 131.7 boxes per question. This indicates
that noisy labels and overlays dominate the visual input rather than selectively
clarifying a few ambiguous objects.

The Thinking checkpoint cannot yet be compared with Instruct. Its first job
exposed the vLLM `reasoning` versus `reasoning_content` schema difference and
was correctly resumed after the compatibility fix. However, 26 of the resumed
primary calls used their entire 2,048-token reasoning budget without a final
answer, and all 26 short retries again omitted the required marker. The raw
parser score of `83/140` is therefore excluded. Even as a weak diagnostic, its
answer distribution (`A/B/C/D = 30/38/51/21`) severely under-selects the
dev-set-majority `C` answer (`91/140` gold).

The first bounded-reasoning rerun, job `49270165`, failed before producing a
prediction because vLLM requires both `thinking_token_budget` and a server-side
reasoning delimiter configuration. The server launcher now supplies
`<think>`/`</think>` through `--reasoning-config`, in addition to the
2,048-token reasoning budget and 256-token final-answer reserve. If a marker is
still absent, the fallback uses zero additional thinking tokens; the runner
aborts rather than writing parser-dependent predictions. The failed job was
archived separately, so the next submission starts the clean `budget2048` run
ID from row zero.

## Job Status Snapshot (2026-07-26 22:16 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_thinking_budget2048_vllm_uniform64_px451584_dev_2026-07-26` | 49270165 | Failed; superseded | vLLM rejected the first request because `thinking_token_budget` lacked `--reasoning-config`; 0/140 predictions; corrected rerun `49270540` completed |
| `qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26` | 49269887 | Completed | 140/140; accuracy 0.7786; runtime 1h51m05s; complete artifacts archived |
| `qwen3_vl_8b_vllm_object_panels_f56_p8_px451584_dev_2026-07-26` | 49269888 | Completed | 140/140; accuracy 0.7643; runtime 1h49m59s; complete artifacts archived |
| `qwen3_vl_8b_vllm_object_ledger_pivot64_px451584_dev_2026-07-26` | 49269889 | Completed | 140/140; accuracy 0.7286; runtime 2h04m32s; complete artifacts archived |
| `qwen3_vl_8b_thinking_budget2048_vllm_uniform64_px451584_dev_2026-07-26` | 49270540 | Completed | 140/140 valid marked predictions; accuracy 0.6714; runtime 2h28m13s; complete artifacts archived |

Completed object-label and invalid Thinking working copies were removed only
after their canonical archives were verified byte-for-byte. Job `49270165` has
its Slurm and vLLM diagnostics under
`runs/egolongqa/qwen3_vl_8b_thinking_budget2048_vllm_uniform64_px451584_dev_2026-07-26_failed_job49270165/`.
The corrected Thinking run completed after this snapshot. Its duplicate working
copy and raw Slurm logs were removed only after byte-for-byte comparison with
the canonical archive. All completed object runs were removed from the working
output under the same rule.
Ten older July 15/22 working directories with complete canonical archives were
also removed. After job `49270540` completed and its archive was verified, the
working output tree and `slurm_logs/` were both empty.

Crop-pairs finishes close to the strongest baselines at `109/140`. Relative to
the temporal pivot it changes 21 answers with eight fixes and ten regressions;
relative to uniform64 it changes 28 with twelve fixes and thirteen regressions.
It contributes five correct answers missed by both, raising the pivot/uniform
oracle from `120/140` to `125/140`. Object crops are therefore useful as a
complementary evidence view even though they do not improve standalone
accuracy. They are markedly better than drawing labels over the full images:
crop-pairs fixes nineteen label-run errors while regressing six, a net gain of
thirteen answers.

Evidence-panels reaches `107/140`. It changes 22 pivot answers with seven fixes
and eleven regressions. It contributes four answers missed by pivot and uniform,
but only one that is also missed by crop-pairs; adding panels to the
pivot/uniform/crop oracle raises it only from `125/140` to `126/140`. Panels are
therefore a weaker, slightly redundant object view, while crop-pairs remains the
better candidate for later disagreement routing.

The object ledger reaches only `102/140`. Relative to pivot it changes 19
answers with five fixes and fourteen regressions; relative to crop-pairs it
changes 26 with eight fixes and fifteen regressions. It contributes one answer
beyond the pivot/uniform/crop/panel oracle, raising that unattainable upper bound
from `126/140` to `127/140`, but this marginal complementarity does not justify
its standalone loss or extra textual clutter.

Object-view aggregation:

| Views | Deterministic result | Oracle union |
| --- | ---: | ---: |
| Pivot + uniform | n/a | 120/140 |
| Pivot + uniform + crop-pairs | majority 112/140 | 125/140 |
| Crop for `object_detail`/`spatial`; pivot otherwise | routed 113/140 | n/a |
| Pivot + uniform + crop-pairs + panels | n/a | 126/140 |
| Pivot + uniform + crop-pairs + panels + ledger | majority 111/140 | 127/140 |

The three-view majority is a small dev-only gain over pivot (`112` versus
`111`), while adding weaker object views reduces majority accuracy. Oracle gains
show useful disagreement evidence, not a deployable selector; routing must be
validated without using each row's label.

The existing label-independent question classifier assigns 29 rows to
`object_detail` and 43 to `spatial`. Selecting the saved crop answer for those
72 rows and the pivot answer otherwise reaches `113/140`: it changes thirteen
pivot outputs with six fixes and four regressions. This rule was not fitted to
individual labels, but its aggregate score was observed on dev140 and therefore
still requires a locked confirmation set before promotion.

## Job Status Snapshot (2026-07-28 13:00 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_ug_segmented_c64_s16x2_u32_final64_px451584_dev20_2026-07-27` | 49279342 | Completed | 20/20; accuracy 0.7500; runtime 37m57s; complete predictions, selections, diagnostics, and logs archived |
| `qwen3_vl_8b_vllm_ug_temporal_pivot_c128_anc24_final64_px451584_dev20_2026-07-27` | 49280485 | Completed | 20/20; accuracy 0.8000; runtime 1h27m37s; complete predictions, selections, diagnostics, and logs archived |
| `qwen3_vl_8b_vllm_ug_c128_s16x4_dynamic_tcot_s4_sel48_u16_px451584_dev20_2026-07-27` | 49283645 | Completed | 20/20; accuracy 0.7500; runtime 1h06m50s; all 20 c128 uncertainty score sets reused from job 49280485 |
| `slurm_longqa_qwen3_ug_short_window_dev20.sh` | n/a | Not submitted | no Slurm log, working output, or archive |
| `slurm_longqa_qwen3_ug_object_crops_dev140prefix20.sh` | n/a | Not submitted | no Slurm log, working output, or archive |
| `slurm_longqa_qwen3_ug_disagreement_router_dev.sh` | n/a | Not submitted | no Slurm log, working output, or archive |

The exact temporal-dev20 references are uniform64 at `17/20` and the existing
SigLIP2 temporal pivot at `15/20`. Segment-balanced uncertainty also reaches
`15/20`: against uniform it makes no fixes and two regressions, while against
SigLIP2 pivot it makes one fix and one regression.

Uncertainty temporal pivot reaches `16/20`, improving the SigLIP2 pivot by one
net answer (`+2/-1`) but remaining one below uniform (`+0/-1`). It contributes
no answer beyond the uniform/SigLIP2 oracle, which remains `17/20`; the pilot
therefore supports a dev140 test of the retrieval signal but does not yet
establish a new best pipeline.

The uncertainty-plus-dynamic-TCoT run produces exactly the same twenty answers
as the existing SigLIP2 pivot. It selected 3-24 seed frames per sample and
formed variable packs of 21-54 frames after adding sixteen uniform frames.
Caching worked as intended: all twenty c128 uncertainty records were hits, so
the run made only eighty segment-selector calls and twenty answer calls.

For all three runs, the top-100 score distributions captured effectively all
probability mass: mean omitted tail was approximately `7e-8`, with maximum
`6.2e-7`. The entropy lower bound was therefore numerically close to
full-vocabulary entropy in this batch. The absent tail is not a plausible
explanation for the lack of a larger gain.

All three completed working directories and six raw Slurm files were removed
only after byte-for-byte comparison with their canonical archives. The working
output tree and `slurm_logs/` are empty.

## Job Status Snapshot (2026-07-29 19:43 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_ug_window_c128_w9_st3_s16_u16_final64_px451584_dev20_2026-07-29` | 49302974 | Running | 14/20 predictions and matching selections written; no stderr; rolling prefix accuracy 12/14 (0.8571). Active output and Slurm files retained in place. |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev140prefix20_2026-07-29` | 49302975 | Failed; fixed | Failed before the first prediction because the detection index reconstructed keys from rows that omit `question`, despite all 140 records carrying valid explicit `sample_key` values. The loader now prioritizes `sample_key`; coverage is 140/140. |
| `qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29` | 49302976 | Failed; fixed | Same detection-index defect as job 49302975. Pivot, uniform, crop-pair prediction inputs and detections now each cover all 140 dev keys. |
| `qwen35_9b_vllm_uniform64_px451584_smoke5_2026-07-29` | 49303038 | Failed; fixed | Qwen3.5 loaded successfully and vLLM became ready, but the first request selected FlashInfer's JIT GDN kernel and failed because no matching CUDA toolkit was available. Qwen3.5 now defaults to vLLM's supported Triton GDN prefill backend. |

The three failed jobs produced no predictions. Their Slurm and vLLM logs were
preserved under matching `runs/egolongqa/*_failed_job<id>/` directories, while
their empty working outputs and duplicate root-level Slurm files were removed.
The corrected object-crop and router jobs require fresh submissions; there is
no lost inference work to resume. The Qwen3.5 fix avoids the runtime CUDA
compilation rather than attempting to use the login node's incompatible CUDA
11.8 compiler.

## Job Status Follow-up (2026-07-29 19:59 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen35_9b_vllm_uniform64_px451584_smoke5_2026-07-29` | 49303124 | Completed | 5/5 correct; runtime 6m33s including 3m25s server warmup; Triton GDN path completed all requests; complete archive retained |
| `qwen3_vl_8b_vllm_ug_window_c128_w9_st3_s16_u16_final64_px451584_dev20_2026-07-29` | 49302974 | Completed | 15/20 (0.7500); runtime 59m56s; complete archive retained and duplicate working output removed |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev140prefix20_2026-07-29` | 49303120 | Running | corrected rerun reached 12/20 with no stderr |
| `qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29` | 49303123 | Running | corrected rerun reached 15/140 with no stderr |

The five-sample Qwen3.5 result establishes runtime compatibility only. Four of
the five gold answers are `C`, so the `1.0000` smoke accuracy is not evidence
of a model-level improvement; the controlled uniform and cached-pivot dev140
comparisons are still required.

## Job Status Follow-up (2026-07-29 20:14 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev140prefix20_2026-07-29` | 49303120 | Completed | 16/20 (0.8000); runtime 22m53s; complete archive retained and duplicate working output removed |
| `qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29` | 49303123 | Running | 58/140 predictions and matching selections written; no stderr |
| `qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29` | 49303163 | Running | vLLM ready; 16/140 predictions written; no request failures |
| `qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29` | 49303164 | Running | selection complete at 140/140; generation reached 15/140; no request failures |

On the exact object-crop prefix, uncertainty-ranked crops score `16/20`,
compared with pivot `14/20`, uniform64 `15/20`, and the original crop-pair
pipeline `15/20`. Relative to pivot, five answers changed, with three fixes and
one regression; their combined oracle is `17/20`. This is a useful pilot signal
but needs a full dev140 confirmation before it can influence a final router.

## Job Status Follow-up (2026-07-29 21:45 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29` | 49303123 | Completed | 115/140 (0.8214); runtime 1h15m09s; complete archive retained |
| `qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29` | 49303163 | Completed | 112/140 (0.8000); runtime 1h42m19s; complete archive retained |
| `qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29` | 49303164 | Completed | 113/140 (0.8071); runtime 1h42m43s; all 140 final frame-index lists exactly match the Qwen3-VL pivot reference |

Qwen3.5 provides a modest model-scaling gain under both controlled inputs.
Uniform changes 27 Qwen3-VL answers with 13 fixes and 11 regressions, while
pivot changes 35 with 16 fixes and 14 regressions. Qwen3.5 is much stronger on
non-`C` gold answers but weaker on the `C`-heavy portion of dev140: pivot moves
from non-`C`/`C` accuracies `0.7551/0.8132` to `0.8776/0.7692`. This explains
why substantial answer churn yields only a two-answer net gain.

The entropy router keeps the common answer on 107 three-view agreements, which
are correct on 94 rows. On the 33 disagreements, it reaches 21 correct versus
17 for pivot, 16 for uniform, and 15 for crop pairs. Relative to pivot, the
final router changes eleven answers with seven fixes and three regressions.
The three-view oracle remains 125/140, so ten recoverable rows are still not
selected by the entropy rule. The observed four-answer router gain is promising
but not statistically decisive on this repeatedly inspected dev split.

Qwen3.5 uniform and pivot disagree on seventeen rows and have a two-view oracle
of 120/140. Combining the current router with Qwen3.5 pivot raises the oracle to
128/140; adding Qwen3.5 uniform raises it to 130/140. These are upper bounds,
not deployable scores, but they establish that model scaling contributes
complementary errors and is now a stronger ensembling direction than another
minor frame-ranking variation.

A label-independent majority vote over the entropy router, Qwen3.5 pivot, and
Qwen3.5 uniform reaches `118/140` (`0.8429`) without another model call. It
changes 21 router answers with eleven fixes and eight regressions. The same
three systems have an oracle of `130/140`, leaving twelve additional
theoretically recoverable answers. The majority result is directly deployable
as a fixed rule, but because it was measured after inspecting dev140 it must be
confirmed on held-out examples before being treated as the expected validation
score.

## Job Status Follow-up (2026-07-30 06:00 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen35_9b_vllm_router_pivot_candidate_pair_verifier_px451584_dev_2026-07-29` | 49303799 | Completed | 115/140 (0.8214); all 31 pivot/router disagreements verified; runtime 17m15s |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev_2026-07-29` | 49303800 | Completed | 106/140 (0.7571); 140/140 selections and predictions; runtime 2h29m24s |
| `qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-29` | 49303814 | Completed | 537/700 (0.7671); validated all 700 fixed proof packs and skipped SigLIP2; runtime 7h53m11s |

The candidate-pair verifier improves the Qwen3.5 pivot candidate by two
answers, but it ties the entropy router that supplied the other candidate.
Relative to the router, it changes seventeen answers with eight fixes and
eight regressions. It therefore adds inference cost without improving the
current deployable dev score.

The uncertainty-ranked crop pilot does not generalize from its first twenty
examples. It falls from `16/20` on that prefix to `106/140` overall, three
answers below the original crop-pair run. The two crop variants differ on
fifteen examples, with six fixes and nine regressions. It contributes no new
oracle answer beyond the router and Qwen3.5 pivot, so this branch should be
retired from the main experiment queue.

The full Qwen3.5 pivot run improves the matched Qwen3-VL pivot from `528/700`
to `537/700`. Across their 167 disagreements, Qwen3.5 makes 75 fixes and 66
regressions. The net nine-answer gain is useful but small, and the previous
Qwen3 disagreement verifier remains two answers higher at `539/700`. The full
run reused the historical proof pack exactly and made no SigLIP2 calls.

The full run scores `115/140` on the dev keys and `422/560` on the remaining
validation keys. Its frame packs exactly match the earlier Qwen3.5 dev run,
but two generated answers changed and both became correct. This is minor
inference variability, not evidence that the full-run selection is better.

After checksum verification, the three duplicate working-output directories,
their six root-level Slurm files, and two empty dry-run directories were
removed. Complete canonical artifacts and copied logs remain under
`runs/egolongqa/`; `slurm_logs/` is empty.

## Job Status Follow-up (2026-07-30 11:05 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-30` | 49306485 | Completed | 113/140 (0.8071); all 19 pivot/uniform disagreements verified; runtime 16m36s |
| `qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-30` | 49306592 | Completed | 539/700 (0.7700); all 122 disagreements verified; runtime 1h23m59s |
| `qwen35_9b_vllm_uniform64_px451584_full_2026-07-30` | 49306486 | Running | 165/700 predictions at 11:04 CEST; rolling matched accuracy 125/165 (0.7576); vLLM requests continue successfully |

Replacing Qwen3 with Qwen3.5 in the original verifier improves dev140 by two
answers, from `111/140` to `113/140`. On the full validation set it ties the
existing Qwen3 verifier at `539/700`. The two full verifier outputs disagree
on 43 examples: Qwen3.5 fixes nineteen, regresses on nineteen, and changes five
answers where both remain wrong. Their two-model oracle is `558/700`, so the
models are complementary even though their aggregate accuracy is identical.

Relative to the Qwen3 full pivot, the Qwen3.5 verifier makes 29 fixes and 18
regressions across 52 changed answers, producing the same eleven-answer gain
as the original Qwen3 verifier. A simple majority over both verifiers and the
Qwen3.5 pivot reaches only `537/700`; disagreement routing rather than majority
voting is required to use the available oracle gain.

The Qwen3.5 uniform job is not failed: its root Slurm output is quiet, but its
prediction file and vLLM server log continue to grow. Its active working
directory and Slurm files must remain in place until terminal output and final
evaluation are written.

## Job Status Follow-up (2026-07-30 17:23 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen35_9b_vllm_uniform64_px451584_full_2026-07-30` | 49306486 | Interrupted after generation; resume prepared | The first 686 predictions are valid and score 531/686 (0.7741). A storage communication outage made the final 14 videos unavailable, so rows 686-699 were generated without images and final evaluation then failed while reading annotations. |

The complete 700-row attempt and its logs are preserved under
`runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_full_2026-07-30_partial_job49306486/`.
The active prediction file was truncated to the 686 validated rows. Resume it
with:

```bash
sbatch slurm_longqa_qwen35_uniform64_px451584_full_resume_20260730.sh
```

The resume launcher keeps the original run name even after midnight. LongQA
generation now fails immediately when a video path is unavailable or frame
extraction returns no images; future storage failures can no longer silently
write video-blind predictions. The dataset mount was still returning I/O errors
at this checkpoint, so the resume should be submitted only after the annotation
file and videos are readable again.

## Job Status Follow-up (2026-07-30 19:30 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen35_9b_vllm_uniform64_px451584_full_2026-07-30` | 49311311 | Failed before inference | Scratch remained unavailable; repeated `mkdir` calls returned `Communication error on send` |
| `qwen35_9b_vllm_uniform64_px451584_full_2026-07-30` | 49312012 | Completed resume | Resumed from 686 rows, generated the final 14, and evaluated all 700; final result 538/700 (0.7686) |
| `offline_majority_qwen35_pivot_qwen35_uniform_qwen3_verifier_2026-07-30` | offline | Completed | Fixed three-system vote; 552/700 (0.7886), currently the best full-validation result |

The second resume confirms that the earlier interruption was a storage outage,
not a model failure. Qwen3.5 uniform64 at 672px reaches `538/700` (`76.86%`),
with temporal accuracy `0.7702`, non-C accuracy `0.8203`, and predicted
`A/B/C/D = 63/218/343/76`. The final 14-row resume took 794 seconds including
model startup. The complete run is archived under
`runs/egolongqa/qwen35_9b_vllm_uniform64_px451584_full_2026-07-30/`; the
original attempt with 14 invalid video-blind rows remains separately preserved
for audit.

The pre-declared majority combines Qwen3.5 temporal pivot (`537/700`), Qwen3.5
uniform (`538/700`), and the Qwen3 pivot/uniform verifier (`539/700`). It reaches
`552/700` (`78.86%`), including `436/560` (`77.86%`) outside dev140. The three
systems agree on 487 questions and disagree on 213. Among the disagreements,
at least one candidate is correct on 190 questions, while majority voting is
correct on 113. The fixed triple therefore has an oracle ceiling of `629/700`
(`89.86%`), making disagreement arbitration the most promising remaining
direction.

A consolidated six-system disagreement dataset is stored under
`runs/egolongqa/model_disagreement_audit_2026-07-30/`. Detailed findings and
the recommended conditional uncertainty/TCoT and crop evaluations are in
`documentation/LONGQA_MODEL_DISAGREEMENT_ANALYSIS_2026-07-30.md`.

## Prepared Conditional Experiments (2026-07-30)

Four gated experiments were added against the fixed `552/700` majority:

| Launcher | Evaluation gate | Model calls |
| --- | --- | ---: |
| `slurm_longqa_qwen35_all_different_arbiter_full.sh` | All 700 rows; intervene only on three-way answer splits | 24 |
| `slurm_longqa_qwen35_disagreement_confidence_router_full.sh` | Score all fixed-majority disagreements for nested grouped CV | 213 rows x 4 views |
| `slurm_longqa_qwen3_conditional_ug_dynamic_tcot_dev.sh` | dev140; intervene on any three-system disagreement | 37 |
| `slurm_longqa_qwen3_conditional_delta_crops_dev.sh` | dev140; intervene on any three-system disagreement | 37 |

The candidate arbiter is restricted to the three proposed answer texts.
Conditional TCoT first shortlists frames with answer uncertainty, then performs
chronological segment selection. Delta-gated crops compare each complete source
frame with the same frame plus its proposed crop and admit the crop only when
entropy decreases by at least `0.05`. The confidence router excludes category
and temporal-operator labels and reports out-of-fold performance rather than a
training-set fit. Full commands and promotion criteria are documented in
`documentation/LONGQA_CONDITIONAL_EXPERIMENTS_2026-07-30.md`.

### Confidence-router launcher fix (2026-07-30)

Slurm job `49312267` failed before inference because its vLLM subprocess could
not import the user-site `psutil` package. No feature rows were produced. The
shared confidence-scoring launcher now exports `site.USER_SITE` through
`PYTHONPATH` and performs a `psutil` import check before model startup. The
failed attempt is archived separately under
`runs/egolongqa/qwen35_9b_vllm_pivot_uniform_mixed_confidence_router_full_2026-07-30_failed_job49312267/`.
The concurrent candidate-arbiter job `49312266` remained healthy and was not
modified.

### Candidate-constrained arbiter result (2026-07-30)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen35_9b_vllm_candidate_arbiter_all_different_mixed64_full_2026-07-30` | 49312266 | **553/700 (0.7900)** | 22m01s |

The arbiter re-evaluated only the 24 questions where Qwen3.5 pivot, Qwen3.5
uniform, and the Qwen3 verifier returned three different answers. It improved
that subset from `6/24` to `7/24`, changing 19 answers with four fixes, three
regressions, and twelve changes where both the old and new answers were wrong.
The net change is `-1` on dev140 and `+2` on the remaining 560 questions.
This is a genuine held-out improvement over the fixed majority, but the small
and unstable fix/regression balance supports retaining the arbiter as an
ensemble candidate rather than replacing the majority unconditionally.

The confidence-scoring job `49312287` reached `188/213` disagreement rows before
its original time limit expired at `2026-07-31 03:15:36`. Resume job `49315037`
completed the remaining 25 rows in `1h09m50s`, including evaluation. The final
feature file contains all 213 fixed-majority disagreements.

Nested five-fold grouped evaluation initially selected the correct candidate on
`147/213` disagreements and implied `586/700` (`0.8371`) when combined with
majority decisions elsewhere. This number is not a deployable model score. The
router included candidate option identity, and the validation labels are
strongly skewed toward option C (`444/700`). A trivial choose-C-when-available
rule already implies `580/700`.

Removing option identity reduces the fixed-split estimate to `563/700`.
Changing the number of grouped folds gives `554/700`, `563/700`, and `561/700`
for three, five, and seven folds. Across ten alternative five-fold hash
assignments, the option-invariant router averages `557.1/700` and ranges from
`554` to `560`. On the original split, visual confidence alone implies
`555/700`, blind question-and-options confidence implies `562/700`, and their
combination implies `563/700`. The apparent gain is therefore dominated by
answer-language and option-distribution priors rather than visual confidence.
The confidence features remain useful for research, but `586/700` must not be
reported as the accuracy of a completed inference pipeline.

### Strict confidence-router audit and follow-up jobs (2026-07-31)

The option-invariant router was trained only on the 37 fixed-ensemble
disagreements inside dev140 and evaluated on the 176 disagreements in the
disjoint val560 remainder. It improves the val560 majority from `436/560`
(`0.7786`) to `443/560` (`0.7911`), a held-out gain of seven answers. Applying
that dev140-fitted router to the complete validation file gives `560/700`
(`0.8000`) for analysis, but only the val560 score is uncontaminated by router
training.

Ten repeated option-invariant grouped-CV assignments imply `556` to `565`
correct answers, with a mean of `560.5/700` and standard deviation `3.63`
answers. These checks are stored under
`runs/egolongqa/qwen35_confidence_router_strict_audits_2026-07-31/`.

Two GPU diagnostics are prepared:

| Launcher | Scope | Purpose |
| --- | --- | --- |
| `slurm_longqa_qwen35_confidence_option_permutation_dev.sh` | 37 dev140 disagreements; four cyclic option placements | Measure and average away option-position sensitivity |
| `slurm_longqa_qwen35_candidate_text_likelihood_dev.sh` | 37 dev140 disagreements | Score complete candidate answer text under pivot, uniform, mixed, and blind contexts |

The option-permutation job should run first. Candidate-text likelihood is the
fallback and complementary test if cyclic placement remains unstable or the
permutation-averaged router does not exceed `116/140`.

Candidate-text job `49316362` failed before producing its first row. The vLLM
server exhausted GPU memory while computing full-vocabulary prompt
log-probabilities for a 64-frame, 672px prompt; this scoring path requires much
more temporary memory than option-letter scoring. The launcher now retains 64
frames at 448px, uses a 32K context window, lowers vLLM's GPU-memory reservation
to `0.80`, and fingerprints the pixel/context settings. The failed artifacts
are archived separately under
`runs/egolongqa/qwen35_9b_candidate_text_likelihood_disagreements_dev_2026-07-31_failed_job49316362/`.

### Option-permutation confidence result (2026-07-31)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen35_9b_vllm_confidence_option_permutation_dev_2026-07-31` | 49316361 | 117/140 (0.8357) OOF router | 1h50m25s |

Four cyclic option placements were scored for each of the 37 fixed-ensemble
disagreements and mapped back to the original answer meanings. The learned OOF
router changes nine majority answers, with five fixes and four regressions.
Only 43.24% of pivot and blind decisions, 56.76% of uniform decisions, and
54.05% of mixed-view decisions are identical across all four placements, which
confirms substantial option-order sensitivity.

The strongest label-free rule is simpler than the router: choose the proposed
candidate with the highest rotation-averaged pivot-view probability. It gets
`24/37` disagreements correct versus `20/37` for majority, yielding `120/140`
(`0.8571`) on dev140. Its eleven interventions contain seven fixes, three
regressions, and one change where both answers are wrong. The same rule without
cyclic averaging scores only `19/37`, so the gain specifically comes from
removing option-position bias. A disjoint val560 evaluation is required before
promotion.

### Candidate-text likelihood result (2026-07-31)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen35_9b_candidate_text_likelihood_disagreements_dev_2026-07-31` | 49318318 | best visual 114/140; blind 117/140 | 1h46m15s |

The rerun completed all 37 disagreements at 64 frames and 448px without memory
errors. Pivot and uniform complete-answer likelihood each imply `114/140`,
mixed implies `113/140`, and the three-view visual mean implies `112/140`.
Blind answer-text likelihood is the strongest tested rule at `117/140`, but its
gain is linguistic rather than visual and remains below the rotation-averaged
pivot rule's `120/140`. Candidate-text visual scoring should not be expanded to
val560.

The disjoint validation launcher
`slurm_longqa_qwen35_rotation_pivot_val560.sh` evaluates the promoted rule on
the 560-question complement of dev140. It scores only the pivot context for the
176 ensemble disagreements under four cyclic option placements, reducing work
from sixteen to four scoring calls per disagreement. The rule contains no
fitted parameters and the evaluator was smoke-tested to reproduce the dev140
result exactly. Promotion requires at least `441/560`, five answers above the
val560 majority baseline of `436/560`.

### Rotation-averaged pivot val560 and full result (2026-07-31)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen35_9b_vllm_rotation_avg_pivot_val560_2026-07-31` | 49320345 | **445/560 (0.7946)** | 2h46m09s |
| `qwen35_9b_vllm_rotation_avg_pivot_full_2026-07-31` | offline merge | **565/700 (0.8071)** | n/a |

The strict held-out experiment passes its promotion gate by four answers and
improves the val560 majority by nine. On 176 held-out disagreements, the rule
changes 50 majority decisions, producing 27 fixes, 18 regressions, and five
both-wrong changes. Combined with the independently computed `120/140` dev
partition, the same label-free rule reaches `565/700`, twelve answers above the
previous `553/700` candidate arbiter and thirteen above fixed majority.

The merged 700-row prediction file was reconstructed in original annotation
order and independently evaluated. A margin threshold selected only on dev140
does not improve held-out accuracy: the dev-optimal `0.16` threshold also gives
`445/560`. The unconditional rotation-averaged pivot rule is therefore the
preferred pipeline.

The follow-up error audit separates the remaining 135 errors into 64 cases
where the correct candidate was available but not selected, ten where only an
additional base run proposed it, and 61 missed by all six strong runs.
Cross-time ordering is the largest weak slice at `153/205` (`0.7463`). Detailed
category, question-type, and temporal-operator results are documented in
`documentation/LONGQA_ROTATION_ERROR_ANALYSIS_2026-07-31.md`.

### Matched shortcut controls and hypothesis-verification results (2026-08-01)

| Run | Slurm ID | Scope | Result | Runtime |
| --- | ---: | --- | ---: | ---: |
| `qwen35_9b_vllm_text_only_full_2026-07-31` | 49332216 | val700 | 452/700 (0.6457) | not recorded |
| `qwen35_9b_vllm_central_frame_px451584_full_2026-07-31` | 49332222 | val700 | 382/700 (0.5457) | not recorded |
| `qwen35_9b_hypothesis_existing_primary565_dev_2026-07-31` | 49332359 | dev140 | **119/140 (0.8500)** | 47m09s |
| `qwen35_9b_hypothesis_fresh_singlepass_primary565_dev_2026-08-01` | 49337951 | dev140 | 118/140 (0.8429) | 30m12s |
| `qwen35_9b_hypothesis_fresh_refine1_primary565_dev_2026-08-01` | 49337952 | dev140 | 117/140 (0.8357) | 46m59s |

The matched controls confirm substantial language and answer-position signal:
Qwen3.5 text-only is only eight answers above always-C, while one central frame
is ten percentage points worse than text-only. The 64-frame primary reaches
`120/140` on the same dev split, a 30-answer visual gain over text-only.

All three hypothesis runs completed 140 predictions and 37 aligned disagreement
audits. Relative to the `120/140` primary fallback, existing-evidence judging
made three fixes and four regressions; fresh candidate-balanced retrieval made
three fixes and five regressions; enabling refinement made three fixes and six
regressions. Their paired-bootstrap differences from primary all include zero.

The refinement-enabled run did not execute a second evidence round. Only 17 of
37 first reports passed strict validation, and none requested refinement; 16
reports inconsistently marked a candidate `SUPPORTED` while leaving an
applicable temporal, identity, or coverage check unresolved, three generations
ended at the output-token limit, and one cited an invalid frame ID. Strict
fallback was beneficial: accepting the 16 structurally inconsistent decisive
reports would have selected only six correct answers, versus 12 correct primary
fallbacks. Fresh SigLIP2 evidence therefore does not justify promotion, while
the bounded-refinement mechanism remains untested rather than disproven.

The aligned stratified-bootstrap report is stored at
`analysis/egolongqa/hypothesis_judge_uncertainty_dev140_2026-08-01.json`.

### Retrieval-text and embedding ablations (2026-08-01)

| Run | Slurm ID | Scope | Result | Runtime |
| --- | ---: | --- | ---: | ---: |
| `qwen35_9b_vllm_siglip2_tokensafe_temporal_pivot_anc24_final64_px451584_dev_2026-08-01` | 49338356 | dev140 | 109/140 (0.7786) | 1h41m04s |
| `qwen35_9b_vllm_qwen3vl_embed2b_temporal_pivot_anc24_final64_px451584_dev_2026-08-01` | 49339169 | dev140 | **114/140 (0.8143)** | 2h34m29s |
| `qwen35_9b_pairwise_candidate_tournament_dev_2026-08-01` | 49344682 | dev140 | failed before inference | n/a |

The token-safe SigLIP2 variant encoded the question and each option separately,
ensuring no option was lost to the text encoder's context limit. It regressed
four answers from the original Qwen3.5 SigLIP2 pivot (`113/140`), so truncation
was a real implementation risk but was not the cause of the remaining accuracy
gap.

Replacing SigLIP2 with the 2B Qwen3-VL embedding model reached `114/140`: four
fixes and three regressions relative to the original pivot, and seven fixes and
five regressions relative to uniform64. Its two-run oracles are `117/140` with
the original pivot and `119/140` with uniform64. It is therefore modestly useful
as a diverse evidence proposal, but it is slower and remains below the promoted
rotation-averaged rule (`120/140`). The embedding run used 64 final frames at
672px; mean/p95/max context fill was `56.88%/57.12%/57.44%` of 49,152 tokens.

The first pairwise-tournament attempt never loaded Qwen: the standalone wrapper
did not add the environment's user-site packages, and vLLM failed to import
`psutil`. The launcher now exports the same user-site path used by the other
Qwen jobs and passes an explicit `import psutil, vllm` preflight. Job `49344682`
is archived as a failed attempt; the corrected rerun is job `49344838` below.

### Corrected retrieval, prompting, and endpoint ablations (2026-08-01)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen35_9b_pairwise_candidate_tournament_dev_2026-08-01` | 49344838 | 116/140 (0.8286) | 34m03s |
| `qwen35_9b_vllm_siglip2_temporal_pivot_v2_anc24_final64_px451584_dev_2026-08-01` | 49344866 | 112/140 (0.8000) | 58m37s |
| `qwen35_9b_vllm_siglip2_option_quota_pivot_anc24_final64_px451584_dev_2026-08-01` | 49344867 | **117/140 (0.8357)** | 57m24s |
| `qwen35_9b_vllm_fixed_pivot_operator_adaptive_px451584_dev_2026-08-01` | 49344868 | 114/140 (0.8143) | 56m28s |
| `qwen35_9b_vllm_fixed_pivot_option_verify_px451584_dev_2026-08-01` | 49344871 | 116/140 (0.8286) | 1h40m45s |
| `qwen35_9b_vllm_uniform64_endpoint_px451584_dev_2026-08-01` | 49344873 | 116/140 (0.8286) | 1h39m53s |

The order-balanced tournament successfully evaluated all 39 candidate
disagreements, including two three-way disagreements, with no failed pair
scores. Against the promoted `120/140` fallback it made three fixes and seven
regressions; pairwise answer-text scoring is therefore rejected as a routing
rule.

Temporal pivot v2 separated requested targets from `AFTER`/`BEFORE` anchors,
used both retrieved pivot occurrences, and routed 30 compound questions through
separate temporal-clause queries. The 110 single-operator rows gained three and
lost two relative to the old pivot, but the compound route gained none and lost
two. The combined run finished one answer below the old pivot (`112` versus
`113`) and should not be promoted in its current form.

Option-quota retrieval was the strongest new ablation. For each of the 113
temporally structured questions, SigLIP2 reserved a target center for every
answer option and used the remaining target budget for the question. This
subgroup improved from `93/113` under the old pivot to `98/113` (six fixes, one
regression); the 27 global questions decreased from `20/27` to `19/27`. Overall
it made seven fixes and three regressions over the matched old pivot, reaching
`117/140`, but remained three answers below the promoted rule.

With the original pivot frames held fixed, the operator-adaptive prompt reached
`114/140` and the option-verification prompt reached `116/140`. The latter made
five fixes and two regressions relative to the baseline pivot prompt, showing a
real prompt effect, but it still made two fixes and six regressions relative to
the promoted pipeline.

Including both video endpoints raised uniform64 from `112/140` to `116/140`
(nine fixes, five regressions). It is particularly interesting on the 18
`FIRST` questions, where it scored `18/18` versus `16/18` for the promoted rule;
substituting endpoint-uniform predictions only for this predefined slice would
give `122/140` with two changed answers, both fixes. This is a dev observation,
not yet a promoted router, and requires a disjoint val560 test.

Simple majorities of the promoted rule, option-quota, endpoint uniform,
option-verification, and Qwen embeddings score only `117–119/140`. Their joint
oracle with the promoted rule is `127/140`, confirming useful diversity but no
reliable general routing signal yet. The promoted rotation-averaged rule remains
the primary pipeline at `120/140` dev and `565/700` full validation.

### Conditional delta-crop result (2026-07-30)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen3_vl_8b_vllm_conditional_delta_crops_c24_top8_d005_dev_2026-07-30` | 49312488 | 112/140 (0.8000) | 44m42s |

The fixed three-system majority scores `116/140` on the same dev keys, so
delta-gated crops regress by four answers. Among the 37 disagreement rows, the
crop answer changed the majority decision 18 times, producing seven fixes and
eleven regressions. Crops were admitted on 34/37 rows, with a mean of 4.95
admitted crops and the maximum eight crops used on fifteen rows. The `0.05`
entropy-reduction threshold therefore behaves as a permissive crop selector,
not a reliable indication that the crop-supported answer should replace the
majority. This branch should not be promoted to full validation.

The Slurm stderr contains a subset-length warning from an auxiliary evaluator,
but key-aligned manual evaluation independently confirms `112/140`; the warning
does not invalidate the result. At this checkpoint, conditional uncertainty
TCoT job `49312487` and confidence-scoring job `49312287` were still active and
their working outputs were retained.

### Conditional uncertainty-TCoT results (2026-07-30)

| Run | Slurm ID | Result | Runtime |
| --- | ---: | ---: | ---: |
| `qwen3_vl_8b_vllm_conditional_ug_c128_dynamic_tcot_sel48_u16_dev_2026-07-30` | 49312487 | 112/140 (0.8000) | 2h02m38s |
| `qwen3_vl_8b_vllm_conditional_ug_c128_dynamic_tcot_s8x3_sel48_u16_dev_2026-07-30` | 49312930 | 110/140 (0.7857) | 2h07m28s |

Both runs intervene only on the 37 dev140 questions where the fixed
three-system ensemble disagrees. The four-section policy changes 16 majority
answers, producing five fixes, nine regressions, and two changes where both
answers are wrong. The eight-section policy changes 19 answers, producing six
fixes, twelve regressions, and one both-wrong change. Splitting the timeline
more finely therefore does not improve routing: it adds one recovery but three
additional regressions.

The fixed majority plus the two TCoT policies and conditional crops has an
oracle score of `124/140`. Their distinct corrections are useful for studying a
router, but none should be promoted to a full-validation answering pipeline.
Both Slurm stderr files contain only the known subset-length warning; key-aligned
evaluation confirms the reported results.

## Dev140 Fair Comparison

The dev140 subset is `configs/egolongqa_dev140_seed20260709.json`. Previous full-validation prediction files were re-scored by matching stable `video_path||question` keys, so these numbers are directly comparable to the new dev-only runs.

| Run ID | Dev140 accuracy | Correct | Temporal acc. | Non-C acc. |
| --- | ---: | ---: | ---: | ---: |
| `offline_majority_qwen35_pivot_qwen35_uniform_qwen3_verifier_2026-07-30` | **0.8286** | **116/140** | n/a | n/a |
| `qwen3_vl_8b_vllm_conditional_ug_c128_dynamic_tcot_sel48_u16_dev_2026-07-30` | **0.8000** | **112/140** | **0.8120** | **0.8367** |
| `qwen3_vl_8b_vllm_conditional_delta_crops_c24_top8_d005_dev_2026-07-30` | **0.8000** | **112/140** | **0.8120** | **0.8163** |
| `qwen3_vl_8b_vllm_conditional_ug_c128_dynamic_tcot_s8x3_sel48_u16_dev_2026-07-30` | **0.7857** | **110/140** | **0.7970** | **0.7959** |
| `qwen35_9b_vllm_uniform64_px451584_full_2026-07-30` | **0.8071** | **113/140** | n/a | n/a |
| `offline_majority_ug_router_qwen35_pivot_qwen35_uniform_2026-07-29` | **0.8429** | **118/140** | **0.8571** | **0.9388** |
| `qwen3_vl_8b_vllm_ug_pivot_uniform_crop_router_px451584_dev_2026-07-29` | **0.8214** | **115/140** | **0.8271** | 0.7959 |
| `qwen35_9b_vllm_router_pivot_candidate_pair_verifier_px451584_dev_2026-07-29` | **0.8214** | **115/140** | **0.8346** | **0.8776** |
| `qwen35_9b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-30` | **0.8071** | **113/140** | **0.8120** | **0.7959** |
| `qwen35_9b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-29` | **0.8071** | **113/140** | 0.8195 | **0.8776** |
| `qwen35_9b_vllm_uniform64_px451584_dev_2026-07-29` | **0.8000** | **112/140** | 0.8120 | **0.9184** |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-11` | **0.7929** | **111/140** | 0.7970 | 0.7551 |
| `qwen3_vl_8b_vllm_support_contradiction_verifier_global_after_before_last_px451584_dev_2026-07-13` | **0.7929** | **111/140** | 0.7970 | 0.7551 |
| `qwen3_vl_8b_vllm_uniform64_px451584_dev_2026-07-11` | **0.7857** | **110/140** | **0.8045** | **0.7959** |
| `qwen3_vl_8b_vllm_uniform64_answer_cot_px451584_dev_2026-07-22` | **0.7857** | **110/140** | **0.8045** | **0.7959** |
| `qwen3_vl_8b_vllm_semantic_candidate_logp_pivot_uniform_dev_2026-07-17` | **0.7857** | **110/140** | 0.7970 | 0.7755 |
| `qwen3_vl_8b_vllm_siglip2_multi_event_router_c3x2_px451584_dev_2026-07-13` | **0.7857** | **110/140** | 0.7895 | 0.7551 |
| `qwen3_vl_8b_vllm_pairwise_order_swap_verifier_px451584_dev_2026-07-15` | 0.7786 | 109/140 | 0.7895 | 0.7143 |
| `qwen3_vl_8b_vllm_option_permutation_verifier_px451584_dev_2026-07-15` | 0.7786 | 109/140 | 0.7895 | 0.7347 |
| `qwen3_vl_8b_vllm_siglip2_qca_global_router_pivot24_s16_b64_px451584_dev_2026-07-13` | 0.7786 | 109/140 | 0.7820 | 0.7347 |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px200704_dev_2026-07-11` | **0.7786** | **109/140** | 0.7820 | 0.7551 |
| `qwen3_vl_8b_vllm_object_crop_pairs_f56_c8_px451584_dev_2026-07-26` | **0.7786** | **109/140** | 0.7820 | 0.7551 |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px451584_dev_2026-07-11` | 0.7714 | 108/140 | 0.7820 | 0.7755 |
| `qwen3_vl_8b_vllm_candidate_logp_verifier_blindwm01_px451584_dev_2026-07-15` | 0.7643 | 107/140 | 0.7669 | 0.7347 |
| `qwen3_vl_8b_vllm_object_panels_f56_p8_px451584_dev_2026-07-26` | 0.7643 | 107/140 | 0.7744 | 0.7347 |
| `qwen3_vl_8b_vllm_ug_object_crops_c24_top8_f56_px451584_dev_2026-07-29` | 0.7571 | 106/140 | 0.7744 | 0.7755 |
| `qwen3_vl_8b_vllm_uniform32_px451584_dev_2026-07-11` | 0.7571 | 106/140 | 0.7594 | 0.6939 |
| `qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07` | 0.7429 | 104/140 | 0.7444 | 0.7551 |
| `qwen3_vl_8b_vllm_siglip2_eventlet8x3_anc32_bnd8_final64_px200704_dev_2026-07-11` | 0.7429 | 104/140 | 0.7519 | 0.7347 |
| `qwen3_vl_8b_vllm_siglip2_qca_s16_b64_px451584_dev_2026-07-13` | 0.7429 | 104/140 | 0.7444 | 0.7143 |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_structured_anc24_final64_px200704_dev_2026-07-11` | 0.7429 | 104/140 | 0.7444 | 0.7755 |
| `qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined_dev_2026-07-09` | 0.7429 | 104/140 | 0.7444 | 0.7551 |
| `qwen3_vl_8b_vllm_cft_question_options_nms10_prompt_combined_dev_2026-07-09` | 0.7429 | 104/140 | 0.7444 | 0.7347 |
| `qwen3_vl_8b_vllm_uniform96_px451584_dev_2026-07-11` | 0.7286 | 102/140 | 0.7368 | 0.7143 |
| `qwen3_vl_8b_vllm_object_ledger_pivot64_px451584_dev_2026-07-26` | 0.7286 | 102/140 | 0.7293 | 0.6531 |
| `qwen3_vl_8b_vllm_siglip2_c128_top24_anc8_px200704_prompt_baseline_dev_2026-07-11` | 0.7214 | 101/140 | 0.7293 | 0.7347 |
| `qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06` | 0.7214 | 101/140 | 0.7293 | 0.7347 |
| `qwen3_vl_8b_vllm_32frames_px200704_2026-07-04` | 0.7143 | 100/140 | 0.7143 | 0.6939 |
| `qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08` | 0.7071 | 99/140 | 0.7143 | 0.7551 |
| `qwen3_vl_8b_vllm_siglip2_option_contrast4x3_anc16_final64_px200704_dev_2026-07-11` | 0.7000 | 98/140 | 0.7068 | 0.7347 |
| `qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07` | 0.7000 | 98/140 | 0.7068 | 0.6939 |
| `qwen3_vl_8b_vllm_pivot_qgate_lite_cap16_narr12_px200704_dev_2026-07-15` | 0.7000 | 98/140 | 0.6992 | 0.6939 |
| `qwen3_vl_8b_vllm_uniform48_px313600_dev_2026-07-11` | 0.6857 | 96/140 | 0.6992 | 0.6531 |
| `qwen3_vl_8b_vllm_siglip2_qframe_c128_h4m8l32_mixedres_dev_2026-07-15` | 0.6857 | 96/140 | 0.6842 | 0.6327 |
| `qwen3_vl_8b_vllm_siglip2_adaq_c256_f64_px200704_dev_2026-07-15` | 0.6857 | 96/140 | 0.6842 | 0.6735 |
| `qwen3_vl_8b_vllm_object_labels_pivot64_px451584_dev_2026-07-26` | 0.6857 | 96/140 | 0.6917 | 0.6327 |
| `qwen3_vl_8b_vllm_tcot_dynamic_c256_s4_sel64_px451584_dev_2026-07-22` | 0.6786 | 95/140 | 0.6767 | 0.7143 |
| `qwen3_vl_8b_thinking_budget2048_vllm_uniform64_px451584_dev_2026-07-26` | 0.6714 | 94/140 | 0.6692 | 0.7755 |
| `qwen2_5_vl_7b_hf_32frames_full_2026-07-03` | 0.6714 | 94/140 | 0.6692 | 0.7347 |
| `qwen3_vl_8b_vllm_32frames_full_2026-07-03` | 0.6714 | 94/140 | 0.6692 | 0.6735 |
| `qwen3_vl_8b_vllm_siglip2_focus_c256_a16_z025_f64_px200704_dev_2026-07-15` | 0.6643 | 93/140 | 0.6692 | 0.6939 |
| `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13` | 0.6429 | 90/140 | 0.6541 | 0.7143 |
| `qwen3_vl_8b_vllm_pivot_letterlogp_blindw05_px451584_dev_2026-07-15` | 0.6429 | 90/140 | 0.6466 | 0.6327 |
| `qwen3_vl_8b_vllm_default32_full_2026-07-02` | 0.6214 | 87/140 | 0.6165 | 0.5510 |
| `qwen2_5_vl_7b_hf_default32_full_2026-07-02` | 0.6071 | 85/140 | 0.6015 | 0.6531 |
| `qwen3_vl_8b_vllm_tcot_single_c128_sel64_px451584_dev_2026-07-22` | 0.5857 | 82/140 | 0.5940 | 0.6327 |
| `qwen3_vl_8b_vllm_openqa_minilm_uniform64_px200704_dev_2026-07-11` | 0.5286 | 74/140 | 0.5263 | 0.6531 |
| `qwen3_vl_8b_vllm_video_blind_prompt_baseline_dev_2026-07-11` | 0.5214 | 73/140 | 0.5338 | 0.4694 |

Dev140 inference:

- The corrected Qwen3-VL Thinking run is now protocol-valid but reaches only
  `94/140`, sixteen below uniform64 and seventeen below temporal pivot. It
  changes 40 uniform answers, with ten fixes and twenty-six regressions. Its
  non-`C` accuracy is strong (`38/49`), but it predicts `C` only 59 times
  against 91 `C` labels; bounded reasoning therefore amplifies answer changes
  without improving the temporal evidence. Do not scale this checkpoint in the
  current formulation.
- Thinking contributes only two answers beyond the pivot/uniform/crop oracle
  (`125/140` to `127/140`). This is insufficient to justify its 2.3K-token
  generation budget as a regular ensemble branch.
- Paired object crops reach `109/140`, two below the parent high-resolution
  pivot and one below uniform64. The standalone score is not a promotion, but
  its five unique fixes lift the pivot/uniform/crop oracle to `125/140`
  (`0.8929`). This is the first object-centric representation in the current
  batch that preserves strong aggregate accuracy and adds meaningful
  complementarity.
- Object evidence panels reach `107/140` and add four fixes beyond pivot plus
  uniform, but only one beyond crop-pairs. They preserve far more accuracy than
  text overlays, yet are not sufficiently distinct from crops to justify
  scaling both representations.
- The detector-derived object ledger reaches `102/140` and adds only one oracle
  answer beyond the stronger crop and panel views. Converting detections into
  more text again appears to distract Qwen; retire this ledger formulation.
- Automatic object labels do not improve the already strong temporal-pivot
  frames. They score `96/140`, fifteen below the unchanged pivot input. The
  detector is not sparse enough to act as a targeted visual cue: it emits a
  mean of 131.7 boxes per question across 29.5 of the 32 inspected frames.
  Label overlays should therefore not be scaled or ensembled in this form.
- The Qwen3-VL Thinking run is not included in the fair-comparison table.
  Twenty-six outputs remained unfinished even after the strict retry, making
  the reported raw `83/140` dependent on incidental letter parsing. A future
  reasoning test needs a reliable final-answer channel or a separate
  answer-only model call before its accuracy can be interpreted.
- Locked likelihood on the disjoint 560-row complement reaches `419/560`
  (`0.7482`), two answers above visual-only likelihood (`417`) and two above
  the original pivot generation on those same rows (`417`). It changes 28 pivot
  answers with a `+12/-10` crossover. This validates a small prior-fusion effect,
  but it remains nine answers below the existing full verifier on val560
  (`428/560`) and is not a new competition best.
- The candidate-likelihood resolver reaches `107/140`. All 121 pivot/uniform
  agreements are copied, including 101 correct; on the 19 disagreements it
  chooses correctly only six times, versus ten for simply retaining pivot. It
  produces one fix and five regressions relative to pivot. Averaging scores is
  the failure point: at locked weight `-0.1`, either the pivot-pack scores or the
  uniform-pack scores alone select `11/19` correctly, while their average selects
  `6/19`. Log probabilities from different visual contexts are not directly
  commensurate enough for naive averaging.
- Option-permutation voting reaches `109/140`, with two fixes and four
  regressions relative to pivot. It selects the correct candidate on `8/19`
  disagreements. Two rows tie candidate votes, one row gives neither candidate
  a vote, and eight of 76 rotated calls vote for a third option. Option order
  materially changes outputs, but majority voting is too noisy to arbitrate the
  strong pivot/uniform pair.
- The held-out fixed-weight curve is shallow: `-0.25` and locked `-0.10` both
  reach `419/560`, `-0.20` reaches `421/560`, and visual-only reaches `417/560`.
  The effect is small but not a dev-only mirage. Do not tune another global
  weight on these 560 labels; future use should keep the pre-registered `-0.10`
  or move to confidence-gated fusion.
- Candidate-constrained, order-swapped verification removes the old verifier's
  third-answer failure mode, but reaches only `109/140`: one fix and three
  regressions relative to the `111/140` pivot. Of 19 disagreements, 13 produce
  order-invariant consensus and six fall back. Candidate restriction is sound;
  the current pairwise arbitration signal is not strong enough to promote.
- Q-Frame mixed resolution reaches `96/140`, with 5 fixes and 20 regressions
  against pivot. Its small context footprint (mean 5,026 tokens) confirms the
  intended efficiency, but 32 low-resolution context frames plus only 12
  medium/high-resolution frames discard too much evidence. It adds only one
  answer to the pivot/uniform oracle (`121/140` versus `120/140`).
- Blind-prior subtraction at weight `0.5` collapses option-letter likelihood to
  `90/140`, below the always-C shortcut (`91/140`). Because both score vectors
  are saved, an offline weight sweep requires no GPU rerun: visual-only reaches
  `109/140`, while a label-tuned weight of `-0.25` reaches `114/140`. The sign is
  opposite to the proposed debiasing operation, implying that the blind scores
  act more like a useful language prior than a nuisance prior on this split.
  Treat `114/140` as exploratory validation tuning, not an unbiased benchmark.
- The narrative gate's `98/140` is not a clean rejection of narrative evidence.
  Prompt adherence failed: 81/140 rows yielded only one usable caption, 9 yielded
  none, and the mean was 5.24 of 16 requested captions. Many one-line outputs
  directly answered with an option instead of describing the timeline. The run
  still adds three answers missed by both pivot and uniform (three-way oracle
  `123/140`), but caption production/parsing must be made reliable before rerun.
- AdaQ reaches `96/140`, seven fixes and twenty regressions against the
  resolution-matched pivot/448. Its adaptive top-p pool remains broad (mean
  164 of 256 candidates), but selecting 64 relevance-ranked frames without a
  guaranteed chronological backbone still leaves large temporal holes. It adds
  only two answers beyond the strong pivot/672 plus uniform/672 pair, raising
  their oracle from `120/140` to `122/140`.
- FOCUS reaches `93/140`, six fixes and twenty-two regressions against
  pivot/448. Its dense zoom regions reduce mean timeline span and create a mean
  largest uncovered gap of 149 seconds (p95 301 seconds). It contributes three
  unique answers beyond pivot/672 plus uniform/672 (`123/140` oracle), but its
  aggregate result confirms that local zoom must be paired with uniform anchors.
- AdaQ and FOCUS are complementary to one another (`106/140` oracle), but that
  oracle is still below either strong 672px baseline. Neither configuration
  warrants a full-validation run in its current form.
- A five-fold, video-grouped offline sweep of the saved visual/blind option
  likelihoods selects small negative blind weights and reaches `112/140`, versus
  `109/140` for visual-only likelihood. Fixed weights from `-0.10` through
  `-0.25` each reach `114/140` on the complete dev slice. This is a promising
  prior-fusion feature, but it remains validation-tuned and needs a locked
  calibration/evaluation protocol before promotion.
- Standalone QCA reaches `104/140`, seven answers below the original pivot and
  six below uniform64/672. It is not a first-pass promotion candidate. Its frame
  packs are genuinely different from pivot (mean Jaccard `0.377`), and the two
  disagree on 21 rows with a `+6/-13` QCA crossover and `117/140` oracle union.
  QCA contributes two correct answers missed by both pivot and uniform, raising
  their three-way oracle from `120` to `122`; multi-event contributes only one
  and does not increase the combined four-way oracle beyond `122`. Retain QCA
  as a possible evidence source for targeted disagreement analysis, not as the
  default `GLOBAL` selector.
- The QCA-global router reaches `109/140`. Its pivot route is answer-identical
  to the original pivot for 100 rows; among 40 `GLOBAL` rows QCA changes only
  three answers, fixes none, and regresses two. This gives no reason to promote
  the current QCA allocation or run it on the full validation set.
- The multi-event router reaches `110/140`. It changes only five of the 31
  `FIRST`/`STATE_CHANGE` answers relative to pivot, with two fixes and three
  regressions. The idea remains structurally plausible, but deterministic
  clause splitting did not improve this dev slice and should not receive a full
  run without a better event parser or an evidence-recall audit.
- The gated support/contradiction verifier ties both the original verifier and
  pivot at `111/140`. It makes 17 model calls, gates two disagreements, and
  differs from the old verifier on four answers with a symmetric `+2/-2`
  crossover. Against pivot it produces three fixes and three regressions. The
  prompt/gate changes behavior but do not reduce aggregate verifier error on
  dev140; the previously completed full verifier remains the competition best.
- The frame-level follow-up is documented in
  `documentation/LONGQA_EVIDENCE_AUDIT_2026-07-14.md`. All 140 QCA rows receive
  the same `[4, ..., 4]` segment quota, so this run did not exercise dynamic
  budget allocation. In the full verifier, third-option outputs are correct
  only once in eight attempts; candidate-constrained verification is therefore
  the highest-confidence next ablation.
- The `combined` prompt uniform64 dev run ties the original full-run uniform64 on this subset (`104/140`). The prompt changed a few answer letters but did not improve aggregate accuracy on dev140.
- Coarse-to-fine retrieval also ties uniform64 at `104/140`, while using 45-57 frames and reducing mean context from 12,646 to 10,518 tokens. It is much slower end-to-end (`4h24m` versus `1h40m`) because SigLIP grounding dominates the run.
- The tie hides real complementarity: coarse-to-fine and uniform64 disagree on 26 examples; each uniquely solves 11, and an oracle union reaches `115/140` (`0.8214`). This supports confidence-based routing or a lightweight reranker, not a blind replacement of uniform sampling.
- Per-option union did not yield a score. Grounding completed, but 40/140 rows contained 65-67 images because anchors and retrieved frames used inconsistent deduplication keys. vLLM rejected the second generation request with `At most 64 image(s) may be provided`. The selector and a regression test were fixed after this run; the invalid working artifacts were removed, so this ablation requires a fresh run if it remains useful.
- Dev140 ranking differs from full-validation ranking: the hybrid run is best on full val (`498/700`) but only `99/140` on dev140. Treat dev140 as an iteration slice, not the final arbiter.
- The subset is hard but shortcut-sensitive: 133/140 questions contain temporal cues and only 49/140 gold answers are non-C, so non-C accuracy and margin over shortcut baselines should remain part of the promotion criteria.
- Uniform32 at 672px is the strongest dev140 run so far (`106/140`). Against uniform32 at 448px it uniquely fixes 10 rows and regresses on 4; against uniform64 at 448px it has a smaller `+12/-10` crossover. The oracle union with uniform64 reaches `116/140` (`0.8286`), so resolution and temporal coverage remain complementary.
- Resolution gains are not uniform. From 448px to 672px, overlapping color/appearance rises from `0.5294` to `0.7059`, fine-object identity from `0.6800` to `0.7600`, and spatial/location from `0.7407` to `0.7963`. OCR/named-detail stays flat at `0.8205`, suggesting 448px is already sufficient for many text questions in this subset.
- Open QA plus MiniLM option matching is not competitive as a standalone formulation (`74/140`). It is only one correct answer above video-blind despite using 64 frames, and uniform64 beats it by 30. Its correct predictions have a much larger median similarity margin than its errors (`0.144` versus `0.041`), so it may still provide a confidence or disagreement feature, but it should not replace direct multiple-choice prompting.
- SigLIP2 ties the original SigLIP run exactly at `101/140`, but this is not because it retrieves the same frames: mean selected-frame Jaccard overlap is only `0.374`, and the two answer sets disagree on 25 rows. Each uniquely solves 12 rows, producing an oracle union of `113/140` (`0.8071`). SigLIP2 is therefore complementary but not a drop-in aggregate improvement under the current retrieval recipe.
- SigLIP2 remains expensive on a cold cache: the dev140 run took `3h42m42s`, with roughly three hours spent grounding. Its normalized c128 image features are now cached in scratch, so query/selection ablations with the same model and candidate count should reuse them instead of repeating image encoding.
- Uniform64 at 672px is the strongest dev140 run at `110/140`, improving uniform64/448 by six net answers (`+11/-5`) and uniform32/672 by four net (`+13/-9`). Context reaches only 57.29% of the 49K window, so this configuration is viable rather than merely a context-limit stress test.
- Temporal-pivot proof packing reaches `109/140` at 448px, a five-answer improvement over uniform64/448. `AFTER` questions improve from `37/52` to `40/52`, `FIRST` from `18/26` to `19/26`, and fallback-global from `32/40` to `33/40`; this is the first grounding policy to beat matched uniform temporal coverage.
- The archived uniform64/672 and temporal-pivot dev runs disagree on 19 rows:
  pivot uniquely solves 10 and uniform uniquely solves 9, for a `120/140`
  (`0.8571`) oracle union. This matches the 19 calls reported by the original
  dev verifier. The earlier `27`/`122` note came from an inconsistent comparison
  and has been corrected here.
- Eventlet hybrid ties uniform64/448 at `104/140`. Local triplets alone do not improve aggregate accuracy, but they are useful components inside the directional pivot policy. Approximately 14 of its 64 frames are generic semantic-boundary fillers, which is likely too much budget for unconditioned scene changes.
- Option-contrastive eventlets fall to `98/140`. Balanced evidence quotas and per-option embedding margins appear to over-select visually distinctive distractors; this formulation should be deprioritized unless revisited only for verifier disagreements.
- The structured evidence prompt uses byte-identical frame packs to the unstructured temporal-pivot run but falls from `109` to `104`. It improves non-C accuracy (`0.7755`) and macro-letter accuracy (`0.7967`) while selecting C less often, but the official validation distribution is itself C-heavy (`444/700`), so the unstructured direct-MCQ prompt remains the competition choice.
- The 48-frame/560px midpoint reaches only `96/140`, despite a visual-token load comparable to stronger endpoints. Uniform sampling quality is not a smooth function of total visual tokens; the changed temporal grid likely misses events that the 32- and 64-frame grids capture. Do not promote this configuration.
- The full coverage-filled pivot/672 run is the new best result at `529/700` (`0.7557`): +15 over uniform64/672, +31 over the previous best full hybrid, and +34 over uniform64/448. It improves temporal-question accuracy to `0.7610`, non-C accuracy to `0.7422`, and macro-letter accuracy to `0.7406`.
- On the full set, coverage-filled pivot/672 beats uniform64/672 primarily on the intended operators: `AFTER` +9, `FIRST` +5, `STATE_CHANGE` +6, and `LAST` +1. `BEFORE` is tied and fallback-global loses 6, yielding the net +15. This is direct evidence that question-conditioned temporal policy is useful beyond dev140.
- Full pivot and uniform64/672 disagree on 126 rows. Pivot uniquely solves 61 and uniform uniquely solves 46; their oracle union reaches `575/700` (`0.8214`). This is now the highest-value pair for a later verifier.
- The original pivot policy at 672px is the best dev run (`111/140`), beating revised coverage fill by `+7/-4`. Because the full c128 SigLIP2 cache is now populated, a full original-pivot/672 run should no longer pay the initial 560-video image-encoding cost and is the clearest next single-model experiment.
- Uniform96/672 regresses to `102/140` while using 64.16% of a 65K window and taking 2h35m. More temporal samples are not monotonically useful; stop increasing uniform frame count beyond 64 for this model/resolution.
- A deterministic operator router over the two completed full runs reaches `535/700` (`0.7643`): use coverage-filled pivot predictions for explicit temporal operators and uniform64/672 for `GLOBAL` fallback questions. The same rule reaches `110/140` on dev140. This is an analysis result rather than an independent model run and was selected using validation labels, but the repeated `GLOBAL` preference suggests implementing the policy at frame-selection time for a cleaner test.
- InternVideo3 with the matched uniform64/672px-cap configuration reaches `90/140` (`0.6429`), 20 answers below Qwen3-VL at the same frame count and pixel cap. It predicts C only 58 times versus 91 C labels in dev140, leaving aggregate accuracy one answer below always-C (`0.6500`) while retaining stronger non-C accuracy (`0.7143`) and macro-letter accuracy (`0.7473`). This is primarily a calibration/distribution mismatch on this C-heavy split rather than uniformly failed visual reasoning.
- InternVideo3 averages 14,266 prompt tokens versus Qwen3-VL's 27,888 for the matched run, but these counts are not directly interchangeable because InternVideo3 receives one timestamped video tensor while Qwen receives separate images. Runtime is effectively matched (`1h42m20s` versus `1h43m54s`) on one H100 despite using the HF/PyTorch backend rather than vLLM.

## Inference

- The original `default32` runs were mislabeled in spirit: `--max-frames 32` only capped the frame count, while LongQA still sampled the default `frames_per_interval=4`. The 2026-07-03/04 runs are the first true 32-frame baselines.
- True 32-frame sampling is the main accuracy lever so far. Qwen2.5 improves from `0.5300` to `0.6543` (+87 correct), and Qwen3 improves from `0.5300` to `0.6429` (+79 correct).
- Increasing Qwen3 image budget from ~224x224 to ~448x448 adds another +14 correct, reaching `0.6629` (464/700). Runtime stayed almost unchanged versus Qwen3 32-frame low-res, likely because the large-image run used lower concurrency/batch size but similar total wall time.
- Hybrid 64-frame sampling is now the best single run: `0.7114` (498/700), +3 correct over uniform64, +8 correct over SigLIP c128, and +34 correct over uniform 32-frame 448px.
- Uniform 64-frame sampling at 448px remains the strongest simple baseline: `0.7071` (495/700), +5 correct over SigLIP c128 and +31 correct over uniform 32-frame 448px.
- SigLIP grounding remains complementary: selecting 32 frames from 128 candidates (`top_k=24` plus 8 uniform anchors) reaches `0.7000` (490/700). Uniform64 beats it by only +5 net, but the two runs solve different examples (`+71/-66` crossover for uniform64 versus SigLIP), suggesting a hybrid can plausibly outperform both.
- The first timeline-summary prototype does not help as implemented: `0.6643` (465/700), essentially flat versus uniform 32-frame 448px despite taking two Qwen calls per sample. The timelines are plausible but compress away option-discriminating visual details and sometimes truncate.
- The c256/top12/window1/final40 SigLIP run was cancelled during grounding after 270/700 rows. At that observed rate, full c256 grounding is too expensive without resume/caching or faster frame extraction.
- Grounding improves categories that likely require finding specific moments: `Hiking-Outdoors` (+0.1352), `Travel-Sightseeing (Indoors)` (+0.1132), `Travel-Sightseeing (Outdoors)` (+0.0678), `Gardening` (+0.0625), and `Daily Activities` (+0.0552). It drops slightly on `Pets/social gatherings` (-0.0400) and `Sightseeing` (-0.0097).
- Context is not the bottleneck for Qwen3 at 448x448 and 32 frames: max observed prompt fill was only `20.16%` of the configured 32K window. That suggests 672x672 may be worth a smoke test, but 896x896 at 32 frames is still likely too aggressive on 1x H100.
- Context is still not the bottleneck for uniform64 at 448px: max observed prompt fill was `39.11%` of 32K. This leaves room for either more frames at lower resolution or careful hybrid frame selection.
- Uniform128 at 224px timed out after 668/700 predictions and only reached partial accuracy `0.6976` (466/668). This suggests more low-resolution frames are not obviously better than 64 higher-resolution/hybrid frames, and full-run iteration is now too slow without smaller subsets or rolling partial evaluation.
- Qwen2.5/HF with 32 frames is competitive but much slower: `5h25m41s` versus `3h40m27s`/`3h42m19s` for Qwen3/vLLM. Unless Qwen2.5 has a qualitative advantage in later experiments, Qwen3/vLLM is the better iteration path.
- Category movement for Qwen3 448px over Qwen3 224px is strongest on `Outdoor Activities and Sports` (+0.1304), `Fashion Advice` (+0.1052), `Events` (+0.0909), and `Travel-Sightseeing (Outdoors)` (+0.0847), but it drops on `Gardening` (-0.0937) and `Hiking-Outdoors` (-0.0541).

## Run Notes

### `qwen2_5_vl_7b_hf_default32_full_2026-07-02`

- **Purpose:** First official starter-kit self-benchmark for Problem 3 / EgoLongQA.
- **Implementation:** Used the workshop starter kit before the LongQA frame-sampling fix. Although the command used `--max-frames 32`, LongQA sampled only 4 frames because `frames_per_interval` was still the default.
- **Command:** `python run_evaluation.py --task longqa --model-type qwen --backend hf --video-folder ../egolongqa/val --max-frames 32`
- **SLURM job:** `48945306`, launched via `slurm_longqa.sh` on partition `gpu24`.
- **Result:** Accuracy `0.5300`, with `371/700` correct.
- **Artifacts:** `runs/egolongqa/qwen2_5_vl_7b_hf_default32_full_2026-07-02/`.

### `qwen3_vl_8b_vllm_default32_full_2026-07-02`

- **Purpose:** Newer Qwen3-VL baseline for comparison against Qwen2.5-VL.
- **Implementation:** Same pre-fix LongQA sampling behavior as above: 4 sampled frames capped by `--max-frames 32`. vLLM used `max_pixels=50176` (~224x224).
- **Command:** `python run_evaluation.py --task longqa --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 8 --video-folder ../egolongqa/val --max-frames 32 --predictions output/egolongqa/qwen3_vl_8b_vllm_default32/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_default32/results.json`
- **SLURM job:** `48947596`, launched via `slurm_longqa_qwen3.sh` on partition `gpu24`.
- **Result:** Accuracy `0.5300`, with `371/700` correct.
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_default32_full_2026-07-02/`.

### `qwen2_5_vl_7b_hf_32frames_full_2026-07-03`

- **Purpose:** True 32-frame Qwen2.5-VL baseline after exposing LongQA `--frames-per-interval`.
- **Implementation:** HF backend, `frames_per_interval=32`, `max_frames=32`, `batch_size=1`.
- **Command:** `python run_evaluation.py --task longqa --model-type qwen --llm-model Qwen/Qwen2.5-VL-7B-Instruct --backend hf --video-folder ../egolongqa/val --max-frames 32 --frames-per-interval 32 --batch-size 1 --predictions output/egolongqa/qwen2_5_vl_7b_hf_32frames_full_2026-07-03/predictions.jsonl --eval-output output/egolongqa/qwen2_5_vl_7b_hf_32frames_full_2026-07-03/results.json`
- **SLURM job:** `48960137`, launched via `slurm_longqa_qwen25_32frames.sh`.
- **Runtime:** `19541` seconds (`5h25m41s`).
- **Context fill:** samples `700`; prompt tokens min/mean/p50/p95/max `36349/51494.24/63813/86285/86419`; max fill `65.93%` of 131K.
- **Result:** Accuracy `0.6543`, with `458/700` correct.
- **Category notes:** Biggest gains over the old Qwen2.5 run were `Travel-Tourism` (+0.2150), `Hobbies-Daily Activities` (+0.2000), `Hiking-Outdoors` (+0.1622), `Sightseeing` (+0.1456), and `Travel-Sightseeing (Outdoors)` (+0.1356).
- **Artifacts:** `runs/egolongqa/qwen2_5_vl_7b_hf_32frames_full_2026-07-03/`.

### `qwen3_vl_8b_vllm_32frames_full_2026-07-03`

- **Purpose:** True 32-frame Qwen3-VL baseline with low-resolution vLLM image preprocessing.
- **Implementation:** vLLM backend, `frames_per_interval=32`, `max_frames=32`, `max_pixels=50176` (~224x224), `max_model_len=16384`, `concurrency=8`, `batch_size=8`.
- **Command:** `python run_evaluation.py --task longqa --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 8 --video-folder ../egolongqa/val --max-frames 32 --frames-per-interval 32 --batch-size 8 --predictions output/egolongqa/qwen3_vl_8b_vllm_32frames_full_2026-07-03/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_32frames_full_2026-07-03/results.json`
- **SLURM job:** `48960138`, launched via `slurm_longqa_qwen3_32frames.sh`.
- **Runtime:** `13227` seconds (`3h40m27s`).
- **Context fill:** samples `700`; prompt tokens min/mean/p50/p95/max `1602/1742.89/1733/1835/1999`; max fill `12.20%` of 16K.
- **Result:** Accuracy `0.6429`, with `450/700` correct.
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_32frames_full_2026-07-03/`.

### `qwen3_vl_8b_vllm_32frames_px200704_2026-07-04`

- **Purpose:** Test whether larger per-frame resolution improves Qwen3-VL once true 32-frame sampling is enabled.
- **Implementation:** vLLM backend, `frames_per_interval=32`, `max_frames=32`, `QWEN_MAX_PIXELS=200704` (~448x448), approximate visual tokens/sample `8192`, `VLLM_QWEN_MAX_MODEL_LEN=32768`, `concurrency=2`, `batch_size=1`.
- **Command:** `python run_evaluation.py --task longqa --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 2 --video-folder ../egolongqa/val --max-frames 32 --frames-per-interval 32 --batch-size 1 --predictions output/egolongqa/qwen3_vl_8b_vllm_32frames_px200704_2026-07-04/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_32frames_px200704_2026-07-04/results.json`
- **SLURM job:** `48960435`, launched via `slurm_longqa_qwen3_32frames_large_image.sh`.
- **Runtime:** `13339` seconds (`3h42m19s`).
- **Context fill:** samples `700`; prompt tokens min/mean/p50/p95/max `5922/6336.08/6341/6443/6607`; max fill `20.16%` of 32K.
- **Result:** Accuracy `0.6629`, with `464/700` correct. This was the best uniform baseline before the 64-frame run.
- **Category notes:** Compared with Qwen3 32-frame 224px, strongest gains were `Outdoor Activities and Sports` (+0.1304), `Fashion Advice` (+0.1052), `Events` (+0.0909), `Travel-Sightseeing (Outdoors)` (+0.0847), and `Hobbies-Daily Activities` (+0.0571). Accuracy decreased on `Gardening` (-0.0937), `Hiking-Outdoors` (-0.0541), and `Shopping` (-0.0139).
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_32frames_px200704_2026-07-04/`.

### `qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06`

- **Purpose:** Test temporal grounding: retrieve question-relevant frames from a larger candidate pool before the final Qwen3-VL answer step.
- **Implementation:** SigLIP selector `google/siglip-base-patch16-224`; 128 uniformly sampled candidate frames; top-24 retrieved frames plus up to 8 uniform anchors; final frame cap 32; Qwen3-vLLM answer step with `QWEN_MAX_PIXELS=200704` (~448x448), `VLLM_QWEN_MAX_MODEL_LEN=32768`, `concurrency=2`, `batch_size=1`.
- **Command:** `python run_generate_longqa_grounded.py --video-folder ../egolongqa/val --candidate-frames 128 --top-k 24 --anchor-k 8 --window-radius 0 --final-max-frames 32 --grounder-model google/siglip-base-patch16-224 --grounder-device cuda --grounder-batch-size 32 --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 2 --batch-size 1 --output output/egolongqa/qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06/results.json --grounding-output output/egolongqa/qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06/grounding.jsonl`
- **SLURM job:** `48979671`, launched via `slurm_longqa_qwen3_siglip_grounded.sh`.
- **Runtime:** `34136` seconds (`9h28m56s`) wall-clock from the wrapper; generation script reported `34054` seconds.
- **Context fill:** samples `700`; prompt tokens min/mean/p50/p95/max `4926/6047.74/6103/6366/6607`; max fill `20.16%` of 32K.
- **Grounding stats:** all rows had 128 candidates; selected frame count varied from 26 to 32 because retrieved frames and uniform anchors can overlap; mean selected composition was 24 retrieved frames and 6.51 anchors.
- **Result:** Accuracy `0.7000`, with `490/700` correct. This was the best run before the uniform 64-frame experiment.
- **Category notes:** Compared with Qwen3 32-frame 448px uniform sampling, largest gains were `Hiking-Outdoors` (+0.1352), `Travel-Sightseeing (Indoors)` (+0.1132), `Travel-Sightseeing (Outdoors)` (+0.0678), `Gardening` (+0.0625), and `Daily Activities` (+0.0552). Accuracy decreased on `Pets, social gatherings with friends and family` (-0.0400) and `Sightseeing` (-0.0097).
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06/`.

### `qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07`

- **Purpose:** Test whether brute temporal coverage continues to improve LongQA beyond 32 frames.
- **Implementation:** vLLM backend, `frames_per_interval=64`, `max_frames=64`, `QWEN_MAX_PIXELS=200704` (~448x448), `VLLM_QWEN_MAX_MODEL_LEN=32768`, `concurrency=1`, `batch_size=1`.
- **Command:** `python run_evaluation.py --task longqa --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 1 --video-folder ../egolongqa/val --max-frames 64 --frames-per-interval 64 --batch-size 1 --predictions output/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07/results.json`
- **SLURM job:** `48984595`, launched via `slurm_longqa_qwen3_uniform_64frames_px200704.sh`.
- **Runtime:** `27692` seconds (`7h41m32s`).
- **Context fill:** samples `700`; prompt tokens min/mean/p50/p95/max `11746/12524.33/12549/12651/12815`; max fill `39.11%` of 32K.
- **Result:** Accuracy `0.7071`, with `495/700` correct. This is the best current single run.
- **Category notes:** Compared with SigLIP c128, uniform64 is stronger on `Pets/social gatherings` (+0.1200), `Hobbies-Daily Activities` (+0.0857), `Sightseeing` (+0.0777), `Fashion Advice` (+0.0527), and `Shopping` (+0.0417), but weaker on travel/hiking categories where targeted event retrieval helped.
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07/`.

### `qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07`

- **Purpose:** Prototype explicit narrative scaffolding: first generate question-aware chronological notes from 64 frames, then answer with timeline notes plus 32 answer frames.
- **Implementation:** vLLM backend, two Qwen calls per sample; timeline call used 64 frames, answer call used 32 frames; `QWEN_MAX_PIXELS=200704`, `VLLM_QWEN_MAX_MODEL_LEN=32768`, `concurrency=1`, `batch_size=1`.
- **Command:** `python run_generate_longqa_timeline.py --video-folder ../egolongqa/val --timeline-frames 64 --answer-frames 32 --max-timeline-bullets 12 --timeline-max-new-tokens 192 --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 1 --batch-size 1 --output output/egolongqa/qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07/results.json --timeline-output output/egolongqa/qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07/timeline.jsonl`
- **SLURM job:** `48984598`, launched via `slurm_longqa_qwen3_timeline_summary_proto.sh`.
- **Runtime:** `41405` seconds (`11h30m05s`).
- **Context fill:** samples/calls `1400`; prompt tokens min/mean/p50/p95/max `5987/9524.71/11782/12661/12851`; max fill `39.22%` of 32K.
- **Result:** Accuracy `0.6643`, with `465/700` correct.
- **Inference:** The naive timeline scaffold is not worth full-run budget in this form. It improved some categories over uniform 32-frame 448px (`Hobbies-Daily Activities`, `Shopping`, `Sightseeing`) but lost heavily on `Outdoor Activities and Sports`, `Travel-Tourism`, and `Pets/social gatherings`.
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07/`.

### `qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07`

- **Purpose:** Test local temporal windows around retrieved frames: 256 candidates, top-12 retrieved frames, +/-1 neighboring candidate, 8 anchors, final cap 40.
- **Implementation:** SigLIP selector `google/siglip-base-patch16-224`; answer generation never started because the job was cancelled during the grounding phase.
- **Command:** `python run_generate_longqa_grounded.py --video-folder ../egolongqa/val --candidate-frames 256 --top-k 12 --anchor-k 8 --window-radius 1 --final-max-frames 40 --grounder-model google/siglip-base-patch16-224 --grounder-device cuda --grounder-batch-size 32 --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 1 --batch-size 1 --output output/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07/results.json --grounding-output output/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07/grounding.jsonl`
- **SLURM job:** `48984596`, launched via `slurm_longqa_qwen3_siglip_c256_top12_win1_anc8_final40_px200704.sh`.
- **Status:** Cancelled at `2026-07-08T10:04:09`; partial `grounding.jsonl` contains `270/700` rows.
- **Inference:** c256/window grounding is too slow in the original no-resume implementation. The partial grounding cache should be reused after adding resume support.
- **Artifacts:** Partial cache/logs archived in `runs/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-07/`.

### `qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08`

- **Purpose:** Combine the two strongest signals so far: broad uniform temporal coverage and question-conditioned SigLIP retrieval.
- **Implementation:** 128 SigLIP candidate frames; top-16 retrieved frames plus 64 uniform anchors; final cap 64, so the final set is effectively retrieved frames plus a reduced uniform timeline; `QWEN_MAX_PIXELS=200704`, `VLLM_QWEN_MAX_MODEL_LEN=32768`, `concurrency=1`.
- **Command:** `python run_generate_longqa_grounded.py --video-folder ../egolongqa/val --candidate-frames 128 --top-k 16 --anchor-k 64 --window-radius 0 --final-max-frames 64 --grounder-model google/siglip-base-patch16-224 --grounder-device cuda --grounder-batch-size 32 --model-type qwen --llm-model Qwen/Qwen3-VL-8B-Instruct --backend vllm --tp 1 --concurrency 1 --batch-size 1 --output output/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08/predictions.jsonl --eval-output output/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08/results.json --grounding-output output/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08/grounding.jsonl`
- **SLURM job:** `48990359`, launched via `slurm_longqa_qwen3_siglip_hybrid_c128_top16_anc64_final64_px200704.sh`.
- **Runtime:** `43330` seconds (`12h02m10s`) wall-clock from the wrapper; generation script reported `43294` seconds.
- **Context fill:** samples `700`; prompt tokens min/mean/p50/p95/max `11746/12524.33/12549/12651/12815`; max fill `39.11%` of 32K.
- **Result:** Accuracy `0.7114`, with `498/700` correct. This is the best current run.
- **Category notes:** Strong versus uniform64 on `Travel-Sightseeing (Outdoors)`, `Gardening`, `Outdoor Activities and Sports`, and `Daily Activities`; weaker on `Hobbies-Daily Activities`, `Events`, `Shopping`, and `Sightseeing`.
- **Artifacts:** `runs/egolongqa/qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08/`.

### `qwen3_vl_8b_vllm_uniform128_px50176_2026-07-08`

- **Purpose:** Test whether lower-resolution 128-frame temporal coverage can outperform 64 frames at 448px.
- **Implementation:** vLLM backend, `frames_per_interval=128`, `max_frames=128`, `QWEN_MAX_PIXELS=50176` (~224x224), `VLLM_QWEN_MAX_MODEL_LEN=32768`, `concurrency=2`, `batch_size=1`.
- **SLURM job:** `48990358`, launched via `slurm_longqa_qwen3_uniform_128frames_px50176.sh`.
- **Status:** Timed out at `2026-07-09T11:40:59` after `668/700` predictions.
- **Partial result:** Prefix accuracy `0.6976` (466/668). Prefix checkpoints: 100=`0.6900`, 300=`0.6667`, 500=`0.6800`, 600=`0.6917`; last 100 rows were `0.7600`.
- **Inference:** The partial result is not promising enough to prioritize completion over hybrid/64-frame variants. If exact final accuracy is needed, resume prediction support now exists in `run_generate_longqa.py`.
- **Artifacts:** Partial predictions/logs archived in `runs/egolongqa/qwen3_vl_8b_vllm_uniform128_px50176_2026-07-08/`.

### `qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-08`

- **Purpose:** Retry the c256/top12/window1/final40 grounding experiment after adding resume support.
- **Implementation:** Same configuration as the 2026-07-07 cancelled run.
- **SLURM job:** `48990360`, launched via `slurm_longqa_qwen3_siglip_c256_top12_win1_anc8_final40_px200704.sh`.
- **Status:** Killed at `2026-07-09T13:51:42` during grounding; partial `grounding.jsonl` contains `380/700` rows. Because the default run name was date-stamped, this rerun did not reuse the previous `270/700` cache; the wrapper has now been patched to seed from the largest matching archived partial cache across dates.
- **Inference:** c256/window remains a major bottleneck. It needs a smaller candidate count, a fixed run name, precomputed/cached frame embeddings, or a small-subset gate before another full attempt.
- **Artifacts:** Partial cache/logs archived in `runs/egolongqa/qwen3_vl_8b_vllm_siglip_c256_top12_win1_anc8_final40_px200704_2026-07-08/`.

### `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13`

- **Purpose:** Hot-swap InternVideo3 into the strongest simple Qwen3 dev140 configuration to compare the model while preserving uniform temporal coverage, resolution budget, prompt, subset, and evaluation code.
- **Implementation:** `yanziang/InternVideo3-8B-Instruct` at revision `c4602918b65225650d152db2850fe34e01d21fcd`; Transformers 4.57.3; PyTorch 2.10.0+cu128; BF16 weights; SDPA attention; HF backend; batch size 1; 64 uniformly sampled frames with original timestamps; per-frame pixel range 262,144-451,584 translated to InternVideo3's total-video processor budget; deterministic direct-MCQ generation with 16 output tokens.
- **Command:** `sbatch scripts/slurm_internvideo3_dev140.sh`
- **SLURM job:** `49058687` on `gpu24-h100-08`; 1x H100, 8 CPUs, 64 GB host RAM, 3-hour limit.
- **Runtime:** `1h42m20s`, including about 28 seconds to load seven checkpoint shards.
- **Context fill:** samples `140`; prompt tokens min/mean/p50/p95/max `13857/14265.89/14271/14363/14521`; max fill `5.54%` of 262K.
- **Result:** Accuracy `0.6429`, with `90/140` correct. Temporal accuracy `0.6541`; non-C accuracy `0.7143`; macro-letter accuracy `0.7473`; margin over always-C `-0.0071`.
- **Answer distribution:** predicted `A/B/C/D = 21/35/58/26`; gold `A/B/C/D = 1/39/91/9`. InternVideo3 substantially under-selects C on this dev slice.
- **Inference:** The matched hot swap is 20 correct answers below Qwen3-VL uniform64/672. Its relatively strong non-C and macro-letter scores suggest useful visual reasoning, but direct option-letter calibration is poorly matched to the unusually C-heavy dev140 distribution.
- **Artifacts:** `runs/egolongqa/internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13/` contains committed predictions, evaluation results, and shortcut-aware diagnostics.
