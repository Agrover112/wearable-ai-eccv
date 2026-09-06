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
# Prepared Candidate-Diversity And Specialist Evidence Suite (2026-08-01)

Eight follow-up experiments were implemented without changing the frozen
`565/700` primary prediction file. The dependency-aware commands and method
descriptions are in
`documentation/LONGQA_NEXT_EXPERIMENTS_2026-08-01.md`.

The CPU-only Candidate-C audit has completed on dev140. Qwen3.5 pivot and
uniform have a `120/140` oracle. Adding the current nested Qwen3 verifier raises
the oracle to `128/140` by uniquely supplying eight correct answers. The
promoted rotation scorer selects an answer unique to Candidate C on seven rows,
five correctly. Qwen-embedding pivot produces a stronger simple majority
(`117/140`) but no additional oracle recall; endpoint uniform raises the oracle
only to `123/140`. The new uncertainty Candidate C must therefore preserve
candidate diversity, not merely match standalone accuracy. Full audit:
`analysis/egolongqa/candidate_c_replacement_audit_dev140_2026-08-01.json`.

Prepared GPU paths:

| Direction | Scope | Launcher | Dependency |
| --- | --- | --- | --- |
| Independent uncertainty Candidate C | dev140 | `slurm_longqa_qwen35_ug_candidate_c_dev.sh` | none |
| Existing candidates + uncertainty/multi-view judge | dev140 disagreements | `slurm_longqa_qwen35_ug_original_multiview_rotation_dev.sh` | uncertainty Candidate C selections |
| Replacement Candidate C + uncertainty/multi-view judge | dev140 disagreements | `slurm_longqa_qwen35_ug_candidate_c_multiview_rotation_dev.sh` | uncertainty Candidate C predictions and selections |
| Hierarchical occurrence-aware pivot | dev20 | `slurm_longqa_qwen35_hierarchical_pivot_dev20.sh` | none |
| HieraMamba interval proposals | dev20 | `slurm_hieramamba_queries_dev20.sh` then extraction and inference | sequential external environment stages |
| Qwen3.5 HieraMamba answer variants | dev20 | `slurm_longqa_qwen35_hieramamba_top{1,3}_*.sh` | normalized HieraMamba proposals |
| Targeted Qwen OCR | 49 gated dev140 rows | `slurm_longqa_qwen35_targeted_ocr_dev.sh` | reuses archived detections |
| SigLIP2 object re-identification | 39 gated dev140 rows | `slurm_longqa_qwen35_object_reid_dev.sh` | reuses archived detections |

Implementation verification: all changed Python files compile; 49 focused
function tests and six object-evidence unit tests pass; all new shell launchers
pass `bash -n`; uncertainty and specialist launcher preflights resolve their
inputs and commands successfully.

# Candidate-Diversity Pilot Results (2026-08-01)

### `qwen35_9b_vllm_hierarchical_pivot_c128_w16_local48_global16_dev20_2026-08-01`

- **Purpose:** Test a two-level temporal search that first selects coarse video windows and then searches for separated event occurrences inside those windows.
- **SLURM job:** `49347129`, launched via `slurm_longqa_qwen35_hierarchical_pivot_dev20.sh`.
- **Runtime:** `4863` seconds (`1h21m03s`). The scorer made `2532` model calls; mean context fill was `0.88%`, with a `57.47%` maximum on final answer calls.
- **Result:** `13/20` (`65.00%`). On the same 20 examples, the frozen Qwen3.5 pivot answered `12/20`; the hierarchical run changed only one answer and corrected it.
- **Inference:** The pilot gives a one-example gain but almost no candidate diversity relative to the original pivot. Its coarse-to-fine scoring cost is not justified by this dev20 result, so it should not be expanded before inspecting the one changed case and simplifying the scorer.
- **Artifacts:** `runs/egolongqa/qwen35_9b_vllm_hierarchical_pivot_c128_w16_local48_global16_dev20_2026-08-01/`.

### `qwen35_9b_vllm_targeted_qwen_ocr_dev_2026-08-01`

- **Purpose:** Transcribe question-conditioned detail crops for 49 text-sensitive dev140 questions, then answer from the transcription, chronological frames, and crops.
- **SLURM job:** `49347130`, launched via `slurm_longqa_qwen35_targeted_ocr_dev.sh`.
- **Runtime:** `3014` seconds (`50m14s`). Specialist-only gated accuracy was `39/49` (`79.59%`).
- **Overlay result:** `111/140` (`79.29%`) versus `115/140` for the frozen Qwen3.5 pivot on the same dev140 rows. The specialist changed 15 answers: five fixes and nine regressions.
- **Inference:** Explicit OCR recovers five pivot errors, but unconditional replacement loses four net answers. Retain OCR as evidence for a judge or tightly calibrated gate; do not promote the current overlay.
- **Artifacts:** `runs/egolongqa/qwen35_9b_vllm_targeted_qwen_ocr_dev_2026-08-01/`.

### `qwen35_9b_vllm_siglip2_object_reid_dev_2026-08-01`

- **Purpose:** Cluster question-conditioned object detections by SigLIP2 appearance, expose first/peak/last observations with stable track IDs, and answer 39 repeated-object or state-change questions.
- **SLURM job:** `49347131`, launched via `slurm_longqa_qwen35_object_reid_dev.sh`.
- **Runtime:** `3305` seconds (`55m05s`), including track preparation. Specialist-only gated accuracy was `34/39` (`87.18%`).
- **Overlay result:** `113/140` (`80.71%`) versus `115/140` for the frozen Qwen3.5 pivot. It changed six answers: two fixes and four regressions.
- **Inference:** The track ledger is more conservative than OCR but still degrades under unconditional replacement. Its two unique fixes may be useful as verifier evidence; the direct overlay is not promoted.
- **Artifacts:** `runs/egolongqa/qwen35_9b_vllm_siglip2_object_reid_dev_2026-08-01/`.

### HieraMamba query conversion failure and fix

- **Failed SLURM job:** `49347668`, launched via `slurm_hieramamba_queries_dev20.sh`; it failed before processing any examples because `convert_queries.py` required `bitsandbytes>=0.46.1`, which is absent from the `wearable-ai` environment.
- **Fix:** Query conversion now accepts `--quantization {auto,4bit,none}`. `auto` uses compatible 4-bit weights when available and otherwise loads the 3B converter in FP16, which fits on the requested H100. The selected mode is printed at startup.
- **Validation:** The converter compiles, its CLI resolves, and the launcher passes `bash -n`. Rerun with `sbatch slurm_hieramamba_queries_dev20.sh`.

### Still running

- **Uncertainty Candidate C:** job `49347126`, `qwen35_9b_vllm_ug_candidate_c_temporal_pivot_dev_2026-08-01`, was active at `25/140` when this log was updated. Its live Slurm and output files were deliberately left untouched. The two multi-view rotation jobs remain blocked until it completes.

### HieraMamba query conversion retry and extraction setup (2026-08-01)

- **Successful query job:** `49347740`, launched via `slurm_hieramamba_queries_dev20.sh`. The FP16 fallback processed all `20/20` dev20 samples and wrote `/scratch/inf0/user/agaur/wai-26/data/wearable-ai/hieramamba/dev20/manifest_with_queries.json`.
- **Query programs:** three samples contain one event query, ten contain two, six contain three, and one contains four. This confirms conversion completeness but does not yet validate temporal localization.
- **Held extraction submission:** job `49347761` remained pending with `user env retrieval failed requeued held`. It produced no Slurm log and never executed `slurm_hieramamba_extract_dev20.sh`.
- **Additional prerequisite found:** the documented HieraMamba environment, HieraMamba checkout, EgoVLP checkout, and EgoVLP checkpoint are not installed under `/scratch/inf0/user/agaur/wai-26/external`. Releasing the held job would therefore fail during script startup.
- **Launcher fix:** `slurm_hieramamba_bootstrap_h100.sh` now performs the one-time external setup. Extraction and inference use stable scratch defaults, validate all inputs, and no longer require command-line `--export` arguments.
- **Required sequence:** cancel held job `49347761`; run the bootstrap once; then submit extraction and inference sequentially with plain `sbatch` commands.

### HieraMamba bootstrap/order failures and fix (2026-08-01)

- **Bootstrap job `49347801`:** failed before creating the environment because Conda attempted to write `/home/agaur/.cache/conda/notices` and hit the home-directory disk quota.
- **Extraction job `49347804`:** was submitted before bootstrap succeeded. Its prerequisite check stopped immediately because the HieraMamba environment did not exist.
- **Inference job `49347807`:** was also submitted before its dependencies. It stopped immediately because neither the environment nor extracted features existed.
- **Data safety:** no partial model environment or feature files were produced. The successful 20-sample query manifest remains valid and does not need to be regenerated.
- **Cache fix:** the bootstrap now redirects `HOME`, XDG, Conda packages, pip, compiler temporaries, Torch, and Hugging Face caches to `/scratch/inf0/user/agaur/wai-26/cache/hieramamba-bootstrap` for the duration of the setup job.
- **Ordering fix:** `scripts/submit_hieramamba_dev20_after_queries.sh` submits bootstrap, extraction, and inference with strict Slurm `afterok` dependencies. This is now the recommended submission path.

### HieraMamba deferred and generated data removed (2026-08-01)

- A subsequent bootstrap retry (`49347817`) created a partial Conda environment and cloned HieraMamba, but failed while building `mamba-ssm` because pip's isolated build environment could not import PyTorch.
- HieraMamba experimentation is deferred. No extraction, temporal proposals, or Qwen answer experiments were completed, so there is no benchmark result to retain.
- Removed all HieraMamba-generated material: the partial environment and external checkout, bootstrap caches, dev20 query manifest/data, temporary Slurm logs, and HieraMamba run archives.
- Repository source, documentation, configurations, and Slurm launchers were deliberately retained for a possible later retry.

# Independent Uncertainty-Pivot Result (2026-08-02)

### `qwen35_9b_vllm_ug_candidate_c_temporal_pivot_dev_2026-08-01`

- **Naming note:** The historical run name contains `candidate_c`, but this is an independent Qwen3.5-9B prediction branch using uncertainty-selected visual evidence. It is not the old verifier-derived Candidate C and is referred to below as the **uncertainty-pivot model**.
- **Purpose:** Sample 128 candidate frames, use Qwen3.5 answer entropy and reference-event visibility to construct a 64-frame temporal evidence pack, and answer all dev140 questions independently.
- **SLURM job:** `49347126`, launched via `slurm_longqa_qwen35_ug_candidate_c_dev.sh`.
- **Runtime:** `33128` seconds (`9h12m08s`). The run made `35980` scoring/answer calls. Mean context fill was `0.54%`; final answer calls reached a maximum of `57.44%` of the 49152-token window.
- **Result:** `113/140` (`80.71%`). Temporal questions scored `82.71%`; non-temporal questions scored `42.86%`. The latter contains only seven examples and reinforces that uncertainty selection should remain a temporal specialist.
- **Clean ensemble contribution:** Qwen3.5 pivot plus uniform gives a primary-tie-broken result of `115/140` and an oracle of `120/140`. Adding uncertainty-pivot produces a three-model majority of `119/140` (`85.00%`), with five fixes and one regression. Its oracle is `121/140`, so a verifier can recover at most two additional answers from this three-model candidate set.
- **Other independent candidates:** Adding endpoint-uniform to the clean three-model pool raises candidate oracle to `124/140`, but naive four-model voting falls to `117/140`. Adding option-quota instead raises oracle to `123/140`. These branches are useful only when a verifier chooses among their proposed answers.
- **Independent-trio audit:** Across the completed normal-model branches, uncertainty-pivot + option-quota pivot + endpoint-uniform is the strongest three-way majority at `121/140` (`86.43%`), with a `124/140` (`88.57%`) candidate oracle. This combination was selected after inspecting dev140 and must therefore be frozen and evaluated on val560 before promotion.
- **Decision:** Exclude the old verifier-derived Candidate C from all subsequent ensembles. Use only independently generated model answers as candidates; verifier outputs are final decisions and must never be fed back as candidates to another verifier.
- **Artifacts:** `runs/egolongqa/qwen35_9b_vllm_ug_candidate_c_temporal_pivot_dev_2026-08-01/`.

# Clean Independent Ensemble Suite Prepared (2026-08-02)

- **Frozen candidates:** independent Qwen3.5 uncertainty-pivot, option-quota pivot, and endpoint-inclusive uniform branches. The old verifier-derived Candidate C is excluded by construction.
- **CPU majority artifact:** `qwen35_9b_clean_trio_majority_dev_2026-08-02` achieves `121/140` (`86.43%`) across 24 disagreement rows; artifacts are in `runs/egolongqa/qwen35_9b_clean_trio_majority_dev_2026-08-02/`.
- **Terminal verifier:** `slurm_longqa_qwen35_clean_trio_verifier_dev.sh` scores only candidate disagreements under four cyclic option placements and pivot, uniform, and uncertainty evidence views. It emits unrestricted, all-different-only, and conservative policies.
- **Specialist verifier ablation:** `slurm_longqa_qwen35_clean_trio_specialist_verifier_dev.sh` additionally supplies existing OCR transcriptions/detail frames and object re-identification observations/frames as uncertain judge evidence. Specialist predictions are never added to the candidate set.
- **Val560 candidates:** option-quota and endpoint-uniform use single resumable jobs. Uncertainty-pivot uses four deterministic 140-row array shards to keep each task near the measured dev140 runtime.
- **Merge safety:** shard merging validates exact key coverage and rejects missing, extra, or duplicate rows. The merge script creates val560 and full-700 candidates and majority outputs; the verifier policy is frozen on dev140 before val560 evaluation.
- **Runbook:** `documentation/LONGQA_CLEAN_ENSEMBLE_RUNBOOK_2026-08-02.md`.

### Initial clean-verifier startup failure and fix (2026-08-02)

- **Failed jobs:** clean verifier `49350995` and specialist-evidence verifier `49350996` both generated the expected `121/140` majority file, then failed before scoring any of the 24 disagreements.
- **Cause:** the shared verifier wrapper did not propagate the Conda user-site directory to the vLLM server subprocess, so vLLM could not import `psutil`.
- **Fix:** `scripts/run_longqa_clean_trio_verifier.sh` now exports the user-site directory through `PYTHONPATH`, restores the standard Qwen3.5 runtime flags, and performs an explicit `import psutil, vllm` preflight before starting work.
- **Validation:** the job environment imports `psutil 7.0.0` and `vllm 0.19.1`; both launchers resolve successfully in dry-run mode. No disagreement features were produced by the failed attempts, so both jobs must be resubmitted.
- **Concurrent candidate status:** uncertainty val560 shard 0 started successfully and reached a healthy vLLM server. Array shards 1-3, option-quota val560, and endpoint-uniform val560 had not started writing logs at the time of inspection and were left untouched.

### Clean-verifier and val560 progress (2026-08-02, evening)

- **Clean multi-view verifier:** retry job `49351024` completed all `24/24` clean-trio disagreement rows in `2362` seconds. The frozen majority remains `121/140` (`86.43%`). Applying the verifier to every disagreement reduced accuracy to `118/140`; the conservative policy reduced it to `119/140`; restricting changes to all-different rows preserved `121/140` but produced no net gain. The terminal judge is therefore not promoted over plain majority.
- **Specialist-evidence verifier:** retry job `49351025` scored `22/24` disagreements and then stopped when answer `B` was absent from the vLLM `top_logprobs` response. It has no valid final accuracy. Its partial feature cache is retained for a resumable retry, but this ablation is lower priority because the non-specialist judge already failed to improve majority.
- **Uncertainty-pivot val560:** all four array shards completed, each producing 140 unique rows. The per-shard accuracies printed by the jobs are invalid because each shard was compared with the first 140 annotations rather than its interleaved subset. After strict key-based merging in val560 order, the valid aggregate is **424/560 (`75.71%`)**. Runtime across the four shards was `31633`--`35478` seconds, or approximately `226`--`253` seconds per question; the branch stays below 300 seconds on average but still lacks per-example timing.
- **Generalization:** uncertainty-pivot falls from `113/140` (`80.71%`) on dev140 to `424/560` (`75.71%`) on held-out val560. Combined without tuning, that is `537/700` (`76.71%`). It remains useful for candidate diversity and conditional routing, not as the primary standalone model.
- **Still running:** option-quota val560 job `49350997` had produced `515/560` predictions; endpoint-uniform val560 job `49350998` had produced `252/560`. Both files were advancing normally and were left untouched. The clean val560 majority and final full-700 comparison must wait for both runs to finish.
- **Artifacts:** the correctly merged uncertainty files are in `runs/egolongqa/qwen35_9b_vllm_uncertainty_pivot_val560_2026-08-02/`; completed shard artifacts remain in their four `shard{0,1,2,3}_of4` directories.

### Clean-trio held-out result and final-push suite (2026-08-03)

- **Completed candidates:** option-quota job `49350997` finished in `22638` seconds at `429/560` (`76.61%`); endpoint-uniform job `49350998` finished in `23220` seconds at `426/560` (`76.07%`). The previously merged uncertainty-pivot result is `424/560` (`75.71%`).
- **Held-out majority:** strict key-aligned merging produced `437/560` (`78.04%`) across 132 val560 disagreement rows. Combining the frozen dev and val outputs gives `558/700` (`79.71%`) across 156 disagreements.
- **Decision:** reject the clean trio. Its dev140 result of `121/140` (`86.43%`) overestimated held-out performance by 8.39 percentage points. The validated rotation-pivot result at `565/700` remains primary.
- **Independent candidate pool:** pivot, uniform, uncertainty-pivot, option-quota, and endpoint-uniform form a five-way majority of `559/700` (`79.86%`) with a `614/700` (`87.71%`) oracle. The old verifier-derived Candidate C is excluded. On dev140, this pool scores `119/140`, has a `124/140` oracle, and disagrees on 28 rows.
- **Larger-model pilot prepared:** `slurm_longqa_qwen35_27b_primary_judge_dev.sh` invokes Qwen3.5-27B only on those 28 dev140 disagreements. Each call uses one chronological 64-frame pack with 24 pivot, 24 global, and 16 uncertainty-prioritized frames and can choose any of the four options.
- **Fixed judge policies:** the dev run emits unrestricted, candidate-supported, and all-different-only policies. A policy must reach at least `122/140`, have positive net gain, and cause at most one regression before `slurm_longqa_qwen35_27b_primary_judge_val560.sh` is eligible.
- **Diagnostic pilot:** `slurm_longqa_qwen35_27b_primary_judge_audit30.sh` evaluates all rows in a reproducible 30-error set split evenly among cross-time, shopping/OCR, and all-runs-missed failures.
- **Router analysis:** after dev/val judge merging, `scripts/run_longqa_qwen35_27b_router_cv.sh` performs grouped out-of-fold routing from five-model vote support, source agreement, the frozen majority, and the 27B answer. Option identity is excluded. This is analysis, not yet a deployable test-set router.
- **Runbook:** `documentation/LONGQA_FINAL_PUSH_RUNBOOK_2026-08-03.md`.

### Qwen3.5-27B selective judge pilot (2026-08-03)

- **SLURM job:** `49366336`, launched with `slurm_longqa_qwen35_27b_primary_judge_dev.sh`; all 28 independent-five disagreements completed in `2368` seconds.
- **Runtime:** mean per judged question `62.62` seconds, maximum `108.66` seconds, and `0/28` above the workshop's 300-second limit. Mean context fill was `57.04%`, with a `57.36%` maximum.
- **Unrestricted result:** applying the 27B answer to every disagreement reduced the independent-five majority from `119/140` to `117/140` (`83.57%`): seven answers changed, with two fixes, four regressions, and one both-wrong change. The judge supplied no correct answer absent from the five candidates.
- **Agreement pattern:** both fixes occurred on `2-2-1` plurality ties. Every regression occurred when the five candidates already had a unique plurality (`3-2`, `4-1`, or `3-1-1`).
- **Tie-break policy:** a label-free `plurality_ties_only` policy applies the 27B judge on the three dev140 plurality ties and reaches **121/140 (`86.43%`)**, with two fixes and zero regressions. There are only four such rows on val560, so its held-out test is inexpensive but its maximum contribution is necessarily small.
- **Decision:** do not run the broad 171-disagreement val560 judge. If validating this narrow tie-break, run the updated val launcher with `JUDGE_POLICY=plurality_ties_only`; it will make only four model calls. The 30-error diagnostic remains useful for determining whether 27B can recover unanimous hard failures.
- **Artifacts:** `runs/egolongqa/qwen35_27b_multievidence_primary_judge_dev140_2026-08-03/`.

### Qwen3.5-27B held-out tie-break and hard-error audit (2026-08-03)

- **Held-out tie-break:** job `49367234` called Qwen3.5-27B on the four val560 `2-2-1` plurality ties. The independent-five majority fell from `440/560` to `439/560`: three answers changed, with one fix and two regressions. Mean judged-row time was `57.61` seconds, maximum `73.78` seconds, and no row exceeded 300 seconds.
- **Full frozen policy:** merging dev and val gives `560/700` (`80.00%`), only one answer above the independent-five majority (`559/700`) and below the validated `565/700` primary. The dev tie-break pattern did not generalize and is rejected.
- **Hard-error audit:** job `49368456` judged all 30 selected primary errors in `1859` seconds. It improved the independent-five fallback from `2/30` to `7/30`, changing nine answers with six fixes and one regression. Three fixes supplied an answer absent from all five candidates.
- **Audit strata:** cross-time errors improved from `1/10` to `2/10` (two fixes, one regression); shopping/OCR improved from `1/10` to `2/10` (one fix); all-runs-missed errors improved from `0/10` to `3/10` (three fixes). This shows limited new reasoning capability, especially on upstream misses, but recall remains only `23.33%` on an error-only sample.
- **Router check:** the post-merge grouped router estimates `561/700` (`80.14%`). It is not deployment-valid because dev contains all 28 disagreement judgments while val contains only four tie judgments, leaving only 32 routed rows with inconsistent coverage. It is not promoted.
- **Decision:** neither broad 27B arbitration nor tie-only arbitration improves the final system. Retain the audit's three candidate-missing recoveries for qualitative analysis; do not run a broad val560 judge from these results.
- **Artifacts:** val tie-break in `runs/egolongqa/qwen35_27b_multievidence_primary_judge_val560_2026-08-03/`, audit in `runs/egolongqa/qwen35_27b_multievidence_primary_judge_audit30_2026-08-03/`, and merged tie policy in `runs/egolongqa/qwen35_27b_multievidence_primary_judge_plurality_ties_only_full_2026-08-03/`.

### Candidate-blind Qwen3.5-27B dev experiment prepared (2026-08-03)

- **Purpose:** determine whether the larger model can become a genuinely independent candidate rather than an arbiter anchored to previous answers.
- **Implementation:** `slurm_longqa_qwen35_27b_candidate_blind_dev.sh` runs all 140 dev questions with the same chronological 64-frame pack used by the judge: 24 temporal-pivot, 24 global, and 16 uncertainty-prioritized frames. Previous model predictions and vote counts are omitted from the prompt; all four complete answer options remain visible.
- **Promotion rule:** expand to val560 only at `>=122/140`, with several unique correct answers beyond the independent five-model oracle and no question above 300 seconds.

### Candidate-blind Qwen3.5-27B dev result (2026-08-03)

- **SLURM job:** `49369598`; all 140 candidate-blind questions completed in `7948` seconds.
- **Standalone result:** `120/140` (`85.71%`) versus `119/140` for the independent-five majority and `120/140` for the validated primary on the same subset. The model changed 20 independent-majority answers, with eight fixes and seven regressions.
- **Candidate diversity:** six fixes selected the correct answer when none of the five independent 9B branches had proposed it. Adding candidate-blind 27B raises the dev candidate oracle from `124/140` (`88.57%`) to **`130/140` (`92.86%`)**. It disagrees with the validated primary on 17 rows, and their two-model oracle is `127/140`.
- **Runtime:** mean `54.98` seconds per question, maximum `102.03` seconds, and `0/140` above 300 seconds. Mean context fill was `56.97%`, with a `57.53%` maximum.
- **Decision:** although standalone accuracy misses the original `122/140` threshold, its six unique recoveries make it the first larger-model run with substantial independent candidate value. A frozen val560 expansion is prepared as `slurm_longqa_qwen35_27b_candidate_blind_val560.sh`; merge with `scripts/merge_longqa_qwen35_27b_candidate_blind.sh` after completion.
- **Artifacts:** `runs/egolongqa/qwen35_27b_multievidence_candidate_blind_dev140_2026-08-03/`.

### Candidate-blind Qwen3.5-27B held-out and full result (2026-08-04)

- **SLURM job:** `49377227`; all 560 val questions completed in `14486` seconds.
- **Held-out result:** `449/560` (`80.18%`) versus `440/560` for the independent-five majority. The model changed 120 answers, with 58 fixes and 49 regressions; 27 fixes supplied a correct answer absent from all five 9B candidates.
- **Full result:** merging dev and val gives **`569/700` (`81.29%`)**, the strongest independent candidate and four answers above the previous validated `565/700` pipeline. It does not use the old verifier-derived Candidate C.
- **Runtime:** held-out mean `25.21` seconds per question, maximum `59.57` seconds, and `0/560` above 300 seconds. Context fill remained approximately `57%` of the 49152-token window.
- **Candidate coverage:** the five independent 9B branches have a `614/700` (`87.71%`) oracle. Adding candidate-blind 27B raises this to **`647/700` (`92.43%`)**, including 33 correct answers unavailable from every 9B branch.
- **Simple voting:** the best exploratory simple majority is uncertainty-pivot + option-quota + candidate-blind 27B at `572/700` (`81.71%`), only three answers above 27B alone. This combination was identified after inspecting all 700 labels and is not treated as held-out validation.
- **Router check:** a grouped out-of-fold router using five-model votes, source identity, and the 27B answer falls back to `559/700`; observable agreement alone cannot select the `647/700` oracle. Do not promote it.
- **Decision:** candidate-blind Qwen3.5-27B becomes the strongest Candidate-C-free standalone result. Its main value is both its `569/700` accuracy and its 33 unique oracle additions; further gains require stronger evidence-aware arbitration, not another naive majority.
- **Artifacts:** val in `runs/egolongqa/qwen35_27b_multievidence_candidate_blind_val560_2026-08-03/`, full in `runs/egolongqa/qwen35_27b_multievidence_candidate_blind_full_2026-08-03/`, and OOF router analysis in `analysis/egolongqa/qwen35_27b_candidate_blind_router_full_2026-08-04/`.

### Bounded-reasoning Qwen3.5-27B experiment prepared (2026-08-04)

- **Question:** can explicit reasoning improve the `569/700` candidate-blind 27B model without changing its visual evidence or exceeding the 300-second limit?
- **Controlled change:** retain the same chronological 64-frame input (24 temporal-pivot, 24 global, and 16 uncertainty-prioritized frames), complete question, and four options. Enable Qwen3.5 reasoning with a fixed 1024-token budget and reserve 128 additional generation tokens.
- **Answer safety:** every response must contain `Final Answer: X`. If the reasoning budget ends first, the runner performs one answer-only completion with thinking disabled; it refuses to write a parser-dependent prediction if the marker is still missing.
- **Dev launcher:** `slurm_longqa_qwen35_27b_candidate_blind_thinking_dev.sh` evaluates the fixed dev140 split. Compare against the non-thinking result of `120/140`; inspect unique fixes, regressions, final-marker retries, and maximum per-row runtime.
- **Promotion rule:** submit `slurm_longqa_qwen35_27b_candidate_blind_thinking_val560.sh` only if dev reasoning improves meaningfully over `120/140`, contributes useful new correct answers, and keeps every question below 300 seconds.
- **Final merge:** after a promoted val560 run, execute `bash scripts/merge_longqa_qwen35_27b_candidate_blind_thinking.sh` to create and evaluate the full 700-row candidate.
- **Meeting note:** `documentation/LONGQA_MEETING_BRIEF_2026-08-04.md` summarizes the current best model and the final experiment sequence.

### Qwen3.5-27B leaderboard submission export (2026-08-04)

- **Raw-output issue:** the merged candidate-blind artifact contained three empty `mcq_answer` values. In all three cases Qwen3.5-27B began a longer explanation but exhausted the 16-token output allowance before producing an option letter.
- **Cause:** the intended validity check used Python string membership, for which the empty string is considered a substring of `"ABCD"`. This prevented the recorded independent-five majority fallback from activating. The runner now tests membership in the explicit set `{A, B, C, D}`.
- **Label-free repair:** `scripts/export_longqa_submission.py` restores the already-recorded fallback answer only for an invalid model output. It does not read annotation answers when choosing a prediction. The repaired video IDs are `438a7c0e65e1ecfd.mp4`, `ce5f716477b5bf44.mp4`, and `dd33e427eabbe593.mp4`.
- **Corrected result:** all three fixed fallback answers are correct, raising the submission-ready pipeline from `569/700` to **`572/700` (`81.71%`)**.
- **Submission artifact:** `submissions/egolongqa/qwen35_27b_multievidence_candidate_blind_2026-08-04/predictions.jsonl` contains exactly 700 rows in annotation order and only the official `video_path` and `mcq_answer` fields.

### Bounded-reasoning runs stopped safely and made resumable (2026-08-04)

- **Dev attempt:** job `49410355` wrote 43/140 valid rows, then stopped when an answer-only retry did not satisfy the strict `Final Answer: X` marker check.
- **Val attempt:** job `49413748` independently reached 57/560 valid rows and stopped for the same reason. This is not a completed val560 result and must not be merged or evaluated yet.
- **Cause:** Qwen can return an unambiguous answer-only letter during the non-thinking completion retry. The validator accepted only the longer marker form, so a valid short retry could terminate the job. A genuinely unfinished retry would also terminate the complete multi-hour run instead of using the configured fallback.
- **Fix:** answer-only `A`/`B`/`C`/`D` retries are canonicalized to `Final Answer: X`. If both reasoning and retry remain incomplete, the multicandidate runner now records the completion error and uses the predetermined independent-five majority fallback for that row.
- **Resume state:** both output files retain their valid prefixes and unchanged fingerprints. Resubmitting the same launchers resumes dev at 43/140 and val at 57/560 rather than restarting.
- **Validation:** Python compilation, shell syntax, and 14 focused thinking/final-answer tests pass.

### Timestamp-aware bounded-reasoning ablation prepared (2026-08-04)

- **Controlled change:** keep the same Qwen3.5-27B model, 1024-token reasoning budget, and chronological 64-frame pack. Add a textual index mapping each image ordinal to seconds from the start of the video, then explicitly ask the model to use it for repeated events and temporal order.
- **Isolation:** timestamp injection is opt-in and is included in the run fingerprint only when enabled. Existing partial non-timestamp dev and val jobs therefore retain their original fingerprints and resume positions.
- **Launchers:** `slurm_longqa_qwen35_27b_candidate_blind_thinking_timestamps_dev.sh` runs dev140; `slurm_longqa_qwen35_27b_candidate_blind_thinking_timestamps_val560.sh` is the held-out counterpart.
- **Validation:** shell syntax, dry-run argument propagation, Python compilation, and 15 focused thinking/timestamp tests pass.

### VideoJudge-7B multimodal arbitration prepared (2026-08-04)

- **Purpose:** test a video-specialized MLLM judge on disagreements among the five independent Qwen3.5-9B branches and the candidate-blind Qwen3.5-27B primary. This does not reuse any verifier-derived answer as a candidate.
- **Checkpoint:** `VideoJudge/Qwen2.5-VL-7B-Instruct-VideoJudgeWithRubric-RS-20K`. Each complete candidate answer is scored independently against identical chronological visual evidence using the checkpoint's rubric, reasoning, and `1-5` score format. Equal top scores retain the 27B primary answer.
- **Visual transport:** selected frames are encoded as one in-memory JPEG video sequence and sent through vLLM's video input path. This preserves the checkpoint's video modality without writing temporary media into the repository.
- **Evidence ablations:** `primary64` reuses the exact 64-frame 27B pack. `union120` keeps those 64 frames and adds 56 temporally distributed, deduplicated frames from pivot, option-quota, uncertainty, uniform, and endpoint evidence.
- **Candidate ablations:** `unique` scores only answers proposed by independent branches; `alloptions` scores all four full option texts on the same disagreement rows and can therefore select an answer absent from the candidate set.
- **Launch order:** first run `slurm_longqa_videojudge7b_rubric64_dev20.sh`. After it validates model loading and output tags, the 64-frame and 120-frame unique-candidate dev140 jobs may run in parallel. Run the 120-frame all-option job only after confirming that the pointwise scores are discriminative and runtime remains below 300 seconds per question.
- **Validation:** Python compilation, shell syntax, launcher dry run, score-parser checks, and exact 120-frame evidence-cap checks pass. Full inference requires the scheduled GPU smoke test.

### VideoJudge-7B first smoke attempt (2026-08-04)

- **SLURM job:** `49421239` stopped before model loading and produced no predictions. The vLLM child process could not import `psutil` because that dependency is installed in the Python user site.
- **Fix:** the shared VideoJudge launcher now propagates the user-site path to the isolated vLLM subprocess and performs a `psutil`/`vllm` import preflight before starting inference. The corrected smoke launcher is unchanged and can be resubmitted safely.

### VideoJudge-7B smoke and dev140 results (2026-08-05)

- **Completed jobs:** corrected smoke job `49421270`, 64-frame dev140 job `49421304`, and 120-frame union dev140 job `49421321` all completed and were archived. The small `.err` files contain only expected warnings from matching subset predictions against the 700-row annotation file; there were no inference failures.
- **Smoke:** all 11 requested candidate scores parsed successfully. The run completed in `420` seconds and produced `12/20`; this subset was used only to validate loading and output structure.
- **Primary baseline on dev140:** after applying the already defined invalid-output fallback, the candidate-blind 27B prediction is `121/140` (`86.43%`). Both VideoJudge experiments acted on the same 39 disagreement rows and made 83 pointwise candidate-scoring calls.
- **Primary64 result:** `114/140` (`81.43%`). VideoJudge changed 14 answers, producing only 3 fixes versus 10 regressions and one both-wrong change. All 83 scores parsed; 13/39 judged rows tied for the highest integer score. Mean judged-row time was `54.27` seconds, p95 `86.90`, maximum `103.98`, and no row exceeded 300 seconds.
- **Union120 result:** `113/140` (`80.71%`). It changed 11 answers, with 1 fix, 9 regressions, and one both-wrong change. All scores parsed, but 15/39 rows tied at the top. Mean judged-row time increased to `99.52` seconds, p95 `155.54`, maximum `190.09`, still with no 300-second violation.
- **Interpretation:** the specialized checkpoint runs reliably but its pointwise 1-5 quality scores are not calibrated well enough to arbitrate EgoLongQA answer options. More frames do not repair this and nearly double judged-row latency. Do not promote either unique-candidate configuration to val560.
- **Next gate:** do not run the expensive 120-frame all-option dev job as currently configured. If VideoJudge is revisited, restrict it to a pairwise, order-swapped comparison or use score-token probabilities to resolve integer-score ties, and first test that intervention only on the 39 dev disagreements.

### Routing diagnosis and sparse-evidence experiments prepared (2026-08-05)

- **Ceiling diagnosis:** on held-out val560, candidate-blind 27B scores `449/560` while the six direct candidates have an oracle of `517/560` (`92.32%`). Across all 700 rows, adding the completed reasoning-enabled 27B candidate raises the seven-candidate oracle to `652/700` (`93.14%`). Crossing 90% is therefore compatible with answers already produced by the current models; candidate selection and evidence presentation are the main unresolved problems.
- **Agreement diagnosis:** on val560, 346 rows have unanimous support for the 27B answer and it is correct on 321 (`92.77%`). On the 214 disagreement rows, however, 27B is correct on only 128 while at least one direct candidate is correct on 196. Simple plurality routing does not recover this gap.
- **Category routing rejected:** choosing the best model per category from dev140 reaches `125/140` but only `436/560` held out, below the 27B result. A question-type route reaches `454/560`, a five-answer improvement but far below the available oracle. Category alone is not a reliable router.
- **Semantic router rejected:** a five-fold out-of-fold linear router using frozen MiniLM question/candidate embeddings, candidate support patterns, categories, and question types reaches `574/700` (`82.00%`). It fixes 24 primary errors but causes 19 regressions and does not beat the fixed three-model ensemble.
- **Option-retrieval audit corrected:** `157/700` existing option-quota proof packs have no option-specific centers, but `152` are `GLOBAL` questions that intentionally returned through the old generic event-retrieval path; only three `AFTER` and two `BEFORE` rows expose the empty-direction defect. The first fallback dev140 run likewise leaves 27 `GLOBAL` rows without quotas. The selector now applies option-specific retrieval to `GLOBAL` questions as well, skips meaningless temporal pivots on those rows, retries genuinely empty directional searches without the restriction, and fails loudly if any option quota remains empty.
- **Corrected retrieval prerequisite:** rerun `slurm_longqa_qwen35_option_quota_fallback_dev.sh`; it now writes the fresh `qwen35_9b_vllm_siglip2_option_quota_globalfix_dev140_2026-08-05` artifact so resume logic cannot reuse the incomplete proof pack. Balanced sparse32 and sparse option support reference this new artifact and deliberately wait for it.
- **New no-GPU ensemble:** the majority of repaired plain 27B, reasoning-enabled 27B, and endpoint-uniform 9B reaches **`582/700` (`83.14%`)**. The tie order is intentionally 27B first for all-different rows. Artifact: `runs/egolongqa/qwen35_q27_thinking_endpoint_majority_full_2026-08-05/`.
- **Mixed32 ablation completed:** job `49427671` finished all 140 rows in `3855` seconds with no stderr and no row above 300 seconds. It scores `111/140` (`79.29%`) versus its `119/140` fallback: 9 fixes, 17 regressions, net `-8`. Simply shrinking a mixed evidence pack does not solve distractor sensitivity.
- **First fallback run completed but superseded:** job `49427663` finishes at `116/140` (`82.86%`) in `5931` seconds, but inspection shows its 27 `GLOBAL` dev rows still used generic event retrieval. Keep it for audit only; downstream sparse experiments must use the new `globalfix` artifact.
- **Global option-quota correction completed:** job `49431158` finishes all 140 rows in `5905` seconds at `115/140` (`82.14%`). The stderr contains only benign Hugging Face metadata probes and evaluation subset warnings. Audit confirms 140 predictions and proof packs, zero missing A/B/C/D quotas, and zero artificial pivot centers on all 27 `GLOBAL` rows. The correction is not a stronger direct 9B answerer, but it is a valid prerequisite for the balanced sparse32 and per-option support tests.
- **Balanced sparse32 ablation:** `slurm_longqa_qwen35_27b_balanced_sparse32_dev.sh` allocates at most one retrieval center per answer option, adds equal local context, temporal-pivot evidence, and limited global coverage. It explicitly states that repeated nearby frames are one event rather than multiple votes.
- **Sparse option-support verifier:** `slurm_longqa_qwen35_27b_sparse_option_support_dev.sh` gives each answer option its own 16-frame evidence pack and obtains next-token probabilities for `SUPPORTED`, `CONTRADICTED`, and `INSUFFICIENT`. It ranks options by support margin only on model-disagreement rows, avoiding both option-order competition and the use of frame quantity as an answer prior.
- **Validation:** Python compilation, shell syntax, launcher dry runs, a forced empty-direction fallback test, exact frame-budget checks, timestamp prompts, and support-margin calculations pass. Mixed32 can run alongside corrected option retrieval; balanced sparse32 and sparse option support begin only after corrected option retrieval finishes.
- **Balanced sparse32 result:** job `49443892` completes all 140 calls in `3897` seconds with empty stderr and no row above 300 seconds. It scores `106/140` (`75.71%`) versus the `119/140` fallback, changing 33 answers for 8 fixes and 21 regressions (`-13` net). Equal option quotas and reduced frame density do not provide enough narrative evidence for direct answering and should not be promoted.
- **Sparse option-support result:** job `49443893` completes 156 option calls across the 39 model-disagreement rows in `2140` seconds with zero scoring errors. It scores `111/140` (`79.29%`). Against the plain 27B predictions it changes 22 answers, with only 2 fixes and 12 regressions. `INSUFFICIENT` is the highest-probability verdict for 103/156 option evaluations, so selecting the least-insufficient option is badly calibrated. Do not promote or run val560; retain the output only for disagreement analysis.

### Answer-calibration breakthrough (2026-08-05)

- **Missed failure mode:** gold answer positions are strongly and consistently imbalanced: full `A/B/C/D = 8/205/444/43`, dev140 `1/39/91/9`, and val560 `7/166/353/34`. Candidate-blind 27B predicts `A` 72 times but is correct on only 7; all 72 are literal `A` model outputs, not parser fallbacks. Its `C` predictions are correct on 366/378 rows. More than half of the remaining 27B errors therefore come from answer-position miscalibration.
- **Held-out calibration:** a categorical Bayesian calibrator uses only the candidate-blind 27B and endpoint-uniform 9B answer letters. Its class prior and per-model confusion tables are fitted on dev140 only with additive smoothing (`alpha=16`). Leave-one-out dev selection is `130/140`; the frozen calibrator reaches `489/560` (`87.32%`) on the untouched complement and `619/700` (`88.43%`) when reported together, versus `572/700` for 27B and `582/700` for the previous three-model majority.
- **Validated artifact:** `runs/egolongqa/qwen35_27b_endpoint_devprior_calibrated_full_2026-08-05/` contains 700 unique valid predictions, summary, and diagnostics. Repository diagnostics confirm `619/700`, non-C accuracy `81.25%`, and temporal accuracy `88.89%`. Implementation: `scripts/calibrate_longqa_answer_prior.py`. Detailed audit: `documentation/LONGQA_CALIBRATION_ANALYSIS_2026-08-05.md`.
- **Remaining headroom:** the calibrated val560 output has 71 errors; another existing direct candidate is correct on 46 of them. The next targeted GPU test is Qwen3.5-27B cyclic option rotation on only calibrator-ambiguous rows, reusing identical evidence and mapping rotated outputs back to semantic options. This directly measures and removes option-position dependence while leaving high-precision calibrated predictions untouched.
- **Label-free 27B rotation prepared:** `run_score_longqa_option_rotation.py` reuses the exact 64 recorded frames and candidate-blind prompt, places every answer text once at A/B/C/D, maps next-letter log probabilities back to original semantics, and averages mapped log probabilities. One feature run produces three frozen policies: unconditional rotation average, 3-of-4 semantic consensus, and endpoint-confirmed correction. No policy uses gold labels for selection. Run smoke5, then dev140; keep `slurm_longqa_qwen35_27b_option_rotation_val560.sh` gated until the dev rule is frozen. The runner requests top-100 token log probabilities to prevent a low-probability option letter from being omitted.
- **27B rotation smoke passed:** job `49455054` completes 20/20 calls over five questions with empty stderr. All rows contain four valid cyclic mappings, 64 unique recorded frames, normalized A/B/C/D probabilities, and no omitted option scores. Per-question four-rotation time is mean `57.09` seconds, maximum `80.43`, with zero rows above 300 seconds. All four rotations agree semantically on each smoke row, so the three policies preserve the primary `3/5`; this validates execution but is too small to measure accuracy. Proceed to dev140, not val560.
- **27B rotation dev140 completed and rejected:** job `49455914` completes 560/560 scoring calls in `9821` seconds with empty stderr, valid mappings, and no omitted option scores. Mean four-rotation time is `68.62` seconds, p95 `101.75`, maximum `119.14`, and no row exceeds 300 seconds. Unconditional mapped-logprob averaging scores `117/140` versus primary `121/140` (9 changes, 1 fix, 5 regressions); 3-of-4 consensus scores `120/140` (one regression); endpoint confirmation makes no changes and remains `121/140`. Of 140 rows, 116 are unanimous across rotations, yet the primary is wrong on eight of them. Wrong A answers often remain semantically A under every placement, showing that option position is not the main source of the calibration gap. Do not run val560 rotation.

### Final label-free verification and latency experiments prepared (2026-08-06)

- **Frozen baseline:** direct 27B, bounded-reasoning 27B, and endpoint-uniform 9B majority remains `582/700` (`83.14%`) full and `122/140` (`87.14%`) on dev140. The reproducible CPU builder is `scripts/build_longqa_qwen35_final_majority.sh`.
- **Pairwise scope:** the majority differs from repaired direct 27B on 25/700 rows, including 6/140 dev rows. `run_score_longqa_full_evidence_pairwise.py` reuses the exact 64 recorded direct-27B frames and scores the two complete candidate answer texts in both display orders. An explicit insufficient-evidence choice allows abstention.
- **Frozen conservative rule:** switch from the majority to direct 27B only when the challenger wins both orders, has at least `0.55` probability in each, insufficient evidence is at most `0.35`, and mean challenger-to-baseline odds are at least `1.5`. The existing majority is retained otherwise. The consensus-only output is diagnostic.
- **Launchers:** run `slurm_longqa_qwen35_27b_full_evidence_pairwise_dev.sh` first. Gate `slurm_longqa_qwen35_27b_full_evidence_pairwise_full.sh` on improvement over `122/140` or a clearly correct high-confidence change without regression.
- **Fresh latency smoke:** `slurm_longqa_qwen35_final_pipeline_latency_dev20.sh` sequentially executes fresh SigLIP2 selection, fresh uncertainty scoring, endpoint 9B, direct 27B, reasoning 27B, and final voting on one H100. Model startup is included once per stage, and `latency_summary.json` reports the conservative amortized time against the 300-second limit.
- **Validation:** Python compilation, shell syntax, four focused pairwise-policy tests, exact 64-frame checks on all 25 target rows, launcher dry run, and a complete CPU rebuild pass. The rebuilt ensemble reproduces all 700 answer letters and `582/700` exactly. Temporary validation artifacts were removed.

### Full-evidence pairwise verifier dev140 result (2026-08-06)

- **SLURM job:** `49499116` completed normally with empty stderr. The scorer processed the six dev140 rows where the `582/700` majority differs from repaired direct Qwen3.5-27B, making 12 model calls with the exact 64 recorded evidence frames.
- **Conservative policy:** `122/140` (`87.14%`), identical to the majority baseline, with zero switches. No row satisfied the frozen two-order probability, insufficient-evidence, and odds requirements.
- **Consensus diagnostic:** `121/140` (`86.43%`) with one switch. It changed `88e8cf11f150ba5e.mp4` from the correct majority answer `C` to the incorrect direct-27B answer `B`, so loosening the rule causes a regression.
- **Missed recoveries:** direct 27B was correct on two of the six disagreements, but the verifier did not recover either. One was judged insufficient in both orders; the other changed its preferred candidate when answer order was reversed. This indicates that the same 64-frame evidence does not let the 27B model reliably identify its own useful minority answers.
- **Runtime:** pairwise inference took `727` seconds overall. Applied rows averaged `47.42` seconds, had a `75.87`-second maximum, and had zero rows above 300 seconds. Mean context fill was `56.96%` of 49,152 tokens.
- **Decision:** reject `slurm_longqa_qwen35_27b_full_evidence_pairwise_full.sh`; it failed the predeclared dev gate and should not consume a full run. Retain the existing `582/700` label-free majority and proceed with `slurm_longqa_qwen35_final_pipeline_latency_dev20.sh`.

### Fresh end-to-end latency audit (2026-08-06)

- **SLURM job:** `49499389` completed all six stages on the fixed dev20 subset with fresh SigLIP2 and uncertainty caches. Every component and the final vote produced 20 valid predictions; stderr contains only non-fatal Hugging Face metadata and subset-evaluation warnings.
- **Measured runtime:** SigLIP2 pivot `3328` seconds, uncertainty selection `5286` seconds, endpoint-uniform 9B `1178` seconds, direct 27B `1474` seconds, reasoning 27B `2698` seconds, and voting below the one-second timer resolution. Total wall time was **`13964` seconds**, or **`698.2` seconds per question** amortized over 20 questions.
- **Limit check:** the complete sequential pipeline is **not compliant** with the workshop's 300-second per-question limit. Uncertainty selection is the largest cost at `264.3` seconds per question, followed by SigLIP2 pivot at `166.4`; together they consume `430.7` seconds before the three answer branches are complete.
- **Audit accuracy:** the fresh majority scores `13/20` (`65.00%`). This subset is for timing, not model selection: its always-C baseline is `14/20`, and the cached full-run majority scores `12/20` on the same rows.
- **Reproducibility:** endpoint-uniform reproduces all 20 cached answers. The fresh direct and reasoning 27B passes differ from their cached counterparts on one and four rows respectively, leading to two final-vote changes. The fresh reasoning pass happens to score `15/20` versus `11/20` cached, but this small, skewed subset and ordinary generation nondeterminism do not establish an accuracy improvement.
- **Decision:** retain the `582/700` majority as the accuracy reference, but do not describe its current uncached sequential implementation as time-compliant. A compliant submission path must remove or precompute the two retrieval stages, reuse one evidence pack across answer branches, or execute independent branches concurrently under the organizer's accounting rules.
- **Artifacts:** `runs/egolongqa/latency_qwen35_final_dev20_49499389_*`; authoritative timing is in `latency_qwen35_final_dev20_49499389_majority/latency_summary.json`.

### Final-day label-invariant routing and candidate suite (2026-08-07)

- **Candidate scope:** direct 27B, reasoning 27B, endpoint-uniform 9B, and rotation-pivot 9B disagree on 35 dev140 rows. The current majority is `122/140`; the four-candidate oracle is `131/140`, leaving nine primary errors recoverable from already proposed answers.
- **Core grouped OOF router:** candidate-correctness scoring uses semantic support, source/evidence family, temporal operator, coarse question type, and cached retrieval statistics, with no A/B/C/D identity feature. It makes one switch, which is a regression, and finishes at `121/140`.
- **Rotation-augmented OOF router:** adding cached four-rotation probabilities after mapping every rotation back to original semantic answers also finishes at `121/140`. It makes two switches: zero fixes, one regression, and one both-wrong change.
- **Decision:** neither router passes the dev gate. Do not run `slurm_longqa_label_invariant_router_val560.sh` or the prepared 179-disagreement val rotation job. Their implementations remain as reproducible negative results.
- **New GPU priority:** `slurm_longqa_qwen35_27b_uniform64_endpoint_dev.sh` tests direct Qwen3.5-27B on the tested 64-frame endpoint-inclusive grid at `451584` pixels without retrieval overhead.
- **Complementary GPU candidate:** `slurm_longqa_qwen35_27b_option_quota_globalfix_dev.sh` reuses all 140 corrected option-quota proof packs from job `49431158` and performs only Qwen3.5-27B answering. It does not rerun SigLIP2 selection.
- **Promotion:** val560 launchers and endpoint merging are prepared but gated on dev accuracy and primary-error recovery. Runbook: `documentation/LONGQA_FINAL_DAY_ARBITRATION_2026-08-07.md`.

### Qwen3.5-27B endpoint and corrected option-quota dev results (2026-08-07)

- **Endpoint-inclusive 27B:** job `49523414` completed all `140/140` rows in `8062` seconds, including a `291`-second model startup. It scores **`130/140` (`92.86%`)**, the strongest label-free dev140 result from a single candidate. Mean context fill is `56.88%`, p95 `57.12%`, and maximum `57.44%` of the 49,152-token window.
- **Robustness diagnostics:** endpoint 27B scores `89.80%` on non-C gold answers and `95.42%` macro accuracy across answer positions. Its prediction distribution is `A/B/C/D = 5/35/87/13`, compared with gold `1/39/91/9`; the gain is not explained by always predicting the dominant answer position.
- **Complementarity with the `122/140` primary:** the two systems disagree on 17 rows. Endpoint 27B recovers 12 primary errors and loses four primary-correct answers, producing a two-model oracle of **`134/140` (`95.71%`)**. Eight of the 12 recoveries are `GLOBAL` questions, while there are no primary-only losses in that operator group.
- **Corrected option-quota 27B:** job `49523426` reused the 140 validated proof packs from job `49431158` and made no new SigLIP2 retrieval pass. It completes in `7977` seconds and scores `123/140` (`87.86%`). Against the primary it makes nine fixes and eight regressions across 19 disagreements, for a `131/140` oracle.
- **Endpoint versus option-quota:** endpoint 27B and option-quota 27B have a `133/140` oracle, so option-quota contributes only three additional correct answers beyond endpoint while being seven answers weaker standalone. A three-way pool of primary, endpoint, and option-quota has a `135/140` oracle, but naive majority does not improve endpoint alone.
- **Frozen operator route:** the stricter v2 temporal compiler marks 27 rows as `GLOBAL`. Using endpoint 27B only on those rows and the prior primary otherwise scores `129/140` (`92.14%`), changing eight answers for a net gain of seven. The older complementarity report groups 40 rows as `GLOBAL` under the v1 compiler; the frozen route deliberately uses the narrower v2 definition.
- **Decision:** promote endpoint 27B immediately to the untouched val560 complement using `slurm_longqa_qwen35_27b_uniform64_endpoint_val560.sh`. Evaluate both endpoint standalone and the now-frozen v2 `GLOBAL` route exactly once after val completes. Do not yet spend a second full 27B pass on option-quota val560 unless compute remains after endpoint finishes.
- **Artifacts:** `runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_dev140_2026-08-07/` and `runs/egolongqa/qwen35_27b_option_quota_globalfix_dev140_2026-08-07/`.

### Corrected option-quota val560 pipeline prepared (2026-08-07)

- **Purpose:** evaluate corrected option-quota 27B as an independent held-out candidate. It adds three correct dev140 answers beyond endpoint 27B, although its standalone dev score is lower.
- **Efficient chain:** `scripts/submit_longqa_qwen35_27b_option_quota_val560.sh` submits corrected frame selection, a dependent Qwen3.5-27B answer pass, and a dependent CPU merge. The selection stage now uses `--selection-only`, so it creates the required proof packs without spending time on an unused Qwen3.5-9B answer pass.
- **Frozen comparison:** after endpoint and option-quota full artifacts both exist, `scripts/merge_longqa_qwen35_27b_option_quota.sh` also constructs the three-way majority of endpoint 27B, option-quota 27B, and the previous primary. Endpoint is the deterministic fallback when all three answers differ. This policy equals endpoint alone on dev140 (`130/140`); its value must be determined on the untouched val560 complement.
- **Launch:** run `bash scripts/submit_longqa_qwen35_27b_option_quota_val560.sh`. If endpoint merging finishes later, rerun `bash scripts/merge_longqa_qwen35_27b_option_quota.sh` to build the full three-candidate comparison.

### Final code and output audit (2026-08-07)

- **Artifact integrity:** all 136 archived `predictions.jsonl` artifacts use valid dataset sample keys in annotation order, with no duplicate or foreign rows. Several old rejected TCoT/timeline runs contain invalid answer strings, but none is used by the current endpoint or option-quota candidates. Re-evaluating endpoint 27B after identity alignment preserves `130/140` exactly.
- **Correctness fixes:** LongQA evaluation and diagnostics now align every prediction by `(video_path, question)` even when file lengths happen to match. Uniform and proof-pack resume paths validate the full sample key and truncate an incompatible cached suffix before appending. Fixed proof packs validate stored sample keys, and inference refuses to continue if selected-frame decoding silently loses an image.
- **Primary metadata:** `configs/egolongqa_primary_pipeline.json` now points to the actual `582/700` August 5 label-free majority rather than the superseded July 31 rotation-pivot candidate.
- **Frozen temporal route:** use endpoint 27B by default and corrected option-quota 27B only when the two disagree on compiler operators `BEFORE` or `MULTI_TIME`. This gives **`132/140` (`94.29%`)** on dev, with two fixes and no regressions. The same rule improves the older complete 9B endpoint/option pair from `542/700` to `548/700`, providing independent support for the route.
- **Held-out checkpoint:** on the first 190 val560 rows completed by both running 27B jobs, endpoint scores `160/190`, option-quota `157/190`, and the frozen route scores **`163/190`**, with six fixes and three regressions. This is a provisional rolling result; the final conclusion waits for all 560 rows.
- **Input ablations prepared:** `slurm_longqa_qwen35_27b_endpoint_timestamps_verify_dev.sh` adds exact per-image timestamps and option-by-option verification to the endpoint pack. `slurm_longqa_qwen35_27b_endpoint_video_smoke5.sh` tests whether Qwen benefits when the same frames are transported as one chronological video rather than 64 independent images; run the prepared full dev script only if the smoke test is healthy.
- **Option-quota implementation audit:** the historical option-quota strategy selected one center per answer option and filled its remaining target-center budget from a combined query; `CENTERS_PER_OPTION=4` was parsed and fingerprinted but not applied in that strategy. Existing results retain that exact meaning. A separate `balanced_option_quota_pivot` implementation now honors multiple centers per option via round-robin allocation, with a unit check and the gated dev launcher `slurm_longqa_qwen35_27b_balanced_option_quota_dev.sh`.
- **Category routing rejected:** choosing the best complete candidate per dataset category on dev gives `127/140`, but those frozen choices fall to `442/560` on the held-out complement. This is worse than direct 27B (`451/560`) and confirms that coarse category routing is overfit.
- **Running held-out checkpoint:** at 207 rows shared by the endpoint and option-quota 27B jobs, endpoint is `175/207`, option-quota is `173/207`, and the frozen `BEFORE`/`MULTI_TIME` route is `178/207`. This remains provisional until both jobs finish.

### Qwen3.5-27B endpoint-inclusive full result (2026-08-07)

- **Held-out completion:** job `49531487` completed all `560/560` rows at `478/560` (`85.36%`) with no inference errors.
- **Full score:** strict key-based merging with the `130/140` dev partition gives **`608/700` (`86.86%`)**, improving the previous `582/700` primary by 26 answers.
- **Calibration check:** macro answer-position accuracy is `88.34%`; non-C accuracy is `87.50%`; per-letter accuracy is `A=87.50%`, `B=86.34%`, `C=86.49%`, and `D=93.02%`. The result is not dependent on the dominant C label.
- **Complementarity:** against the previous primary there are 96 disagreements: 56 endpoint-only correct, 30 primary-only correct, and a `638/700` two-model oracle. Endpoint is now the standalone default; any verifier must preserve it unless it can reliably recover part of those 30 losses.
- **Runtime/context:** held-out runtime was `14,454` seconds, approximately `25.81` seconds per question including startup. Mean/p95/max context fill was `56.87%/57.14%/57.47%`.
- **Artifact:** `runs/egolongqa/qwen35_27b_vllm_uniform64_endpoint_px451584_full_2026-08-07/predictions.jsonl`.

### Sampling-phase and endpoint-preserving retrieval suite prepared (2026-08-07)

- **Controlled phase ablations:** `slurm_longqa_qwen35_27b_uniform64_legacy_dev.sh` and `slurm_longqa_qwen35_27b_uniform64_midpoint_dev.sh` change only the 64-frame temporal grid. Model, resolution, prompt, reasoning state, context, and dev140 subset match the `130/140` endpoint run.
- **Three-phase analysis:** `scripts/merge_longqa_qwen35_27b_sampling_phases_dev.sh` uses endpoint as the deterministic all-different fallback and reports each candidate's accuracy, unique fixes, majority accuracy, agreement structure, and candidate oracle. It runs only after both new phase candidates finish.
- **Conservative retrieval:** `endpoint_mmr_hybrid` retains 48 endpoint anchors and adds exactly 16 token-safe SigLIP2 retrievals. Maximal marginal relevance penalizes visually redundant selections, and a 10-second temporal NMS is applied where possible. Launcher: `slurm_longqa_qwen35_27b_endpoint_mmr_hybrid_dev.sh`.
- **Validation:** midpoint index tests, exact 48+16 frame-budget tests, Python compilation, shell syntax checks, launcher dry runs, and candidate-pool smoke analysis pass. Existing endpoint behavior remains unchanged.

### Sampling-phase ablations and temporal route completed (2026-08-07)

- **Controlled phase ablations:** legacy-grid job `49542064` completed at `123/140` (`87.86%`), while midpoint-grid job `49542066` completed at `121/140` (`86.43%`). Both used the same Qwen3.5-27B checkpoint, 64-frame budget, resolution, prompt, and reasoning setting as endpoint 27B. The endpoint-inclusive grid remains substantially stronger at `130/140` (`92.86%`): seven answers above legacy and nine above midpoint.
- **Phase voting rejected:** the three-grid majority scores `127/140` (`90.71%`), below endpoint alone. The candidates agree on 117 rows; among the 23 disagreements, the majority is correct on 15. Their diagnostic oracle is `135/140` (`96.43%`), with four midpoint-only correct answers, but no label-free phase rule has yet isolated those cases safely. Do not expand legacy to val560, and do not replace endpoint with phase voting.
- **Option-quota held-out completion:** job `49532556` produced all 560 predictions at `468/560` (`83.57%`). Combined with dev140, corrected option-quota 27B scores `591/700` (`84.43%`). Its inference artifacts are valid; an obsolete copy of the wrapper encountered an unmatched quote only after evaluation, during archival. Diagnostics and archival were completed manually, and the current wrapper passes shell syntax validation, so no GPU rerun is required.
- **Complementarity:** endpoint and option-quota disagree on 92/700 rows. Endpoint alone is uniquely correct on 52 of these and option-quota on 35, giving a diagnostic oracle of `643/700` (`91.86%`). Option-quota is too weak to replace endpoint globally but remains a meaningful temporal specialist.
- **Frozen temporal route:** endpoint remains the default; option-quota is used only when the two predictions disagree and compiler v2 identifies `BEFORE` or `MULTI_TIME`. This rule was frozen before val560 completion. It scores `132/140` on dev and `481/560` on the held-out complement, for **`613/700` (`87.57%`)** overall. On held-out data it adds three answers over endpoint, confirming a modest independent gain rather than a dev-only fit.
- **Route behavior:** the full route changes 19 endpoint predictions, producing 12 fixes and seven regressions. A three-way majority with the previous primary also reaches `613/700`, but the two-candidate temporal route is the cleaner primary because both inputs are direct visual answerers and no verifier-derived candidate, answer labels, or fitted calibrator is used.
- **Decision:** promote `runs/egolongqa/qwen35_27b_endpoint_optionquota_temporal_route_full_2026-08-07/` as the current primary accuracy artifact. Keep endpoint-only `608/700` as the simpler and faster submission candidate when retrieval or multi-pass latency is a concern.

### Final direct-candidate ensemble and retrieval audit (2026-08-07)

- **Candidate hygiene:** the final vote audit includes only seven direct visual answerers: endpoint 27B, option-quota 27B, and five completed 9B evidence variants. Bayesian calibration and the historical candidate-blind judge outputs are excluded because they are fitted or derived decisions, not independent candidate models.
- **Available oracle:** the seven direct candidates contain a correct answer on `662/700` (`94.57%`) rows. Endpoint plus option-quota 27B alone reaches a `643/700` oracle. There is real ensemble headroom, but it is concentrated in disagreements.
- **Classical hard voting rejected:** unweighted plurality across the seven candidates scores `573/700` (`81.86%`). Weighting votes by smoothed dev140 log-odds reaches only `582/700` (`83.14%`). Correlated retrieval failures let several weaker variants outvote endpoint, so vote count and dev accuracy are not adequate arbitration statistics.
- **Agreement as a gate:** endpoint is correct on `426/451` (`94.46%`) rows when all seven candidates support its answer, but only `15/31` (`48.39%`) when it has one vote. This supports restricting future visual verification to low-agreement rows, not switching directly to the plurality answer.
- **Reproducible audit:** `scripts/analyze_longqa_direct_vote_fusion.py` writes split-wise accuracy, pairwise complementarity, pool oracle, hard-vote baselines, endpoint-support buckets, and subset-majority diagnostics. Current output: `analysis/longqa_direct_vote_fusion_2026-08-07.json`.
- **Classical frame rank fusion:** `rank_fusion_balanced` applies reciprocal-rank fusion separately to the target and four option retrieval rankings, then uses MMR and temporal NMS. The corresponding 27B dev job preserves 48 endpoint anchors and adds 16 RRF-ranked frames: `slurm_longqa_qwen35_27b_endpoint_rrf_hybrid_dev.sh`.
- **Additional direct controls:** `slurm_longqa_qwen35_27b_uniform48_endpoint_dev.sh` tests whether fewer global frames reduce distractor density. `slurm_longqa_qwen35_27b_uniform64_endpoint_thinking_dev.sh` creates a direct reasoning-enabled endpoint candidate rather than reusing the derived candidate-blind judge.
- **Validation:** Python compilation, shell syntax, launcher dry runs, and direct RRF assertions pass. The conda environment does not include `pytest`, so the focused RRF checks were executed directly rather than through the test runner.
- **Run plan:** endpoint-48, endpoint reasoning, timestamped verification, balanced option-quota, MMR hybrid, and RRF-MMR hybrid are independent dev140 gates. Promote only a result above `130/140` or a specialist with at least three endpoint fixes and at most one regression under a rule fixed before val560 completion.

### Final-day six-job launch checkpoint (2026-08-07)

- **Healthy jobs:** endpoint-48 `49544971`, timestamped verification `49544973`, balanced option-quota `49544974`, MMR hybrid `49544975`, and RRF-MMR hybrid `49544976` all started normally. The option-quota and MMR selection stages completed from cached SigLIP2 features before their Qwen answer passes; RRF selection was progressing normally. Hugging Face `404` metadata probes in the retrieval stderr files are benign because the required processor and model files resolve successfully afterward.
- **Early failure:** direct endpoint reasoning job `49544972` stopped before model startup. The launcher passed `--longqa-max-new-tokens` and `--require-final-answer-marker`, but the unified `run_evaluation.py` CLI had not exposed or forwarded those existing LongQA generator controls.
- **Fix:** the unified CLI now accepts both controls, forwards them into the LongQA generation namespace, and forwards them through its optional SLURM submission path. The corrected launcher passes parser, Python compilation, and shell validation. Job `49544972` produced no predictions and should be resubmitted with the same launcher.
- **Second reasoning failure:** resubmitted job `49545055` reached Qwen inference but stopped on the first sample because its answer-only retry reopened reasoning and again omitted the final-answer marker. The launcher exported `THINKING_TOKEN_BUDGET`, while the direct model backend reads `VLLM_THINKING_TOKEN_BUDGET`; consequently the retry could not force a zero-token reasoning budget. The launcher now exports the backend variable, allowing the existing `generate_final_answer_batch` path to disable thinking during the 64-token continuation. The failed job wrote no predictions and can be resubmitted from the beginning.
- **Third reasoning failure and final retry fix:** job `49545884` confirmed that a zero reasoning-token budget alone does not override Qwen3.5's `enable_thinking=true` chat-template setting. It again stopped safely on the first row with zero predictions. The VLLM answer-only retry now explicitly sends both `thinking_token_budget=0` and `enable_thinking=false`; normal generation retains the configured thinking mode. A focused mock assertion verifies both retry controls before another submission.

### Final-day evidence-selection results (2026-08-08)

- **Completed jobs:** endpoint-48 `49544971`, timestamped verification `49544973`, balanced option-quota `49544974`, MMR hybrid `49544975`, and RRF-MMR hybrid `49544976` completed all `140/140` dev rows. Their scores are respectively `124/140` (`88.57%`), `116/140` (`82.86%`), `121/140` (`86.43%`), `125/140` (`89.29%`), and `123/140` (`87.86%`). The unchanged endpoint-64 reference is `130/140` (`92.86%`), and the frozen temporal route is `132/140` (`94.29%`).
- **Endpoint recovery:** endpoint-48, balanced option-quota, MMR, and RRF-MMR recover `3`, `3`, `5`, and `5` endpoint errors, but introduce `9`, `12`, `10`, and `12` regressions. Timestamped verification recovers zero endpoint errors and introduces 14 regressions. None passes the predeclared promotion gate of either exceeding `130/140` or making at least three fixes with at most one regression.
- **Route recovery:** relative to the stronger frozen temporal route, MMR and RRF-MMR each repair three errors but introduce ten and twelve regressions. Endpoint-48 and balanced option-quota each repair one route error. Timestamped verification repairs none. No endpoint-preserving consensus rule over these candidates improves the route.
- **Voting and oracle:** plurality across endpoint and the five completed candidates scores `129/140`, below endpoint alone. Their diagnostic oracle is `135/140`, so five additional answers exist in the pool but cannot be isolated by ordinary voting. MMR is the most complementary new branch, especially on `AFTER` and `MULTI_TIME`, but its global-question changes are all harmful and its routing precision is not adequate for held-out promotion.
- **Rank-fusion result:** MMR and RRF-MMR agree on `138/140` answers. Their 64-frame proof packs overlap by `62.85` frames on average; the 16 retrieved supplements overlap by `14.85` frames. Reciprocal-rank fusion therefore did not create meaningfully different evidence, and its two answer changes relative to MMR were both detrimental.
- **Runtime/context:** answer-stage runtimes were `6055` seconds for endpoint-48, `8073` for timestamps, `7735` for balanced option-quota, `7993` for MMR, and `8062` for RRF-MMR. Mean context fill was `42.74%` for endpoint-48, `58.59%` for timestamps, and `56.88%` for all three 64-frame retrieval variants.
- **Decision:** do not run val560 expansions for these five candidates. Retain the `613/700` frozen temporal route as the primary accuracy artifact and endpoint-only `608/700` as the simpler submission. The direct-thinking candidate remains unscored after three safe first-row failures; its final retry path is fixed, but it requires a fresh submission if still worth the compute.
- **Analysis artifact:** `analysis/longqa_final_day_evidence_candidates_dev140_2026-08-08.json` records the six-candidate accuracies, agreement, majority, and oracle without using labels to construct predictions.

### Pre-test deep-push audit and final launchers (2026-08-08)

- **Residual-error split:** the `613/700` route has 87 errors. Existing direct candidates contain the correct answer for 49; all seven direct candidates miss the remaining 38, including 25 unanimous wrong answers. Arbitration and missing/incorrect evidence are therefore both material.
- **Hard-vote conclusion:** endpoint-preserving consensus thresholds and candidate-subset votes were exhaustively checked over the completed direct pool. None improves the frozen route. RRF is not defined for one-letter outputs because they provide no option ranking.
- **Boundary hypothesis:** the BlackSwan pre/post-state mechanism is only an analogy, not a matching task. The local controlled evidence is stronger: exact endpoint coverage beats both endpoint-omitting grids. `endpoint_guarded_midpoint` now retains the exact first and last frames and fills the other 62 positions from midpoint bins. Existing sampling modes are unchanged.
- **Answer-rank fusion:** the new evidence-rank scorer runs only on endpoint/option-quota disagreements, obtains A-D log-probability rankings under both evidence packs, and evaluates consensus, probability fusion, candidate-restricted fusion, cross-view confirmation, and RRF with endpoint fallback. Dev140 requires 34 Qwen calls over 17 disagreements.
- **Launchers:** run `slurm_longqa_qwen35_27b_endpoint_guarded_midpoint_dev.sh` and `slurm_longqa_qwen35_27b_endpoint_guarded_midpoint_val560.sh` concurrently if the deadline window requires a full candidate. Run `slurm_longqa_qwen35_27b_evidence_rank_fusion_dev.sh`; gate its val560 counterpart on at least `132/140` or a no-regression correction pattern.
- **Validation:** Python compilation, shell syntax, parser checks, exact 64-frame boundary assertions, proof-pack coverage checks (`140 + 560`), fusion-policy assertions, and dry runs pass. Full rationale and commands: `documentation/LONGQA_PRETEST_DEEP_PUSH_2026-08-08.md`.

### Pre-test launch checkpoint (2026-08-08, 09:27 CEST)

- **Guarded-midpoint jobs:** dev140 job `49555295` and val560 job `49555298` both started successfully and continue writing valid predictions. At the latest checkpoint they contain 30 and 25 rows respectively; neither run is complete.
- **Rolling comparison:** guarded midpoint is `26/30` on the ordered dev prefix, versus `29/30` for endpoint and the frozen temporal route. On the first 25 val-complement rows it is `22/25`, equal to endpoint and the route, with one fix and one regression. These ordered-prefix figures are monitoring signals only and must not be reported as final accuracies.
- **Rank-fusion failure and repair:** dev job `49555300` exited before model startup because proof-pack records carry an explicit `sample_key` but no repeated `question` field. The rank-fusion index reconstructed an incomplete key instead of honoring that field. `run_score_longqa_evidence_rank_fusion.py` now uses the stored key first; all 140 proof-pack rows align after the fix, and the failed job produced no predictions.
- **Deadline-safe sharding:** contiguous 140-row quarters of val560 and a three-task tail array are prepared. This can complement the already-running unsharded job without overlapping the first quarter; the merge utility validates exact ordered coverage and can ignore extra rows from the live first-quarter source. Do not launch the extra GPUs solely from the current weak dev prefix; first allow more dev evidence or use it only when completing a full guarded-midpoint artifact before the submission deadline is itself the priority.

### Evidence-rank fusion dev result (2026-08-08)

- **Completion:** repaired job `49557610` completed all 140 rows and archived cleanly with an empty stderr. Endpoint and option-quota disagree on 14 dev rows, so the run required 28 multimodal scoring calls and finished in 995 seconds. Mean context fill was `56.93%`.
- **Best policies:** mean log probability and candidate-restricted mean log probability both score **`132/140` (`94.29%`)**, versus endpoint at `130/140`. Mean probability scores `131/140`; RRF, view consensus, and cross-view confirmation remain at `130/140`.
- **Change audit:** mean-log-probability fusion changes five endpoint answers: three fixes, one regression, and one wrong-to-wrong change. Relative to the frozen `132/140` temporal route it makes three changes: one fix, one regression, and one wrong-to-wrong change. It therefore ties rather than exceeds the route, but provides a genuinely different, label-free arbitration rule that merits the held-out val560 gate.
- **Next gate:** val560 contains 78 endpoint/option-quota disagreements, requiring 156 scoring calls. Extrapolating from dev suggests roughly 70--90 minutes after scheduling and model startup. Promote only the policy named before inspecting held-out labels: `mean_logprob`.
- **Guarded-midpoint rolling checkpoint:** at 60 dev rows it scores `55/60`, versus `57/60` for endpoint and the route, with two fixes and four regressions. At 51 held-out rows it scores `42/51`, tied with endpoint and one behind the route, with two fixes and two regressions. Both guarded-midpoint jobs remain incomplete.

### Evidence-rank fusion held-out result and guarded-midpoint gate (2026-08-08)

- **Held-out completion:** evidence-rank-fusion job `49558139` completed all `560/560` rows cleanly in 8,870 seconds. It scored 78 endpoint/option-quota disagreements with 156 multimodal calls; mean context fill was `56.85%`.
- **Frozen-policy result:** the policy fixed before held-out evaluation, `mean_logprob`, scores `483/560` on val560. Together with its `132/140` dev result, this gives **`615/700` (`87.86%`)**, two answers above the frozen temporal route and seven above endpoint alone. Across all 700 rows it changes 46 endpoint answers, producing 25 fixes, 18 regressions, and three wrong-to-wrong changes.
- **Exploratory policies:** `mean_probability` scores `131/140 + 487/560 = 618/700` (`88.29%`), and candidate-restricted mean log probability scores `132/140 + 484/560 = 616/700` (`88.00%`). These are valid label-free inference policies but were not the held-out policy named in advance; their full-validation scores should be treated as exploratory rather than used to claim an unbiased selection result. RRF falls to `605/700`.
- **Guarded-midpoint dev rejection:** job `49555295` completed at `125/140` (`89.29%`), below endpoint `130/140` and the route `132/140`. It does not pass the promotion gate.
- **Guarded-midpoint val status:** job `49555298` remains active at 222/560 rows. On this ordered prefix it scores `181/222`, versus endpoint `189/222` and the route `192/222`. Given the completed dev rejection and negative held-out prefix, completing this candidate is not justified for accuracy selection.
- **Submission exports:** validated 700-row leaderboard files are available for the predeclared `mean_logprob` result at `submissions/egolongqa/qwen35_27b_evidence_rank_mean_logprob_2026-08-08/` and the exploratory `mean_probability` result at `submissions/egolongqa/qwen35_27b_evidence_rank_mean_probability_2026-08-08/`. Both contain exactly one valid answer per unique video in annotation order and require no fallback repairs.

### Test-preparation split and error audit (2026-08-08)

- **Dev140 diagnosis:** the old category/letter/temporal-stratified dev set contains 130 endpoint-correct rows and only 14 endpoint/option disagreements, making it materially easier than the full validation distribution. It remains preserved for historical comparisons but should no longer be the sole promotion gate.
- **Balanced evaluation folds:** five new disjoint 140-row folds cover all 700 examples exactly once and balance category, compiler-v2 temporal operator, answer position, duration quartile, frozen endpoint correctness, and endpoint/option agreement. Endpoint correct counts are `122, 121, 122, 122, 121`; disagreement counts are `18, 19, 18, 18, 19`.
- **Cross-fold results:** temporal route scores `121,122,124,123,123`; mean log probability scores `123,123,123,125,121`; mean probability scores `124,121,124,127,122`. The probability rule beats the log-probability rule on four folds but does not beat the temporal route on every fold.
- **Fusion error split:** the `618/700` candidate has 82 errors. Thirty-five are incorrect endpoint/option consensus rows that bypass scoring, 18 are fusion regressions from a correct endpoint answer, 19 have a correct direct specialist that both rank views miss, seven have the correct option ranked first by one view but suppressed by averaging, and three are missed by all direct candidates and both scored views.
- **Available headroom:** a seven-direct-candidate oracle remains `662/700` (`94.57%`); 44 current fusion errors have a correct direct candidate and 38 do not. The uncertainty branch is correct on 25 fusion errors, supporting a third uncertainty evidence view. Consensus-hard misses are dominated by repeated instances, compound options, first/last transitions, OCR, exact prices/counts, and object-state tracking.
- **Artifacts:** detailed rationale and test plan are in `documentation/LONGQA_TEST_PREPARATION_AUDIT_2026-08-08.md`; machine-readable fold and error artifacts are under `analysis/longqa_*20260808*` and `configs/egolongqa_balanced_fold*_of5_20260808.json`.

### Balanced-fold temporal suppression results (2026-08-08)

- **Evaluation reference:** all three experiments use balanced fold 0, where the current endpoint/option-quota mean-probability fusion scores `124/140` (`88.57%`). This replaces the unusually easy historical dev140 as the immediate screening set.
- **Uncertainty third view:** repaired job `49562658` scored the uncertainty-selected frame pack on only the 18 endpoint/option disagreements. Three-view mean probability scores `122/140`, median probability and RRF score `120/140`, entropy-weighted probability scores `123/140`, and conservative uncertainty tie-breaking scores `121/140`. No policy improves the cached two-view fusion; the uncertainty evidence is not a reliable equal-weight vote.
- **Atomic hypothesis verifier:** job `49562442` compared only three direct candidates: endpoint 27B, option-quota 27B, and uncertainty-pivot 9B. It planned and retrieved evidence for 41 disagreements, requested four refinements, and scores `122/140`. Its eight overrides contain three fixes and five regressions. The regressions concentrate on repeated occurrences and first/last questions in Shopping and Sightseeing, showing that visible support for one occurrence was still mistaken for proof of the requested occurrence.
- **Wrong-time contrast:** repaired job `49562659` applied to 53 directional questions and scores `115/140` with requested-side frames alone. Subtracting opposite-side option support degrades monotonically: lambda `0.25`, `0.5`, and `1.0` score `110`, `104`, and `94` correct. The unchanged fusion already scores `49/53` on these directional rows; requested-side sampling fixes one error but causes ten regressions. Opposite-side visibility is therefore not valid negative evidence, especially when objects or actions recur.
- **Decision:** do not expand any of these three methods to all 700 rows and do not add them to the primary ensemble. Retain `618/700` mean-probability fusion as the current submission artifact. The next useful gates are the question-only targeted object ledger and hierarchical occurrence search; defer the 560-call option-permutation experiment until either produces a genuinely complementary candidate.

### Balanced-fold object ledger and hierarchical selection (2026-08-08)

- **Targeted object ledger:** job `49564239` completed all 140 balanced-fold rows in `12,182` seconds and scores `113/140` (`80.71%`), versus `124/140` for the current probability fusion. Relative to that fusion it makes two fixes, thirteen regressions, and three wrong-to-wrong changes; their diagnostic oracle is only `126/140`.
- **Useful ledger recoveries:** the two fixes are a Shopping `MULTI_TIME` first/last-product question and a global outdoor-sign question. This confirms that explicit object occurrences can occasionally recover evidence absent from the fusion, but the current branch is far too imprecise for routing.
- **Ledger implementation diagnosis:** question-only Grounding DINO prompts avoid direct option leakage, but detections remain overly dense: approximately `184.7` detections and `45.2` hit frames per question across 48 inspected frames. The current ledger then retains only the first 16 non-empty chronological lines rather than consolidating each concept into first, strongest, and last distinct occurrences. This biases the prompt toward early repeated detections and explains many identity, price, and final-state regressions.
- **Hierarchical selection:** job `49564240` completed the selection-only stage for all 140 rows in `23,160` seconds with empty stderr. Every record contains exactly 64 unique frames: 48 locally selected frames plus 16 global anchors, with three pivot centers and eight target centers. No accuracy exists yet because the 27B answer stage is deliberately separate.
- **Latency implication:** hierarchical selection averages about `165.4` seconds per question before the 27B answer call. It remains under the 300-second test limit in isolation, but leaves limited room for additional answerers or verification passes and would add a separate 9B checkpoint to the submitted model.
- **Hierarchical answer result:** job `49571948` consumed the cached selections and completed all 140 rows in `7,469` seconds. The 27B answer stage averages `53.4` seconds per question with `56.88%` mean context fill; combined with selection, the full method takes approximately `218.8` seconds per question.
- **Accuracy and change audit:** hierarchical selection scores `117/140` (`83.57%`). Relative to balanced-fold mean-probability fusion (`124/140`), it changes 17 answers: three fixes, ten regressions, and four wrong-to-wrong changes, for a two-method oracle of `127/140`. Its useful fixes cover a gate/fence state change, a first/last wall-patching sequence, and an outdoor-sign inference, but it loses several OCR, repeated-instance, first/last, and cross-event questions.
- **Decision:** reject the hierarchical branch for expansion. Its three complementary answers do not justify ten regressions, a separate 9B selector checkpoint, or approximately `219` seconds per question. Do not run fold 1 or val560. Retain the cached outputs only for error analysis; the current `618/700` mean-probability fusion remains the primary artifact. The object ledger likewise remains rejected until redesigned around temporally deduplicated first/peak/last occurrences and explicit OCR handling.

### Targeted rescue experiment setup (2026-08-09)

- **Reference and split:** all new arms use balanced fold 0 and preserve the `124/140` mean-probability-fusion answer on non-target rows. No full-set expansion is scheduled automatically.
- **Prompt ablation:** `slurm_longqa_qwen35_27b_prompt_ablation_fold0_array.sh` holds Qwen3.5-27B, endpoint-inclusive 64-frame evidence, resolution, and decoding fixed while comparing evidence-first, option-verification, visible-support, and clause-completeness instructions. The cached endpoint run is the unchanged baseline.
- **OCR ablation:** `slurm_longqa_qwen35_27b_ocr_ablation_fold0_array.sh` targets 38 text-sensitive fold-0 questions. Both arms use identical question-conditioned detail crops; task 0 answers from the crops directly, while task 1 first creates a transcription and then answers. This isolates the value of textual transcription from the value of enlargement.
- **Occurrence strips:** `slurm_longqa_qwen35_27b_occurrence_strips_fold0.sh` targets 50 repeated-occurrence, paired-endpoint, or state-change questions. Question-only detections are linked by appearance, adjacent observations within eight seconds are merged, and first/strongest/last distinct occurrences receive one-second before/center/after context.
- **Density ablation:** `slurm_longqa_qwen35_27b_density_ablation_fold0_array.sh` compares raw top-48 pivot evidence against temporal and perceptual deduplication under the same prompt and answerer. It directly tests whether repeated nearby frames bias Qwen toward a visually frequent distractor.
- **Support ablation:** `slurm_longqa_qwen35_27b_clause_support_fold0_array.sh` targets 98 compound or candidate-disagreement rows and makes approximately 336 scored calls per arm. Whole-option and explicit-clause prompts see identical endpoint-inclusive evidence and each emit unrestricted, conservative, and strict fallback policies.
- **Conservative router:** `slurm_longqa_targeted_rescue_router_fold0.sh` is CPU-only and must wait for both OCR arms, occurrence strips, and clause-support task 1. The two OCR arms form one evidence family rather than two votes; an alternative replaces fusion only when confirmed by clause support or by the independent occurrence family.
- **Validation:** modified/new Python files compile, all shell launchers pass `bash -n`, real prerequisite paths resolve, dry-run commands construct successfully, exact endpoint inclusion and occurrence grouping assertions pass, and dry-run output directories were removed. Detailed gates and run order are in `documentation/LONGQA_TARGETED_RESCUE_EXPERIMENTS_2026-08-09.md`.

### Targeted rescue results (2026-08-10)

- **Completion:** prompt array `49595487`, OCR array `49595488`, density array `49595489`, occurrence job `49595490`, and support array `49595491` completed every balanced-fold row. The first CPU router submission `49598746` failed before reading data because its standalone script did not add the starter-kit directory to `sys.path`; the import path was fixed and the router was then executed successfully from the same completed artifacts.
- **Prompt ablation:** evidence-first, option-verification, visible-support, and clause-completeness prompts score respectively `103/140`, `119/140`, `118/140`, and `100/140`. The unchanged endpoint prompt scores `122/140` on this fold, while fusion scores `124/140`. Relative to endpoint, option verification makes one fix and four regressions; visible support makes two fixes and six regressions. More elaborate instructions therefore reduce accuracy despite identical frames and decoding. Runtimes are `7,692--7,830` seconds per arm with approximately `57%` mean context fill.
- **OCR ablation:** crops alone score `113/140`; adding an intermediate transcription raises this to `118/140`, showing that transcription is useful relative to the same crop pack but not relative to fusion. The transcription arm changes ten fusion answers with one fix, seven regressions, and two wrong-to-wrong changes. The 38-question crop and transcription runs take `2,319` and `2,778` seconds respectively.
- **Occurrence strips:** the 50-question appearance-linked first/strongest/last occurrence branch scores `113/140`. It changes fifteen fusion answers with one fix, twelve regressions, and two wrong-to-wrong changes. The full job takes `4,642` seconds including SigLIP2 track preparation; merging adjacent detections does not make question-only Grounding DINO identity links reliable enough for answering.
- **Density ablation:** raw top-48 pivot evidence scores `116/140`, temporal deduplication scores `115/140`, and perceptual deduplication scores `117/140`. Temporal suppression retains only `37.87` frames on average and hurts accuracy. Perceptual suppression finds only `0.59` near duplicates per question and still retains 48 frames by pulling lower-ranked evidence, yielding only one answer over raw-48 and remaining seven below fusion. The direct distractor-density hypothesis is therefore not supported by these selectors.
- **Support scoring:** whole-option unrestricted, conservative, and strict policies score `114`, `122`, and `123` correct. Explicit-clause variants score `117`, `119`, and `122`. Strict whole-option scoring is the closest result but its five overrides contain one fix, two regressions, and two wrong-to-wrong changes. The one genuine correction is a kitchen-tool/use question; similarly confident support margins also cause OCR and compound-temporal regressions, so a stronger fixed threshold does not isolate it. Each arm makes 336 multimodal calls and takes about `6,780` seconds.
- **Router result:** OCR-plus-clause and any-family-plus-clause policies remain at `124/140`, each making one wrong-to-wrong change. Occurrence-plus-clause makes no changes. Requiring OCR and occurrence families to agree causes one regression and scores `123/140`. The two OCR arms are still treated as one family; no synthetic candidate is introduced.
- **Decision:** none passes the predeclared correction/regression gate. Do not run fold 1 or full-set expansions for prompt, OCR, occurrence, density, support, or rescue-routing variants. Retain the `618/700` mean-probability fusion artifact as primary. The router import and array-log archival paths are fixed for reproducibility; cosmetic `cp` messages in the completed array stderr files did not affect predictions or scores.

### Confidence and segment-fusion results (2026-08-10)

- **Completion:** margin export job `49601012`, consensus-challenge job `49601013`, and segment late-fusion job `49601014` completed cleanly with empty stderr. The two GPU jobs each targeted the same 29 balanced-fold rows: 18 endpoint/option disagreements plus 11 endpoint/option consensus rows challenged by at least three of four independent 9B branches.
- **Margin-weighted full export:** weighting each evidence view by its own top-versus-second option-probability margin scores **`619/700` (`88.43%`)**. It differs from equal mean probability on exactly one row, correcting the Brea Boulevard cross-street question from `A` to gold `C`. This is a fixed label-invariant rule, but its selection followed inspection of validation results and must be reported as exploratory rather than as an unbiased held-out gain.
- **Consensus challenge:** rescoring both complete 64-frame views does not change any answer relative to the `618/700` primary, including all 11 challenged consensus rows. Mean log probability, mean probability, and margin-weighted probability therefore remain at `124/140` on fold 0. The independent 9B disagreement signal identifies difficult rows, but another full-context score preserves the same distractor bias.
- **Segment late fusion:** splitting each 64-frame view into four chronological 16-frame blocks does not work as a global replacement. Mean probability, per-view top-two evidence, adjacent-block evidence, block RRF, and stability-guarded adjacent evidence score respectively `120`, `121`, `122`, `119`, `121`, and `121` out of 140, versus primary `124/140`.
- **Segment correction pattern:** adjacent-block fusion makes three fixes, five regressions, and one wrong-to-wrong change. All three fixes occur among challenged 27B consensus rows rather than ordinary endpoint/option disagreements. They cover a gate/fence modification, an unused paint color, and a playground slide after a temporal anchor.
- **Exploratory unanimity gate:** requiring all four 9B branches to agree on the same alternative and requiring adjacent-block fusion to select that alternative changes three fold-0 answers. It produces two fixes, zero regressions, and one wrong-to-wrong change, raising the fold from `124/140` to **`126/140` (`90.00%`)**. This rule was discovered after inspecting fold 0 and needs confirmation unchanged on fold 1 before any expansion.
- **Runtime/context:** full-view consensus scoring makes 58 calls in 3,276 seconds with about `57%` context fill. Segment scoring makes 232 shorter calls in 3,263 seconds with about `14.5%` context fill. Both average roughly 113 seconds per targeted row, but a deployable unanimity gate would also need the four 9B branches, so its complete per-video latency must be audited against the 300-second test limit.
- **Decision:** retain `618/700` as the conservative primary and `619/700` as an exploratory confidence-weighted export. Reject unrestricted segment aggregation. The only justified next gate is the frozen four-branch-unanimity plus adjacent-segment rule on balanced fold 1; do not fit a threshold or category router on fold 0.
- **Fold-1 confirmation prepared:** `slurm_longqa_qwen35_27b_unanimous_segment_gate_fold1.sh` freezes the rule above, excludes ordinary endpoint/option disagreements, and scores only six unanimous consensus challenges on balanced fold 1. It requires 48 short-context scoring calls. The evaluator reproduces `126/140` on the cached fold-0 features before launch.

### Unanimous segment gate fold-1 rejection (2026-08-10)

- **Completion:** job `49610099` completed cleanly with empty stderr. It scored the six precomputed unanimous consensus challenges using 48 short-context calls in 1,079 seconds; mean context fill was `14.49%`.
- **Frozen result:** the existing mean-probability primary scores `121/140` on balanced fold 1. The predeclared four-branch-unanimity plus adjacent-segment rule scores **`119/140` (`85.00%`)**, making two changes, zero fixes, and two regressions. It therefore fails the no-regression confirmation gate.
- **Row audit:** all six challenged endpoint/option consensus answers were already correct. Adjacent-block scoring correctly retained four of them but changed two correct answers: the Nature Store ornament-location question from `B` to `D`, and the road-surface comparison question from `B` to `C`.
- **Interpretation:** unanimity among the four 9B branches is not reliable evidence that a 27B consensus is wrong. The branches share enough model and evidence-selection bias to agree on the same distractor. Segment scoring can reinforce that correlated error rather than independently verify it.
- **Decision:** reject the unanimity gate and do not run it on folds 2--4 or export it over all 700 rows. Retain `618/700` as the conservative primary and `619/700` only as the explicitly exploratory margin-weighted result. Do not tune the unanimity threshold or segment policy using fold-1 labels.
# Open video-model evaluation suite prepared (2026-08-10; not yet run)

- Added consistent five-sample smoke jobs for MiniCPM-V 4.5 (192 frames),
  Molmo2-8B (256), InternVL3.5-30B-A3B-Flash (64), Cosmos-Reason2-8B (128),
  and NVILA-8B-HD-Video with AutoGaze (128 tile frames plus 64 thumbnails).
- All runs preserve the baseline MCQ prompt, use endpoint-inclusive chronological
  sampling, record per-sample generation time, and fail promotion when a call
  exceeds the workshop's 300-second limit.
- Added gated balanced-fold-2 launchers. No accuracy is recorded yet; these are
  prepared experiments, not completed results.
- Run order and exact commands: `documentation/LONGQA_OPEN_VLM_RUNBOOK_2026-08-10.md`.

### Initial smoke-launch failures and fixes (2026-08-10)

- Jobs `49612893` (MiniCPM-V 4.5), `49612894` (Molmo2-8B), `49612895`
  (InternVL3.5 Flash), and `49612896` (Cosmos-Reason2-8B) exited before model
  startup. The common launcher incorrectly passed `--input` to
  `run_evaluation.py`; it now passes the supported `--golden` argument. These
  jobs produced no predictions and are not benchmark results.
- NVILA setup job `49612898` created the empty Python environment but failed
  while building `flash-attn`: no CUDA toolkit or `CUDA_HOME` was available.
  Smoke job `49612919` consequently failed on `import torch`. The setup now
  installs CUDA Toolkit 12.8, pins PyTorch 2.10/torchvision 0.25, builds
  `flash-attn` against that stack, and writes `.autogaze_ready` only after all
  imports and CUDA availability pass. The NVILA runner refuses incomplete
  environments.
- The corrected common vLLM command and all modified Python/shell files pass
  local parser and syntax checks. All five smoke tests require a fresh run.

### Second smoke-launch failures and fixes (2026-08-10)

- Jobs `49612953`--`49612956` reached the vLLM startup path but exited before
  loading MiniCPM, Molmo2, InternVL, or Cosmos. The isolated vLLM subprocess
  could not import `psutil`, which is installed in the cluster user-site rather
  than the conda environment. The common launcher now adds that exact user-site
  directory to `PYTHONPATH` and verifies both `psutil` and `vllm` before model
  startup. No predictions were produced.
- NVILA setup job `49613419` successfully installed CUDA Toolkit 12.8, then a
  CUDA conda activation hook failed under strict shell mode because
  `NVCC_PREPEND_FLAGS` was unset. The setup now initializes the CUDA compiler
  environment variables before activation; this has been validated against the
  partially installed environment and its CUDA 12.8 `nvcc`. Smoke job
  `49613451` correctly refused to use the incomplete environment.
- These remain infrastructure failures rather than accuracy or latency results.
  The four vLLM smoke tests and NVILA setup each require another fresh run.

### Third open-video-model smoke status (2026-08-10)

- Jobs `49613478`--`49613481` passed the shared launcher checks, but none
  produced predictions. The remaining failures are now model-specific rather
  than defects in the common evaluation command.
- **MiniCPM-V 4.5 (`49613478`):** vLLM fails while applying its multimodal
  processor because the cached tokenizer wrapper lacks `im_start_id`. Repeating
  this vLLM job unchanged will not help; MiniCPM requires its native
  Transformers `model.chat` video path or a compatible vLLM/Transformers stack.
- **Molmo2-8B (`49613479`):** the requested context length was `49,152`, above
  the checkpoint's declared limit of `36,864`. Both the smoke and fold-2
  launchers now use a conservative `32,768`; this is the only vLLM smoke job
  ready for an immediate rerun.
- **InternVL3.5-30B-A3B-Flash (`49613480`):** all checkpoint shards downloaded,
  but vLLM's loader rejects a checkpoint `gating` weight. This is a checkpoint
  compatibility failure, not GPU memory exhaustion. A native Transformers
  video adapter is required before another run.
- **Cosmos-Reason2-8B (`49613481`):** Hugging Face returned `401` for the gated
  repository. The job must wait until access is approved and the cluster cache
  is authenticated; there is currently no Hugging Face token in either checked
  cache location.
- **NVILA setup (`49613482`):** dependency installation reached the
  `flash-attn==2.8.3.post1` build, but the `.autogaze_ready` marker has not yet
  been written and the captured log contains no final success or failure. Do
  not launch the NVILA smoke test until that marker exists.
- **Molmo corrected rerun (`49613789`):** lowering the context limit passed the
  original validation check, but vLLM then rejected the 256-frame visual item:
  its estimated `31,872` multimodal tokens exceeded the default `8,192` batched
  prefill budget. The Molmo launchers now set both the context and batched-token
  budgets to `32,768`. This run also produced no predictions and requires one
  more smoke submission.
- **Molmo successful smoke (`49613918`):** the final corrected job completed all
  five samples and scores `4/5` (`80.00%`). Mean generation time is `101.85`
  seconds and the maximum is `121.59` seconds, so every measured call passes the
  workshop's 300-second gate. Mean context fill is `80.16%` of the `32,768`
  token window. This is only a five-row compatibility/latency result; four of
  the five gold labels are `C`, so it is not an accuracy estimate suitable for
  model selection. Molmo is now eligible for the balanced-fold evaluation.
- **NVILA storage/status check:** the environment currently occupies about
  `13 GB`, the AutoGaze source tree about `1.2 MB`, and the setup writes its
  dependency/build cache under scratch rather than the repository. Scratch had
  approximately `14 TB` free at inspection time. Setup `49613482` remains
  incomplete at the silent `flash-attn` build with no readiness marker; its
  four-hour Slurm limit bounds the unattended run.
- **NVILA setup completion (`49613482`):** the previously silent
  `flash-attn==2.8.3.post1` build completed after approximately 95 minutes.
  Final checks report PyTorch `2.10.0+cu128`, CUDA `12.8`, Transformers
  `4.57.6`, AutoGaze import success, and `torch.cuda.is_available() == True`.
  The `.autogaze_ready` marker was written at `22:56`; the five-row NVILA smoke
  test is now eligible to run.
- **Molmo fold-2 scheduler hold (`49614243`):** Slurm held and requeued the job
  with `user env retrieval failed` before executing the batch script. It
  produced no log or prediction artifacts and was canceled by the user. This is
  a scheduler launch failure rather than a Molmo/runtime failure; resubmit the
  unchanged fold-2 command as a fresh job.
- **Repeated Molmo scheduler hold (`49621732`):** a second submission using
  command-line `--export=ALL,OPEN_VLM_KEY=molmo` received the same pre-launch
  `user env retrieval failed requeued held` state and was canceled. A dedicated
  Molmo fold-2 launcher now embeds the fixed model configuration and declares
  `#SBATCH --export=ALL`, so it can be submitted without dynamic environment
  export.
- **NVILA smoke failure (`49621731`):** the job entered the native Transformers
  path but failed before model loading because the remote NVILA processor
  imports `cv2`, which the setup did not install or test. Setup now installs
  `opencv-python-headless`, imports `cv2` in its final verification, and writes
  a versioned `nvila-autogaze-v2` marker. The runner rejects the old marker;
  rerun the idempotent setup before retrying the smoke test.
- **NVILA checkpoint-ID failure (`49621760`):** repaired setup `49621754`
  completed successfully with OpenCV `5.0.0` and wrote the v2 marker. The next
  smoke reached processor construction, then failed because the downloaded
  NVILA processor defaults to the obsolete/private `bfshi/AutoGaze` checkpoint
  identifier. The current official AutoGaze repository documents
  `nvidia/AutoGaze`; the local runner now passes that checkpoint explicitly.
  No predictions were produced, and the five-row smoke must be rerun.
- **Molmo fold-2 start (`49621755`):** the dedicated launcher bypassed the
  previous Slurm environment-retrieval hold. The model server loaded, completed
  multimodal warmup, and accepted its first request; the 140-row run was active
  at the latest inspection.
- **NVILA model-load failure (`49621837`):** the corrected official AutoGaze
  checkpoint loaded successfully, confirming both the ID and processor setup.
  Model construction then stopped because `device_map="auto"` requires the
  optional `accelerate` package. Since this experiment uses exactly one H100,
  the runner now loads the 8B model directly in BF16 and moves it to CUDA,
  removing the unnecessary dependency. No predictions were produced; setup
  does not need to be rerun before the next smoke attempt.
- **NVILA successful smoke (`49622042`):** NVILA-8B-HD-Video with the official
  AutoGaze checkpoint completed all five rows and passed the mechanical smoke
  gate. Mean generation time is `118.31` seconds and the maximum is `172.60`
  seconds, so every measured call is below the 300-second limit. Accuracy is
  only `1/5` (`20.00%`), with predictions `A,D,C,D,D` against gold
  `B,C,C,C,C`; the model is therefore not promoted directly to the 140-row
  fold. The five-row set is label-skewed and not a stable accuracy estimate,
  but this result is poor enough to require prompt/input diagnosis before
  spending a larger allocation.
- **NVILA cached rerun (`49622118`):** the job loaded AutoGaze and NVILA, found
  all five valid cached predictions, and reevaluated them without new video
  inference. It confirms deterministic resume behavior but adds no result; the
  smoke remains `1/5` with `118.31` seconds mean recorded generation time.
- **Molmo fold-2 throughput audit (`49621755`):** the run was healthy at
  `31/140`, with predictions continuing to update and no request failures.
  Recorded model generation averages `90.96` seconds, but extraction,
  timestamping, serialization, and transfer of 256 frames raise observed
  end-to-end throughput to approximately `278` seconds (`4.6` minutes) per
  row. The original ten-hour allocation would likely stop around row 128; the
  launcher now requests 14 hours for future submissions. Predictions resume
  from the validated cached prefix, so a timeout does not discard completed
  rows.

### Open-model partial result and hard iteration suite (2026-08-12)

- **Molmo fold-2 timeout:** job `49621755` reached `132/140` valid predictions
  before Slurm stopped it at the original time limit. On those 132 rows,
  Molmo2-8B scores `98/132` (`74.24%`), versus `116/132` (`87.88%`) for the
  current mean-probability Qwen fusion. Their partial oracle is `120/132`
  (`90.91%`): Molmo uniquely repairs four Qwen errors but introduces 22 losses.
  The four repairs are repeated-occurrence or state-transition questions in
  Shopping, Sightseeing, Daily Activities, and Gardening. Rerunning
  `slurm_longqa_molmo2_8b_video256_fold2.sh` resumes the fixed run directory and
  computes only the remaining eight rows.
- **NVILA decision:** the completed five-row AutoGaze smoke remains `1/5` with
  `118.31` seconds mean model generation. Do not spend a full fold on the
  current prompt/input formulation. AutoGaze may still be useful as a patch
  selector, but NVILA has not earned promotion as an answer model.
- **Fast diagnostic sets:** added deterministic `rescue40`, `guard20`, and
  combined `hard60` configs under `configs/`. Rescue rows cover all five known
  fusion failure mechanisms; guard rows are difficult examples that the fusion
  gets right despite direct-candidate disagreement. These sets use validation
  labels and are for diagnosis only, not accuracy estimation.
- **Diagnostic metric:** `scripts/evaluate_longqa_hard_iteration.py` reports
  rescue fixes, guard regressions, and net gain separately. A method must be
  frozen and confirmed on an untouched balanced fold before promotion.
- **Cached-candidate sanity check:** endpoint 27B repairs `8/40` rescue rows
  while breaking `4/20` guards. Option-quota 27B is `5/40` and `6/20`;
  endpoint 9B is `9/40` and `17/20`; SigLIP2 pivot 9B is `9/40` and `17/20`;
  uncertainty-pivot 9B is `12/40` and `15/20`; option-quota pivot 9B is
  `10/40` and `17/20`. The suite therefore exposes the expected trade-off:
  existing specialists find some missing evidence but are too imprecise for
  unconditional routing.

### Qwen multimodal window-reranking suite prepared (2026-08-12; not yet run)

- **Pipeline:** SigLIP2 performs broad recall over 128 uniformly spaced
  candidates. Qwen3-VL-Reranker-2B then cross-encodes at most 24 shortlisted
  three-frame chronological windows against each complete answer hypothesis.
  Every option receives a fixed minimum quota, and the selector prefers windows
  whose support for one option exceeds support for the competing options. The
  final Qwen3.5-27B answer pack contains 40 endpoint-inclusive global frames and
  up to 24 frames from eight reranked windows, with exactly 64 frames total.
- **Controlled variants:** `balanced` tests multimodal reranking without
  before/after masking. `temporal` uses the same recall, model, frame budget,
  and answer prompt, but identifies the strongest reference-event window and
  prevents option-window selection from the wrong side for explicit `BEFORE`
  and `AFTER` questions. Other temporal operators retain balanced selection.
- **Staging:** first run the three-row temporal smoke. After it succeeds, the
  balanced and temporal `rescue40` jobs are independent and may run together.
  Promote only a variant repairing at least `4/40` rescue failures. Run the
  corresponding `guard20` job next and reject the variant if it breaks more
  than one guard answer. Only a surviving frozen variant should run on balanced
  fold 3.
- **Implementation safeguards:** the official Qwen reranker checkpoint is
  pinned to revision `93eac850736c677b682c67fc0302b03e552a7b16`; proof packs
  and answer predictions resume independently; the reranker process exits
  before Qwen3.5-27B starts; every selected pack validates the exact frame
  budget. Python compilation, all launcher dry runs, and direct option-balance,
  temporal-mask, and 64-frame invariant tests pass. `pytest` is unavailable in
  the environment, so the focused test functions were invoked directly.

### Qwen multimodal reranker smoke result (2026-08-12)

- **Completion:** job `49669496` completed the temporal three-row smoke end to
  end. Qwen3-VL-Reranker-2B produced `3/3` proof packs, and Qwen3.5-27B produced
  `3/3` parseable answers. The apparent Hugging Face `404` metadata probes are
  non-fatal; the pinned checkpoint, processor, and tokenizer all loaded.
- **Selection invariants:** every row used 128 candidates, 24 shortlisted
  windows, eight selected window centers, and exactly 64 final frames. The
  retrieval windows contributed 19--24 unique frames alongside 40 global
  endpoint-inclusive frames. All four options received their required minimum
  evidence quota, and neither `AFTER` row required temporal-filter fallback.
- **Diagnostic signal:** the smoke repairs `1/3` sampled fusion failures. The
  repaired row is from the bucket where one existing rank view had already seen
  the correct answer; the two endpoint/option-consensus errors remain wrong.
  This is a mechanical and directional signal only, not enough data to judge
  accuracy.
- **Runtime/context:** the Qwen3.5 answer stage, including its 150-second vLLM
  startup, completed in 352 seconds for all three rows. Mean context fill was
  `56.91%`. The smoke passes promotion to the two independent rescue40 jobs.

### Qwen reranker rescue40 and Molmo fold-2 results (2026-08-13)

- **Completion:** balanced Qwen-reranker job `49669639` and temporal
  Qwen-reranker job `49669640` completed all `40/40` rescue rows without fatal
  errors. Molmo continuation `49669426` resumed from `132/140` and completed the
  remaining eight rows cleanly.
- **Qwen reranker results:** balanced reranking repairs **`15/40`** known fusion
  errors, while temporal reranking repairs `14/40`. Balanced repairs examples
  in every failure bucket: `4/10` where another direct candidate was correct,
  `3/5` where one rank view saw the answer, `4/8` fusion regressions, `3/14`
  wrong endpoint/option consensuses, and `1/3` complete candidate-pool misses.
  It therefore clears the predeclared `4/40` guard-stage threshold.
- **Temporal ablation:** balanced and temporal predictions differ on exactly one
  row. The temporal restriction changes the pharmacy-between-two-events answer
  from gold `B` to `A`; all other 39 answers are identical. The reranker score
  tensors are identical by construction, and two rows required temporal-filter
  fallback. Current gains therefore come from cross-encoded window reranking,
  not the simple before/after side mask. Reject the temporal variant and advance
  only balanced reranking to guard20.
- **Complementarity:** on rescue40, endpoint 27B repairs eight errors and
  balanced reranking repairs fifteen. Eleven balanced repairs are missed by
  endpoint, while endpoint has four repairs missed by balanced; their diagnostic
  oracle is `19/40`. Adding option-quota raises that oracle to `21/40`. These are
  gold-selected diagnostic figures and cannot be used as a validation estimate.
- **Reranker behavior:** Qwen relevance scores are not saturated (`0.113--0.828`),
  but the mean top-versus-second option margin per window is only `0.045`. The
  final packs contain a mean `21.5` unique reranked frames plus 40 global frames.
  This supports testing the branch as distinct evidence, while also warning
  against treating its relevance margin as calibrated confidence.
- **Molmo full fold:** Molmo2-8B with 256 endpoint-inclusive frames scores
  `104/140` (`74.29%`) on balanced fold 2, versus `124/140` for mean-probability
  Qwen fusion. Molmo uniquely corrects four fusion errors and loses 24
  fusion-correct rows; their oracle is `128/140` (`91.43%`). The four Molmo-only
  fixes concern repeated occurrences or state transitions in Shopping,
  Sightseeing, Daily Activities, and Gardening. Retain Molmo as an audit signal,
  not a general voting member.

### Qwen reranker guard20 rejection (2026-08-13)

- **Completion:** balanced guard job `49675030` completed all `20/20` proof
  packs and predictions without fatal errors. It preserves 15 fusion-correct
  answers but regresses five, scoring `15/20` on the deliberately difficult
  guard set. Mean answer-stage context fill is `56.90%`.
- **Gate decision:** five regressions exceed the predeclared maximum of one.
  Reject balanced reranking as a general replacement and do not run the existing
  140-row balanced-fold launcher. The five losses are three Shopping questions,
  one Hiking/OCR question, and one Fashion question; operators are three
  `GLOBAL`, one `MULTI_TIME`, and one `STATE_CHANGE`.
- **Failure behavior:** all five bad reranker answers disagree with endpoint
  27B, and four agree with option-quota 27B. Thus the new visual evidence can
  inherit option-focused distractor bias rather than independently resolve it.
  Raw reranker score margins do not separate repairs from regressions and must
  not be used as calibrated confidence.
- **Restricted follow-up:** on the gold-selected hard60 audit, a conservative
  rule that restores endpoint only when fusion differs from endpoint and the
  reranked branch independently agrees with endpoint makes four fixes and zero
  regressions (`20/60` to `24/60`). This rule was discovered after inspecting
  hard60 and is not a result. There are 47 endpoint/fusion disagreements in the
  full validation set and seven in untouched balanced fold 3. A valid next test
  would freeze this rule and evaluate only those seven fold-3 disagreements,
  without inspecting their labels beforehand.

### Frozen endpoint-confirmation fold-3 job prepared (2026-08-13; not yet run)

- **Target set:** a label-free subset builder finds exactly seven rows in
  balanced fold 3 where endpoint 27B and the current mean-probability fusion
  disagree. The frozen config is
  `configs/egolongqa_fold3_endpoint_fusion_disagreements7_20260813.json`.
- **Policy:** run balanced Qwen window reranking only on those seven rows. Keep
  fusion on every fold-3 row unless the reranked answer exactly matches endpoint;
  in that case restore endpoint. The merge script validates that the candidate
  predictions cover exactly the seven expected disagreement keys before making
  any changes. Gold labels are consulted only after the 140 frozen predictions
  have been written, to report fixes and regressions.
- **Launcher:**
  `slurm_longqa_qwen3vl_reranker2b_endpoint_confirm_fold3.sh`. The reranker
  proof pack and candidate answers resume independently. Python compilation,
  shell syntax, launcher dry-run construction, and a synthetic end-to-end merge
  test pass.

### Frozen endpoint-confirmation fold-3 result (2026-08-13)

- **Completion:** job `49676131` completed all seven targeted reranker proof
  packs and answers, then produced and evaluated the complete 140-row frozen
  policy output without fatal errors. The reranker candidate itself scores
  `6/7` on the endpoint/fusion disagreements, but candidate accuracy is not the
  routing metric.
- **Policy outcome:** the reranker agrees with endpoint on two of seven rows, so
  the frozen rule makes two overrides. One fixes the irrigation tubing-cutter
  location from fusion `C` to gold `B`; one regresses the repeated blue beach
  cart question from gold/fusion `B` to endpoint/reranker `D`. Net change is
  zero: both baseline fusion and the endpoint-confirmation policy score
  **`127/140` (`90.71%`)** on balanced fold 3.
- **Decision:** the conservative endpoint-restoration rule does not generalize
  beyond the gold-selected hard60 discovery set. Do not expand it to the full
  validation set or use it for test inference. Retain the original
  mean-probability fusion as primary; retain Qwen window reranking only as an
  analysis branch until a stronger, independently confirmed routing signal is
  found.

### Qwen-verified event-burst results (2026-08-13)

- **Smoke completion (`49680624`):** the event-only three-row smoke completed
  mechanically with `3/3` proof packs and parseable Qwen3.5-27B answers. It
  scored `0/3`; mean context fill was `57.07%`. The Hugging Face `404` lines in
  stderr were non-fatal metadata probes.
- **Option-conditioned rescue40 (`49680864`):** the full diagnostic job
  completed without runtime errors and repairs **`15/40`** known fusion errors.
  It repairs `2/3` complete candidate-pool misses, `6/10` cases where another
  direct candidate was correct, `3/5` rank-view misses, `3/8` fusion
  regressions, and `1/14` unanimous endpoint/option errors. Mean context fill
  was `57.08%`; total wall time was approximately 2.52 hours.
- **Complementarity:** the earlier balanced Qwen-window reranker also repairs
  `15/40`, but only seven repairs overlap. Each branch contributes eight unique
  repairs, and their diagnostic oracle is `23/40`. This establishes distinct
  evidence behavior, not a deployable routing policy; the rescue40 set was
  selected using validation labels.
- **Promotion state:** the option-conditioned event-burst branch clears the
  predefined rescue threshold. It must now run on guard20 before any balanced
  fold. The detail-panel answer ablation can reuse the completed rescue40 proof
  pack and run independently while guard20 is evaluated.

### Qwen-verified event-burst guard and panel rejection (2026-08-13)

- **Guard20 (`49681842`):** the job completed all `20/20` predictions without
  fatal errors, but preserves only **`11/20`** fusion-correct answers. Its nine
  regressions fail the predeclared preservation gate by a wide margin. Mean
  context fill was `57.06%`; total runtime was approximately 1.30 hours.
- **Detail-panel rescue40 (`49681843`):** adding object-detail panels produces
  **`15/40`** repairs, exactly matching the event-burst result without panels.
  It changes only three answers relative to that branch: one additional repair,
  one lost repair, and one wrong-to-wrong change. Their oracle is `16/40`, so
  the panels add negligible complementary value. Mean context fill was `57.17%`;
  total runtime was approximately 1.41 hours.
- **Decision:** reject both variants for promotion. Do not run event bursts on
  balanced fold 4 or full validation, and do not spend another guard run on the
  panel variant. The event evidence is useful on selected failures but is much
  too willing to overturn already-correct answers.
- **Exploratory agreement diagnostic:** on the gold-selected hard60 audit, the
  event-burst and balanced Qwen-reranker branches agree on eight changes among
  rescue40; seven are repairs. They agree on two changes among guard20; both are
  regressions. Applying only these agreements changes hard60 from `20/60` to
  `25/60`, but this post-hoc rule remains unsafe and is not a validation result.

### Frozen candidate-agreement fold-4 experiment prepared (2026-08-13; not yet run)

- **Target:** 11 label-free endpoint/fusion disagreements in untouched balanced
  fold 4, stored in
  `configs/egolongqa_fold4_endpoint_fusion_disagreements_20260813.json`.
- **Independent candidates:** balanced Qwen window reranking and
  option-conditioned verified event bursts run only on those 11 rows. Their GPU
  jobs are independent and may run concurrently.
- **Frozen policies:** start from primary fusion. `dual_agreement` changes an
  answer only when both evidence branches agree on the same alternative;
  `endpoint_confirmed_agreement` additionally requires that alternative to equal
  the endpoint-uniform answer. Both complete 140-row prediction files are
  written before labels are used for evaluation.
- **Launchers:**
  `slurm_longqa_qwen3vl_reranker2b_agreement_fold4.sh`,
  `slurm_longqa_qwen35_27b_verified_bursts_agreement_fold4.sh`, then
  `slurm_longqa_qwen35_27b_candidate_agreement_fold4_merge.sh` after both GPU
  jobs finish successfully.

### Frozen candidate-agreement fold-4 result (2026-08-14)

- **Candidate completion:** Qwen window-reranker job `49682131` and verified
  event-burst job `49682132` both completed all `11/11` targeted rows without
  fatal errors. The reranker candidate scores `7/11`; event bursts score `8/11`.
  Mean answer-stage context fill is `56.9%` and `57.05%`, respectively.
- **Agreement outcome:** the two branches agree against primary fusion on only
  one row, and that answer also matches endpoint-uniform Qwen. Both frozen
  policies therefore make the same single override: the radio-display state
  change moves from fusion `A` to gold `B`.
- **Held-out fold result:** primary fusion scores `122/140` (`87.14%`) on
  balanced fold 4. Dual agreement and endpoint-confirmed agreement both score
  **`123/140` (`87.86%`)**, with one repair and zero regressions.
- **Interpretation:** the conservative agreement gate passes its first untouched
  fold, but the evidence is one changed example. It is suitable for a cautious
  full-validation candidate and ablation, not yet strong enough to replace the
  primary submission without comparing complete outputs.

### Full candidate-agreement validation prepared (2026-08-14; not yet run)

- The fold-4 policy is frozen and expanded to all 47 endpoint/fusion
  disagreements. Only these 47 rows receive new GPU inference; all remaining
  rows retain primary fusion predictions.
- Run the balanced Qwen reranker and verified event-burst launchers concurrently,
  then run the CPU merge after both succeed. The merge again writes complete
  700-row dual-agreement and endpoint-confirmed prediction files before
  evaluating either policy.

### Full candidate-agreement validation result (2026-08-14)

- **Candidate completion:** reranker job `49691478` and verified event-burst job
  `49691479` completed all `47/47` endpoint/fusion disagreements. The candidate
  branches score `30/47` and `29/47`, respectively. Mean context fill is
  `56.87%` and `57.02%`.
- **Parser handling:** two event-burst responses did not contain a parseable
  option letter. The agreement router now treats malformed auxiliary responses
  as abstentions, preserving primary fusion. No GPU rerun was required.
- **Dual agreement:** five overrides yield three repairs, one regression, and
  one wrong-to-wrong change. Full validation improves from `618/700` (`88.29%`)
  to **`620/700` (`88.57%`)**.
- **Endpoint-confirmed agreement:** four overrides yield the same three repairs
  and one regression, without the wrong-to-wrong change. It also scores
  **`620/700` (`88.57%`)** and is the cleaner policy because it changes fewer
  baseline answers.
- **Decision:** retain endpoint-confirmed agreement as a positive validation
  candidate, but not as proof of test improvement. The net gain is two examples,
  and the full validation labels have now been observed. Keep the original
  `618/700` fusion submission as the lower-risk reference.

### Qwen3.8-27B evaluation prepared (2026-08-14; not yet run)

- The newly released Apache-2.0 `Qwen/Qwen3.8-27B` is a dense native
  vision-language model with image/video support. Local metadata loading confirms
  that it reuses the `Qwen3_5ForConditionalGeneration` architecture supported by
  the current Transformers and vLLM environment.
- The backend family detection now applies non-thinking defaults, the Triton GDN
  prefill path, and the Qwen reasoning parser to both Qwen3.5 and Qwen3.8.
- The staged evaluation consists of endpoint64 smoke5, matched fold-4 endpoint64,
  cached option-quota fold 4, and Qwen3.8 evidence-rank fusion. The two fold-4
  answer branches may run concurrently only after smoke succeeds; fusion waits
  for both answer branches.
