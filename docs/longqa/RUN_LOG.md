# Wearable AI LongQA Run Log

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
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12` | 2026-07-12 | EgoLongQA dev140 | Qwen/Qwen3-VL-8B-Instruct | vLLM + pivot/uniform disagreement verifier | 1x H100 NVL | 140 (19 verifier calls) | 64 on disagreements | 451,584 px (~672x672) | 12m56s | **0.7929** | **111/140** | verifier calls: mean 27,967 tok; p95 28,033; max 57.44% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12/` |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12` | 2026-07-12 | EgoLongQA val | Qwen/Qwen3-VL-8B-Instruct | vLLM + pivot/uniform disagreement verifier | 1x H100 NVL | 700 (122 verifier calls) | 64 on disagreements | 451,584 px (~672x672) | 1h22m07s | **0.7700** | **539/700** | verifier calls: mean 27,942 tok; p95 28,033; max 57.44% of 49K | `runs/egolongqa/qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12/` |
| `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13` | 2026-07-13 | EgoLongQA dev140 | yanziang/InternVideo3-8B-Instruct | HF/PyTorch | 1x H100 | 140 | 64 | 451,584 px per-frame cap | 1h42m20s | 0.6429 | 90/140 | mean 14,266 tok; p95 14,363; max 5.54% of 262K | `runs/egolongqa/internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13/` |

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

## Job Status Snapshot (2026-07-12 20:18 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_full_2026-07-12` | 49050528 | Completed | 700/700; accuracy 0.7543; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_dev_2026-07-12` | 49052006 | Completed | 140/140; accuracy 0.7929; final artifacts archived |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_dev_2026-07-12` | 49052008 | Completed | 140/140; 19 verifier calls; accuracy 0.7929; final artifacts archived |
| `qwen3_vl_8b_vllm_pivot_uniform_disagreement_verifier_px451584_full_2026-07-12` | 49052012 | Completed | 700/700; 122 verifier calls; accuracy 0.7700; final artifacts archived |
| `qwen3_vl_8b_vllm_siglip2_operator_router_pivot24_globalu64_px451584_full_2026-07-12` | 49052007 | Running | proof packs 700/700; predictions 201/700 at 20:15 CEST and still advancing; working artifacts retained |

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

## Job Status Snapshot (2026-07-13 02:41 CEST)

| Job | Slurm ID | Status | Evidence |
| --- | ---: | --- | --- |
| `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13` | 49058687 | Completed | 140/140; accuracy 0.6429; runtime 1h42m20s; final predictions, evaluation, and diagnostics archived |

## Dev140 Fair Comparison

The dev140 subset is `configs/egolongqa_dev140_seed20260709.json`. Previous full-validation prediction files were re-scored by matching stable `video_path||question` keys, so these numbers are directly comparable to the new dev-only runs.

| Run ID | Dev140 accuracy | Correct | Temporal acc. | Non-C acc. |
| --- | ---: | ---: | ---: | ---: |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px451584_dev_2026-07-11` | **0.7929** | **111/140** | 0.7970 | 0.7551 |
| `qwen3_vl_8b_vllm_uniform64_px451584_dev_2026-07-11` | **0.7857** | **110/140** | **0.8045** | **0.7959** |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_anc24_final64_px200704_dev_2026-07-11` | **0.7786** | **109/140** | 0.7820 | 0.7551 |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px451584_dev_2026-07-11` | 0.7714 | 108/140 | 0.7820 | 0.7755 |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_covfill_anc32_tgt6_final64_px200704_dev_2026-07-11` | 0.7571 | 106/140 | 0.7669 | 0.7143 |
| `qwen3_vl_8b_vllm_uniform32_px451584_dev_2026-07-11` | 0.7571 | 106/140 | 0.7594 | 0.6939 |
| `qwen3_vl_8b_vllm_uniform64_px200704_2026-07-07` | 0.7429 | 104/140 | 0.7444 | 0.7551 |
| `qwen3_vl_8b_vllm_siglip2_eventlet8x3_anc32_bnd8_final64_px200704_dev_2026-07-11` | 0.7429 | 104/140 | 0.7519 | 0.7347 |
| `qwen3_vl_8b_vllm_siglip2_temporal_pivot_structured_anc24_final64_px200704_dev_2026-07-11` | 0.7429 | 104/140 | 0.7444 | 0.7755 |
| `qwen3_vl_8b_vllm_uniform64_px200704_prompt_combined_dev_2026-07-09` | 0.7429 | 104/140 | 0.7444 | 0.7551 |
| `qwen3_vl_8b_vllm_cft_question_options_nms10_prompt_combined_dev_2026-07-09` | 0.7429 | 104/140 | 0.7444 | 0.7347 |
| `qwen3_vl_8b_vllm_uniform96_px451584_dev_2026-07-11` | 0.7286 | 102/140 | 0.7368 | 0.7143 |
| `qwen3_vl_8b_vllm_siglip2_c128_top24_anc8_px200704_prompt_baseline_dev_2026-07-11` | 0.7214 | 101/140 | 0.7293 | 0.7347 |
| `qwen3_vl_8b_vllm_siglip_c128_top24_anc8_px200704_2026-07-06` | 0.7214 | 101/140 | 0.7293 | 0.7347 |
| `qwen3_vl_8b_vllm_32frames_px200704_2026-07-04` | 0.7143 | 100/140 | 0.7143 | 0.6939 |
| `qwen3_vl_8b_vllm_hybrid_c128_top16_anc64_final64_px200704_2026-07-08` | 0.7071 | 99/140 | 0.7143 | 0.7551 |
| `qwen3_vl_8b_vllm_siglip2_option_contrast4x3_anc16_final64_px200704_dev_2026-07-11` | 0.7000 | 98/140 | 0.7068 | 0.7347 |
| `qwen3_vl_8b_vllm_timeline_t64_a32_px200704_2026-07-07` | 0.7000 | 98/140 | 0.7068 | 0.6939 |
| `qwen3_vl_8b_vllm_uniform48_px313600_dev_2026-07-11` | 0.6857 | 96/140 | 0.6992 | 0.6531 |
| `qwen2_5_vl_7b_hf_32frames_full_2026-07-03` | 0.6714 | 94/140 | 0.6692 | 0.7347 |
| `qwen3_vl_8b_vllm_32frames_full_2026-07-03` | 0.6714 | 94/140 | 0.6692 | 0.6735 |
| `internvideo3_8b_hf_uniform64_px451584_dev140_2026-07-13` | 0.6429 | 90/140 | 0.6541 | 0.7143 |
| `qwen3_vl_8b_vllm_default32_full_2026-07-02` | 0.6214 | 87/140 | 0.6165 | 0.5510 |
| `qwen2_5_vl_7b_hf_default32_full_2026-07-02` | 0.6071 | 85/140 | 0.6015 | 0.6531 |
| `qwen3_vl_8b_vllm_openqa_minilm_uniform64_px200704_dev_2026-07-11` | 0.5286 | 74/140 | 0.5263 | 0.6531 |
| `qwen3_vl_8b_vllm_video_blind_prompt_baseline_dev_2026-07-11` | 0.5214 | 73/140 | 0.5338 | 0.4694 |

Dev140 inference:

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
- Uniform64/672 and temporal pivot disagree on 27 rows and each uniquely solves 13 and 12 respectively. Their oracle union is `122/140` (`0.8714`), making them strong candidates for the later disagreement verifier.
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

### `internvideo3_timestamp_profile_2026-07-14`

- **Purpose:** Measure challenge-timed inference at 512, 1,024, and 2,048 native video frames before selecting a larger InternVideo3 evaluation configuration.
- **Sample:** `31cdcd6a7135a92b.mp4`, a 600-second first/last temporal question about decorative-light colors. The 64-frame InternVideo3 run predicted D; the gold answer is C.
- **Implementation:** Raw MP4 input through the checkpoint's native video processor; TorchCodec 0.10 CPU decoding; 65,536-131,072 pixels per frame; automatic timestamp labels for every two-frame temporal patch; timestamp-grounded prompt with a concise evidence trace and terminal `Final answer: X`; BF16; SDPA; deterministic generation; one H100 NVL. The challenge excludes processor/frame-extraction time from its 300-second inference limit.
- **SLURM jobs:** `49083271` for the direct-answer control and `49083291` for the corrected timestamp-grounded prompt.

| Frames | Effective fps | Prompt tokens | Processor | Inference | Peak allocated / reserved | Grounded answer |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 512 | 0.85 | 32,692 | 13.49s | 9.09s | 24.46 / 28.60 GiB | **C** |
| 1,024 | 1.71 | 65,157 | 18.78s | 25.56s | 31.39 / 39.68 GiB | D |
| 2,048 | 3.41 | 130,087 | 28.86s | 114.10s | 45.27 / 61.84 GiB | D |

- **Timestamp check:** All configurations cover the full video. The 512-frame prompt spans 0.6-599.4 seconds with 256 timestamp labels; the 2,048-frame prompt spans 0.1-599.9 seconds with 1,024 labels.
- **Finding:** All frame counts fit comfortably inside the 300-second model-inference limit, including a 153-token explanation at 2,048 frames. Accuracy is not monotonic on this diagnostic: 512 frames recovers the correct option and identifies an early occurrence near 100 seconds, while the denser settings incorrectly treat the late 439-second house as the first occurrence. This is a long-context evidence-selection failure, not missing temporal coverage.
- **Recommendation:** Use 512 frames as the first promotion candidate, but compare 512 and 1,024 on a small stratified temporal subset before dev140. Do not promote 2,048 solely because it fits the latency and memory budgets.
- **Artifacts:** `runs/egolongqa/internvideo3_timestamp_profile_2026-07-14/profile.json`; direct-answer control in `profile_direct_answer.json`.

### `internvideo3_timestamp512_pilot30_fa2_2026-07-14`

- **Purpose:** Test whether the successful 512-frame timestamp-grounded diagnostic generalizes before promoting the configuration to dev140, and validate FlashAttention-2 on the production H100 path.
- **Subset:** `configs/internvideo3_temporal_pilot30_seed20260714.json`; 24 temporal questions with four prior 64-frame errors and four prior successes in each of first/last, before/after, and other-temporal groups, plus three prior errors and three prior successes marked non-temporal. This deliberately balanced gate is not an unbiased dev140 accuracy estimate.
- **Implementation:** `yanziang/InternVideo3-8B-Instruct` at revision `c4602918b65225650d152db2850fe34e01d21fcd`; raw MP4 input through TorchCodec and the checkpoint's native processor; 512 uniform frames; 65,536-131,072 pixels per frame; 32,615 mean prompt tokens; native timestamp labels; timestamp-grounded prompt; deterministic generation with 192 output tokens; BF16; Hugging Face/PyTorch backend. FlashAttention `2.8.3.post1` was compiled for SM90 against PyTorch `2.10.0+cu128` and CUDA `12.8`.
- **Command:** `python baselines/longqa/run_internvideo3_timestamp_pilot.py --input ../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl --video-folder /scratch/inf0/user/kkumar/val --subset-file configs/internvideo3_temporal_pilot30_seed20260714.json --output runs/egolongqa/internvideo3_timestamp512_pilot30_fa2_2026-07-14/predictions.jsonl --attn-implementation flash_attention_2 --frames 512 --min-pixels 65536 --max-pixels 131072 --max-new-tokens 192`
- **SLURM jobs:** `49110433` for the SDPA/A100 comparison and `49112806` for FA2/H100. Wall times were `34m44s` and `23m02s`, respectively.
- **Result:** The paired 64-frame baseline has `15/30`; timestamped 512-frame InternVideo3 has `19/30` (`63.33%`), with six wrong-to-correct changes and two correct-to-wrong changes. FA2 and SDPA agree on all 30 final option letters.
- **Subtype changes:** first/last `4/8 -> 4/8`; before/after `4/8 -> 4/8`; other temporal `4/8 -> 7/8`; controls `3/6 -> 4/6`. The gain is concentrated in general cross-time comparisons, not first/last or before/after questions.
- **FA2/H100 latency:** processor mean/p50/p95/max `39.51/31.98/65.11/70.98s`; inference mean/p50/p95/max `5.90/4.93/10.52/12.25s`; combined processor+transfer+inference mean/p50/p95/max `45.55/37.08/71.95/76.04s`. All samples are comfortably within 300 seconds even when preprocessing is included.
- **Evidence audit:** 20/30 responses contain only an option letter, only three cite timestamps, and one correct first/last response cites its alleged first event later than its alleged last event. The one-shot prompt therefore improves paired accuracy but does not yet provide reliable temporal evidence traces.
- **Parser correction:** The original summary counted `18/30` because the fallback parser interpreted the lowercase article in `D. It defecated near a bush...` as option A. `normalize_answer` now restricts prose fallback matches to uppercase option letters while retaining lowercase single-letter and explicit answer declarations; the corrected result is `19/30`.
- **Recommendation:** Keep FA2 as the H100 backend and retain 512 frames as a useful branch, but do not replace the 64-frame path globally. The next experiment should use chunked/coarse-to-fine evidence selection for first/last and before/after questions, where this pilot shows no aggregate gain.
- **Artifacts:** `runs/egolongqa/internvideo3_timestamp512_pilot30_fa2_2026-07-14/` and the SDPA comparison in `runs/egolongqa/internvideo3_timestamp512_pilot30_sdpa_2026-07-14/`.

### `internvideo3_timestamp512_fa2_dev140_2026-07-14`

- **Purpose:** Evaluate the promoted native-video, timestamp-grounded 512-frame InternVideo3 configuration on the complete reduced dev140 split, including all 133 temporal and 7 non-temporal questions.
- **Implementation:** Same checkpoint and processor path as the pilot; 512 uniform native-video frames; 65,536-131,072 pixels per frame; 32,703 mean prompt tokens; timestamp-grounded prompt; deterministic generation with 192 output tokens; BF16; Hugging Face/PyTorch backend; FlashAttention-2 on one H100 NVL.
- **Command:** `sbatch scripts/slurm_internvideo3_timestamp512_dev140.sh`
- **SLURM job:** `49114620` on `gpu24-h100-06`; completed with exit code 0 in `1h55m01s` under a three-hour allocation.
- **Result:** Accuracy `0.6571`, with `92/140` correct. The prior 64-frame InternVideo3 result was `90/140` (`0.6429`), while the Qwen3-VL dev140 reference was `110/140` (`0.7857`). Relative to 64-frame InternVideo3, the run had 19 wrong-to-correct changes and 17 correct-to-wrong changes.
- **Question types:** Temporal accuracy changed from `87/133` to `88/133`; non-temporal from `3/7` to `4/7`. Exclusive subtype counts changed as follows: cross-time ordering `23/41 -> 29/41`, OCR/named detail `30/39 -> 26/39`, color/appearance `4/11 -> 5/11`, fine object identity `6/9 -> 5/9`, and spatial/location remained `27/38`.
- **Answer distribution:** Gold `A/B/C/D = 1/39/91/9`; 64-frame predictions `21/35/58/26`; 512-frame predictions `21/40/64/14`, plus one unparsed response. Correct C answers increased from 55 to 60 while correct B answers decreased from 28 to 25, so the small aggregate gain is partly a calibration shift toward the C-heavy split.
- **Latency:** Model inference mean/p50/p95/max `5.64/4.90/9.93/11.43s`; complete processor+transfer+inference mean/p50/p95/max `48.88/45.90/77.70/90.38s`. Every query remained below the 300-second challenge budget even when preprocessing was included.
- **Generation behavior:** 108/140 responses were a single option letter and 12 cited timestamps. One response exhausted all 192 tokens while narrating the video without emitting a final option; it is counted as incorrect.
- **Inference:** The dense timestamp path helps cross-time ordering but loses OCR/named-detail accuracy, consistent with exchanging spatial resolution for temporal coverage. It should not globally replace the 64-frame path. A question-conditioned router or coarse-to-fine pass should retain high-resolution evidence for text and object details while using the denser timeline for ordering questions.
- **Artifacts:** `runs/egolongqa/internvideo3_timestamp512_fa2_dev140_2026-07-14/` contains the complete predictions and summary.
