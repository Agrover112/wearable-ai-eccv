# LongQA answer calibration - 2026-08-05

## Status

This is a completed result, not a projected experiment. The calibrated system
produces 700 valid answers and scores **619/700 (88.43%)** locally. It uses two
already completed visual-model runs and a small statistical decision rule; no
additional GPU inference was required.

Only 560 questions are strictly held out. We used the fixed dev140 split to
learn the decision rule, then froze it before evaluating the other 560:

- dev140: 130/140 (92.86%);
- held-out val560: 489/560 (87.32%);
- combined 700-row prediction file: 619/700 (88.43%).

The method is implemented in `scripts/calibrate_longqa_answer_prior.py`. Its
predictions and diagnostics are stored in
`runs/egolongqa/qwen35_27b_endpoint_devprior_calibrated_full_2026-08-05/`.

## Model 1: candidate-blind Qwen3.5-27B

The first answer comes from **Qwen3.5-27B**. It receives one set of 64 images,
shown in chronological order, together with the question and all four answer
options.

The 64-image input combines three kinds of coverage:

1. images spread regularly across the entire video;
2. images selected because SigLIP2 finds them relevant to the question and
   answer options, including nearby moments;
3. images from regions where a smaller Qwen model was uncertain about the
   answer.

Duplicate images are removed and the final images are sorted by timestamp.
The 27B model then answers once, without long-form thinking.

"Candidate-blind" means that the 27B model is **not shown the answers predicted
by the smaller models**, their vote counts, or a suggested answer. It still
sees the original question and all four options. The smaller models help choose
images, but they cannot directly tell the 27B model which option to select.

After repairing three invalid outputs with the predefined label-free fallback,
this model scores **572/700 (81.71%)**.

## Model 2: endpoint-uniform Qwen3.5-9B

The second answer comes from **Qwen3.5-9B**. It uses a deliberately simple and
independent view of the video: 64 images are spaced evenly from the beginning
to the end, with the exact first and last images included. Each image has a
maximum pixel budget of 451,584, approximately 672 by 672 pixels.

This model receives the same question and four options and returns one option
letter. It does not use SigLIP2 retrieval, uncertainty selection, or the 27B
prediction. It scores **542/700 (77.43%)** by itself.

Although the 9B model is weaker overall, it is useful because it sees a
different set of evidence. It can disagree with the 27B model when the focused
64-image pack missed an event or when the 27B model selected an unreliable
option position.

## The missed problem: answer-position calibration

The correct option letters are not evenly distributed:

| Split | A | B | C | D |
|---|---:|---:|---:|---:|
| dev140 | 1 | 39 | 91 | 9 |
| held-out val560 | 7 | 166 | 353 | 34 |
| full set | 8 | 205 | 444 | 43 |

The imbalance is very similar in dev140 and val560. Option C is correct in
roughly two thirds of the questions, while A is correct only eight times.

The 27B model nevertheless predicts A on 72 questions. Only 7 of those answers
are correct. By comparison, 366 of its 378 C predictions are correct. All 72 A
predictions are literal model outputs; this is not a parser or fallback bug.

Our earlier voting and verifier methods treated every predicted letter as if it
had the same reliability. That discarded an important signal: a C prediction
from this 27B pipeline is much more trustworthy than an A prediction.

## What the Bayesian calibrator learns

The calibrator is not another neural network and it does not inspect images.
It is a small table of counts learned from the labeled dev140 examples.

For each dev question, it records three values:

1. the correct option letter;
2. the letter predicted by the 27B model;
3. the letter predicted by the endpoint-uniform 9B model.

From these 140 examples, it estimates:

- how frequently A, B, C, and D are correct;
- how often the 27B model produces each letter when the true answer is A, B,
  C, or D;
- how often the endpoint model produces each letter for each true answer.

For a new question, suppose the 27B model says A and the endpoint model says B.
The calibrator evaluates four possibilities: "the true answer is A", "the true
answer is B", "the true answer is C", and "the true answer is D". For each
possibility it combines:

1. how common that true letter was in dev140;
2. how likely the 27B output would be under that true letter;
3. how likely the endpoint output would be under that true letter.

It selects the true letter with the largest combined score. In compact form:

```text
score(true letter) = frequency of true letter
                   x reliability of the 27B output
                   x reliability of the endpoint output
```

Small smoothing counts are added to every combination so that a rare or unseen
dev140 combination is not assigned a probability of zero. The smoothing value
was selected from tied best settings using leave-one-out evaluation on dev140;
no val560 labels were used to choose it.

Examples of the resulting mapping are:

| 27B answer | Endpoint answer | Calibrated answer |
|---|---|---|
| A | A | C |
| A | B | B |
| A | C | C |
| B | C | C |
| C | B | C |
| D | B | B |
| D | D | D |

This is more informative than majority voting. When the two models disagree,
there is no majority; the calibrator uses the reliability learned for that
specific pair of outputs. It also corrects cases where both models repeat the
same historically unreliable letter.

## Why the result improves

The candidate-blind 27B model starts at 572 correct answers. Calibration changes
103 predictions and finishes with 619 correct answers, a net improvement of 47.
The final prediction distribution is B=203, C=451, and D=46; it predicts no A.

Predicting no A necessarily loses the eight true-A questions. It still improves
overall because the original models incorrectly selected A far more often than
that. This is a deliberate calibration trade-off, not evidence that option A
can never be correct.

The repository evaluator confirms:

- overall accuracy: 619/700 (88.43%);
- non-C accuracy: 81.25%;
- temporal-question accuracy: 88.89%;
- 700 unique rows and no invalid answers.

## Limitations and next experiment

The 88.43% number includes the 140 examples used to learn the count tables. The
best estimate of performance on unseen questions is therefore the held-out
489/560 result, or 87.32%.

The calibration also assumes the hidden test is produced with a similar option
distribution. Dev140 and val560 closely agree, which supports that assumption,
but it is still a risk. A hidden set with uniformly shuffled correct options
would require recalibration.

The calibrated model has 71 errors on val560. Another existing direct model is
correct on 46 of those questions. The next experiment should therefore test
**27B option-order rotation** only on ambiguous cases:

1. reuse exactly the same visual evidence;
2. cyclically rotate the four displayed options;
3. run the 27B model four times;
4. map every response back to the original answer text;
5. combine the four decisions and preserve the calibrated answer when the
   rotations do not provide consistent contrary evidence.

This directly tests whether the remaining B/C and B/D errors are caused by
option position. It is a more targeted next step than another global change to
frame sampling.

## Prepared rotation runs

The GPU scorer reuses the exact 64 frame indices recorded by the completed 27B
run and the same candidate-blind prompt. It does not add timestamps, change the
frame selector, or expose another model's answer. A single scoring run records
all four mapped letter distributions. Offline evaluation then produces:

- `average`: select the highest geometric-mean probability across rotations;
- `consensus`: switch only when at least three rotations select the same
  underlying answer;
- `endpoint_confirmed`: switch only when the rotation-average answer also
  matches the independent endpoint-uniform answer.

Run the five-question smoke test first:

```bash
sbatch slurm_longqa_qwen35_27b_option_rotation_smoke.sh
```

If all 20 scoring calls finish and no question exceeds 300 seconds, run dev140:

```bash
sbatch slurm_longqa_qwen35_27b_option_rotation_dev.sh
```

Only after inspecting and freezing the dev result should the val560 launcher be
used:

```bash
sbatch slurm_longqa_qwen35_27b_option_rotation_val560.sh
```

## Rotation result

The dev140 job completed all 560 scoring calls without errors. Four rotations
took 68.62 seconds per question on average and at most 119.14 seconds, so the
method satisfies the 300-second limit.

It did not improve accuracy:

| Policy | Correct | Changes | Fixes | Regressions |
|---|---:|---:|---:|---:|
| Original 27B | 121/140 | - | - | - |
| Rotation average | 117/140 | 9 | 1 | 5 |
| 3-of-4 consensus | 120/140 | 1 | 0 | 1 |
| Endpoint-confirmed | 121/140 | 0 | 0 | 0 |

All four rotations selected the same underlying answer on 116/140 questions.
The original 27B answer was still wrong on eight of those unanimous questions.
Among the nine questions where the original model predicted A, rotation still
selected A on seven. This means the main error is not a mechanical preference
for whichever answer is displayed at position A or C. The model often prefers
the same incorrect answer content regardless of its displayed letter.

The val560 rotation job should therefore not be run. Rotation remains useful as
an audit showing that the answer-prior calibration gain cannot be replaced by
simple option-order invariance.
