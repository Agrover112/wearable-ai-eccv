# EgoLongQA — EDA (Steps 1-5)

**Rows:** 700  |  **Columns:** video_path, question, answer, mcq_options, mcq_answer, category

## 1. Structure
- Unique videos: **700**
- Questions per video: min 1, max 1, mean 1.00
- Categories: **13**

| category | count | % |
|---|---:|---:|
| Daily Activities | 127 | 18.1% |
| Sightseeing | 103 | 14.7% |
| Travel-Tourism | 93 | 13.3% |
| Shopping | 72 | 10.3% |
| Travel-Sightseeing (Outdoors) | 59 | 8.4% |
| Travel-Sightseeing (Indoors) | 53 | 7.6% |
| Hiking-Outdoors | 37 | 5.3% |
| Hobbies-Daily Activities | 35 | 5.0% |
| Gardening | 32 | 4.6% |
| Pets, social gatherings with friends and family | 25 | 3.6% |
| Outdoor Activities and Sports | 23 | 3.3% |
| Events | 22 | 3.1% |
| Fashion Advice | 19 | 2.7% |

## 2. MCQ answer balance
| letter | count | % |
|---|---:|---:|
| A | 8 | 1.1% |
| B | 205 | 29.3% |
| C | 444 | 63.4% |
| D | 43 | 6.1% |

- Most common: **C** (63.4%) — uniform would be 25%
- Options per question: 4-8 (mode 4)

## 3. Text stats (characters)
| field | mean | min | max |
|---|---:|---:|---:|
| question | 146 | 65 | 362 |
| answer | 61 | 1 | 438 |
| mcq_options | 271 | 19 | 1337 |

- Question first words: what (249), after (204), i (60), which (45), earlier (33), when (24), early (16), where (15)

## 4. Temporal-reasoning share
- Temporal questions: **554 / 700 = 79.1%**
- Top cues: after (285), first (183), earlier (117), before (83), last (39), next (31)

## 5. Shortcut check — length bias
- Correct option is the **longest**: 170/700 = **24.3%** (random ~25%)
  - >25% ⇒ a 'pick longest' baseline beats chance ⇒ length is a real shortcut.
