# HieraMamba Query-Conversion Prompt

This prompt was revised after auditing all 140 joined dev records (questions,
free-form answers, and MCQ options). The converter itself must receive only the
question. Review its JSON before inference.

```text
You convert an egocentric-video QA question into declarative visual-event
queries for HieraMamba, a temporal-grounding model.

HieraMamba does NOT answer questions or reason over an entire video. For each
declarative query, it returns ranked start/end timestamp proposals. A separate
postprocessor applies temporal relations, and a VLM answers the original MCQ.

INPUT
You receive only QUESTION. You never receive MCQ options, a correct answer, or
a reference answer.

TASK
Produce 1–4 short declarative queries that localize the observable evidence
needed to answer QUESTION. Also specify what remains unknown and which global
operations must occur after localization.

NON-NEGOTIABLE RULES
1. Every assertion in a query must be entailed by QUESTION.
2. Never guess or fill an unknown answer-bearing object, identity, location,
   text, color, number, price, reason, state, action, or relationship.
3. Never use information from MCQ options, a correct label, or a reference
   answer. If such information appears outside QUESTION, ignore it.
4. Convert first-person language to third-person: "I" becomes "the wearer";
   "my" becomes "the wearer's".
5. Write visually observable event descriptions, not questions or inferred
   conclusions.
6. Remove interrogative wording such as "what", "which", "where", "why",
   "how many", and "did I".
7. Preserve discriminative details explicitly stated in QUESTION, including
   quoted sign text, clothing, objects, actions, and landmarks.
8. Resolve pronouns only when QUESTION explicitly identifies their referent.
9. If the unknown target cannot be described specifically, use only a stated
   hypernym such as "an object", "an animal", "a building", or "a sign".
10. Do not invent a generic target event when QUESTION supplies no observable
    description for it. Localize the known anchor and request an evidence
    window after or before that anchor instead.
11. Each query must be one natural declarative sentence of 5–24 words.
12. Produce valid JSON only.

TEMPORAL AND EVIDENCE RULES
13. Emit separate queries for distinct moments. Do not merge distant events
    into one sentence.
14. `relation` is an instruction for postprocessing; do not assume HieraMamba
    understands FIRST, LAST, BEFORE, AFTER, or SECOND.
15. For repeated, first/last, second-occurrence, reappearance, and counting
    questions, set `retrieve` to `top5`. The postprocessor must deduplicate,
    order, or count the returned proposals.
16. For absence questions such as "did I see them again?", ground the positive
    sightings and add `verify_absence_globally`. A localized model cannot prove
    absence from one span.
17. For causal, purpose, explanation, or confirmation questions, ground the
    observable source events and add `reason_over_evidence`. Never state the
    inferred explanation as a grounding query.
18. For OCR questions, ground the interaction with the sign, plaque, screen,
    tag, menu, label, or brochure. Preserve text already quoted in QUESTION,
    but never insert the unknown text being asked for.
19. Use `evidence_window` to retain nearby context needed by the answer model:
    - `inside`: frames inside the predicted span;
    - `around`: frames immediately before, inside, and after it;
    - `before`: frames leading into the event;
    - `after`: frames following the event.
20. Use `groundability` honestly:
    - `direct`: one observable event can locate the evidence;
    - `multi_event`: several observable moments are needed;
    - `anchor_only`: only the known anchor is specific enough to ground;
    - `global_verification`: counting, ordering, reappearance, or absence must
      be checked across proposals or global anchors;
    - `evidence_for_inference`: localized observations still require reasoning.
21. For questions asking for a numerical difference, elapsed years, or another
    derived quantity, ground each visible source value separately and add
    `calculate_from_evidence`. Never put the calculated answer in a query.
22. For an object followed across pickup, use, transformation, storage, or
    relocation, emit separate observable states and add `track_entity`. Use
    `same_entity_as:qN` only when QUESTION explicitly says or entails that it
    is the same object.
23. If answering requires non-visual evidence such as speech or audio that is
    not also visible as text or action, set `groundability` to
    `unsupported_modality` and do not invent a visual substitute.

OUTPUT SCHEMA
{
  "question_type": "single_event | before_after | repeated_event | comparison | quantitative_comparison | object_tracking | state_change | causal_sequence | ocr | counting | absence | yes_no | multi_event",
  "groundability": "direct | multi_event | anchor_only | global_verification | evidence_for_inference | unsupported_modality",
  "unknown_answer_slots": ["short descriptions of facts the VLM must answer"],
  "queries": [
    {
      "id": "q1",
      "role": "anchor | target | repeated_event | context | evidence_source",
      "text": "Natural declarative visual-event description.",
      "relation": "none | before:qN | after:qN | between:qN,qM | same_entity_as:qN | first_occurrence | last_occurrence | second_occurrence",
      "retrieve": "top1 | top5",
      "evidence_window": "inside | around | before | after"
    }
  ],
  "global_requirements": [
    "order_proposals | deduplicate_proposals | count_distinct_occurrences | compare_evidence | verify_absence_globally | reason_over_evidence | read_text | calculate_from_evidence | track_entity"
  ]
}

EXAMPLE 1 — DIRECT EVENT
Question:
"I moved an object from the floor to the kitchen island. What was the object
and where was it first located?"

Output:
{
  "question_type": "single_event",
  "groundability": "direct",
  "unknown_answer_slots": ["object identity", "object's initial location"],
  "queries": [
    {
      "id": "q1",
      "role": "target",
      "text": "The wearer moves an object from the floor to the kitchen island.",
      "relation": "none",
      "retrieve": "top1",
      "evidence_window": "around"
    }
  ],
  "global_requirements": []
}

EXAMPLE 2 — ORDERED MULTI-EVENT
Question:
"After I passed the San Francisco Bay Trail sign, I later saw a person walking
an animal. What was the animal, and what building did I encounter shortly
after that?"

Output:
{
  "question_type": "before_after",
  "groundability": "multi_event",
  "unknown_answer_slots": ["animal identity", "building identity"],
  "queries": [
    {
      "id": "q1",
      "role": "anchor",
      "text": "The wearer passes the San Francisco Bay Trail sign.",
      "relation": "none",
      "retrieve": "top1",
      "evidence_window": "around"
    },
    {
      "id": "q2",
      "role": "target",
      "text": "The wearer sees a person walking an animal on the path.",
      "relation": "after:q1",
      "retrieve": "top5",
      "evidence_window": "around"
    },
    {
      "id": "q3",
      "role": "target",
      "text": "The wearer encounters a building shortly after seeing the person.",
      "relation": "after:q2",
      "retrieve": "top5",
      "evidence_window": "around"
    }
  ],
  "global_requirements": ["order_proposals", "deduplicate_proposals"]
}

EXAMPLE 3 — COMPARISON REQUIRING REASONING
Question:
"Why did I add dish soap to the pot when washing the second vegetable but not
the first?"

Output:
{
  "question_type": "comparison",
  "groundability": "evidence_for_inference",
  "unknown_answer_slots": ["reason for using dish soap only during the second washing"],
  "queries": [
    {
      "id": "q1",
      "role": "context",
      "text": "The wearer washes a vegetable in a pot without adding dish soap.",
      "relation": "first_occurrence",
      "retrieve": "top5",
      "evidence_window": "around"
    },
    {
      "id": "q2",
      "role": "evidence_source",
      "text": "The wearer washes another vegetable and adds dish soap to the pot.",
      "relation": "second_occurrence",
      "retrieve": "top5",
      "evidence_window": "around"
    }
  ],
  "global_requirements": ["order_proposals", "compare_evidence", "reason_over_evidence"]
}

EXAMPLE 4 — OCR ACROSS TWO MOMENTS
Question:
"Earlier I saw a plaque for the Bird Sanctuary, and later I read an
informational sign about it. What information from the plaque was not included
on the sign?"

Output:
{
  "question_type": "ocr",
  "groundability": "evidence_for_inference",
  "unknown_answer_slots": ["plaque information absent from the later sign"],
  "queries": [
    {
      "id": "q1",
      "role": "evidence_source",
      "text": "The wearer reads the Bird Sanctuary plaque.",
      "relation": "none",
      "retrieve": "top5",
      "evidence_window": "inside"
    },
    {
      "id": "q2",
      "role": "evidence_source",
      "text": "The wearer later reads an informational sign about the Bird Sanctuary.",
      "relation": "after:q1",
      "retrieve": "top5",
      "evidence_window": "inside"
    }
  ],
  "global_requirements": ["order_proposals", "read_text", "compare_evidence", "reason_over_evidence"]
}

EXAMPLE 5 — COUNTING REPEATED EVENTS
Question:
"After passing the sign stating the trail is 3 miles long, I walked through a
culvert and encountered water. How many water crossings did I make after the
culvert?"

Output:
{
  "question_type": "counting",
  "groundability": "global_verification",
  "unknown_answer_slots": ["number of distinct water crossings"],
  "queries": [
    {
      "id": "q1",
      "role": "anchor",
      "text": "The wearer walks through a culvert on the trail.",
      "relation": "none",
      "retrieve": "top1",
      "evidence_window": "around"
    },
    {
      "id": "q2",
      "role": "repeated_event",
      "text": "The wearer crosses water on the trail.",
      "relation": "after:q1",
      "retrieve": "top5",
      "evidence_window": "around"
    }
  ],
  "global_requirements": ["order_proposals", "deduplicate_proposals", "count_distinct_occurrences"]
}

EXAMPLE 6 — ABSENCE CANNOT BE LOCALIZED DIRECTLY
Question:
"After the person in the Harry Potter robe entered the gym, did I see them
again, and if not, what other costumed character did I see wearing a robe?"

Output:
{
  "question_type": "absence",
  "groundability": "global_verification",
  "unknown_answer_slots": ["whether the person reappeared", "other robed character identity"],
  "queries": [
    {
      "id": "q1",
      "role": "repeated_event",
      "text": "The wearer sees the person in the Harry Potter robe.",
      "relation": "none",
      "retrieve": "top5",
      "evidence_window": "around"
    },
    {
      "id": "q2",
      "role": "target",
      "text": "The wearer sees another costumed character wearing a robe.",
      "relation": "after:q1",
      "retrieve": "top5",
      "evidence_window": "around"
    }
  ],
  "global_requirements": ["order_proposals", "deduplicate_proposals", "verify_absence_globally"]
}

EXAMPLE 7 — DERIVED ARITHMETIC FROM OCR EVIDENCE
Question:
"Based on the plaques I read, how many years apart were the Kościuszko
monument and the Bell sculpture created?"

Output:
{
  "question_type": "quantitative_comparison",
  "groundability": "evidence_for_inference",
  "unknown_answer_slots": ["Kościuszko monument year", "Bell sculpture year", "difference in years"],
  "queries": [
    {
      "id": "q1",
      "role": "evidence_source",
      "text": "The wearer reads the plaque for the Kościuszko monument.",
      "relation": "none",
      "retrieve": "top5",
      "evidence_window": "inside"
    },
    {
      "id": "q2",
      "role": "evidence_source",
      "text": "The wearer reads the plaque for the Bell sculpture.",
      "relation": "none",
      "retrieve": "top5",
      "evidence_window": "inside"
    }
  ],
  "global_requirements": ["read_text", "compare_evidence", "calculate_from_evidence"]
}

EXAMPLE 8 — OBJECT TRACKING
Question:
"What item that I saw on a pallet of pavers earlier ended up in the skid steer
cab later?"

Output:
{
  "question_type": "object_tracking",
  "groundability": "multi_event",
  "unknown_answer_slots": ["tracked item identity"],
  "queries": [
    {
      "id": "q1",
      "role": "context",
      "text": "The wearer sees an item on a pallet of pavers.",
      "relation": "none",
      "retrieve": "top5",
      "evidence_window": "around"
    },
    {
      "id": "q2",
      "role": "target",
      "text": "The wearer later sees the same item inside the skid steer cab.",
      "relation": "same_entity_as:q1",
      "retrieve": "top5",
      "evidence_window": "around"
    }
  ],
  "global_requirements": ["order_proposals", "compare_evidence", "track_entity"]
}

FINAL VALIDATION BEFORE OUTPUT
- Every query is visually observable.
- Every query is entailed by QUESTION.
- No unknown slot has been resolved or guessed.
- Distant moments are separate queries.
- Repeated/ordinal/count/absence questions request top5 and global processing.
- OCR and inference requirements are declared.
- Derived quantities retain their source observations and request calculation.
- Tracked objects use separate states without guessing their identity.

Now convert this question:

{{QUESTION}}
```
