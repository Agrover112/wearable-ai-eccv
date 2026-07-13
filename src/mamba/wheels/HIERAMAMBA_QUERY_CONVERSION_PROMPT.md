# HieraMamba Query-Conversion Prompt

Use this prompt to convert an EgoLongQA question into one or more declarative
events that HieraMamba can localize. Review conversions before inference.

```text
You convert an egocentric-video QA question into declarative event
descriptions for a temporal-grounding model.

The grounding model does NOT answer questions. It locates when a visually
described event occurs and returns start/end timestamps.

TASK
Given one question, produce 1–4 short declarative grounding queries covering
the video moments needed to answer it.

RULES
1. Use only information explicitly stated in the question.
2. Never use multiple-choice options, the correct answer, or a reference answer.
3. Never guess an unknown object, person, place, text, color, number, reason,
   price, or other answer.
4. Convert first-person language to third-person:
   - "I" -> "the wearer"
   - "my" -> "the wearer's"
5. Describe observable events, actions, states, signs, or interactions.
6. Remove interrogative wording such as "what", "which", "where", "why",
   "how many", and "did I".
7. Preserve visible identifying details already stated in the question.
8. If the question relates multiple moments, emit separate queries for the
   anchor and target events.
9. For FIRST, LAST, SECOND, BEFORE, AFTER, or LATER questions, preserve that
   information in the query role and relation fields, without inventing answers.
10. If an answer-bearing detail is unknown, describe the containing event
    generically:
    - unknown object -> "an object"
    - unknown animal -> "an animal"
    - unknown text -> "a sign" or "text on a sign"
    - unknown setting -> "a touchscreen setting"
11. For repeated events, use a query that can retrieve multiple occurrences.
12. Each query must be a natural declarative sentence of 5–20 words.
13. Produce valid JSON only.

OUTPUT SCHEMA
{
  "question_type": "single_event | before_after | repeated_event | comparison | state_change | causal_sequence | multi_event",
  "queries": [
    {
      "id": "q1",
      "role": "anchor | target | repeated_event | context",
      "text": "Natural declarative event description.",
      "relation": "none | before:qN | after:qN | first | last | second"
    }
  ]
}

EXAMPLE 1
Question:
"I moved an object from the floor to the kitchen island. What was the object
and where was it first located?"

Output:
{
  "question_type": "single_event",
  "queries": [
    {
      "id": "q1",
      "role": "target",
      "text": "The wearer moves an object from the floor to the kitchen island.",
      "relation": "none"
    }
  ]
}

EXAMPLE 2
Question:
"After I passed the San Francisco Bay Trail sign, I later saw a person walking
an animal. What was the animal, and what building did I encounter shortly
after that?"

Output:
{
  "question_type": "before_after",
  "queries": [
    {
      "id": "q1",
      "role": "anchor",
      "text": "The wearer passes the San Francisco Bay Trail sign.",
      "relation": "none"
    },
    {
      "id": "q2",
      "role": "target",
      "text": "The wearer sees a person walking an animal on the path.",
      "relation": "after:q1"
    },
    {
      "id": "q3",
      "role": "target",
      "text": "The wearer encounters a building on the path.",
      "relation": "after:q2"
    }
  ]
}

EXAMPLE 3
Question:
"Why did I add dish soap to the pot when washing the second vegetable but not
the first?"

Output:
{
  "question_type": "comparison",
  "queries": [
    {
      "id": "q1",
      "role": "context",
      "text": "The wearer washes the first vegetable in a pot.",
      "relation": "first"
    },
    {
      "id": "q2",
      "role": "target",
      "text": "The wearer washes the second vegetable and adds dish soap.",
      "relation": "second"
    }
  ]
}

Now convert this question:

{{QUESTION}}
```
