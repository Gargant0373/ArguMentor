# Argument Annotation Grading Rubric

This document defines the grading scheme and guidelines for annotating argument suggestions and feedback, as established during the calibration meeting. Feedback is evaluated across three dimensions: **Relevance**, **Actionability**, and **Clarity**, each on a scale from 1 to 5.

---

## 1. Core Metrics & Scoring Guidelines

### Relevance
Measures how applicable and useful the suggestion/feedback is to the given argument and topic.

* **5 (High):** The feedback is highly relevant (adresses the core of the argument), or, in the case of highly graded arguments, it addresses how to **extend the argument in a better way** (e.g., it addresses a potential improvement or angle that hasn't been mentioned in the present argument).
* **4 (Medium-High):** The feedback is relevant and directly addresses the core argument, pointing out a valid modification with a small degree of vagueness.
* **3 (Medium):** The feedback has moderate relevance or partial alignment with the argument's core focus.
* **2 (Low):** The feedback has low relevance to improving the argument's stance or topic. 
  * **EDGE CASE:** Use this score if the feedback is structured purely as a simple replacement (e.g., "just replace X with Y").
  * **EDGE CASE:** Use this score if the feedback is given for an argument that states the **opposite** of the intended stance.
* **1 (None):** The feedback is completely irrelevant, off-topic, or fails to address any aspect of the argument or stance.

### Actionability
Measures how concrete and easy-to-implement the suggestion is for the writer. 

* **5 (High):** The suggestion is completely actionable and **gives a concrete example** of how to implement the change.
  * *Note:* Even if the underlying argument states the opposite stance, if a concrete example is provided for it, it should still be scored a 5. Do not penalize actionability for opposite stance errors.
* **4 (Medium-High):** The suggestion **gives a way/direction, but not the exact final solution**.
  * **EDGE CASE:** Use this score if the feedback is structured purely as a simple replacement (e.g., "just replace X with Y"), as it tells the writer exactly what to do but doesn't provide a deeper solution or broader context.
* **3 (Medium):** The suggestion provides general guidance but lacks a clear pathway, concrete text replacements, or an explicitly outlined method for execution.
* **2 (Low):** The suggestion identifies a problem but gives very little or highly ambiguous direction on how to fix it.
* **1 (None):** The feedback provides absolutely no suggestion, direction, or pathway for improvement (e.g., simply stating "this is bad" without further instruction).

### Clarity
Measures how clear, understandable, and well-phrased the feedback is.

* **5 (High):** The feedback is direct, crystal clear, easy to understand, and perfectly unambiguous.
  * **EDGE CASE:** Simple "just replace" text suggestions often qualify for a 5 if they are completely straightforward.
* **4 (Medium-High):** The feedback is clear and easy to understand, with only minor phrasing stiffness or trivial ambiguities.
  * **EDGE CASE:** Simple "just replace" text suggestions can also fall here if easily understood.
* **3 (Medium):** The feedback is understandable but requires re-reading due to slightly awkward phrasing or minor structural issues.
* **2 (Low):** The feedback is highly confusing, vague, or poorly structured, hindering immediate comprehension.
* **1 (Extremely Poor):** The feedback is completely incomprehensible, broken by severe grammatical errors, formatting malfunctions, or other critical appearance issues.

---

## 2. Quick-Reference Rules for Edge Cases

### A. Opposite Stance Arguments
When the argument text being evaluated states the **opposite** of the intended stance:
1. **Relevance:** Automatically drop to **2 (Low)**.
2. **Actionability & Clarity:** Grade these independently based on their own merits. Do **not** penalize Actionability or Clarity because the suggestion is correcting an argument for the opposite stance. Grade them as if the stance alignment were not an issue.

### B. "Just Replace" Suggestions
When the feedback is structured purely as a simple replacement (e.g., "just replace X with Y"):
* **Relevance:** **2**
* **Actionability:** **4-5** (tells you exactly what to do, but lacks a deeper solution or broad concrete example)
* **Clarity:** **4–5** (depending on how cleanly phrased the substitution is)

---

*Note: Use the examples in `calibration.csv` as a benchmark for baseline argument structures and topics.*