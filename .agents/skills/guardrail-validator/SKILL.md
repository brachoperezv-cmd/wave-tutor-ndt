---
name: guardrail-validator
description: Instructions for checking solver results against expected physical ranges and classifying errors.
---

# Guardrail Validator Skill

This skill outlines the rules used by the Guardrail Agent to grade the Solver's calculated sound speed values.

## 1. Safety & Naming Rules
*   Longitudinal sound speed ($V_L$) must always be strictly greater than Shear sound speed ($V_S$):
    $$V_L > V_S$$
*   Calculated speeds must be within the expected physical ranges of the selected metal in the database.

---

## 2. Error Categorization
If a calculation fails the rules, log one of these errors:

### Category 2 Error (Impossibly Fast / t = 0)
*   **Trigger**: The calculated sound speed is extremely high (typically $> 10,000\text{ m/s}$) or the travel time is $< 1\text{ \mu s}$.
*   **Cause**: The solver selected the electrical noise/initial bang at $t=0$.
*   **Instruction**: Reject the solution. Tell the solver to ignore the $t=0$ peak and try the next peak cluster.

### Category 1 Error (Mode Leakage Overlap)
*   **Trigger**: The calculated shear sound speed is too close to the longitudinal sound speed (often matching the longitudinal speed of that metal).
*   **Cause**: The solver selected the early longitudinal wave leakage in the shear trace.
*   **Instruction**: Reject the solution. Tell the solver to look for a later peak cluster.
