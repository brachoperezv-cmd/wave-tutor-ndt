---
name: socratic-tutor
description: Guides the Teaching Agent on Socratic explanation structure and density measurement topics.
---

# Socratic Tutor Skill

This skill outlines how the Teaching Agent should explain the results to the student.

## 1. Socratic Teaching Rules
*   Do not just give the final values. Review the journey of the Solver Agent.
*   First, explain how the Solver got tripped up by the early transducer noise/initial bang at $t=0$ (explain this using the concepts in `references/initial_bang.txt`).
*   Second, explain how the Guardrail detected this as a **Category 2 Error** because the calculated velocity was physically impossible for any metal.
*   Third, show how the Solver found the correct wave packet using **cross-correlation** (explain this using `references/cross_correlation.txt`).

---

## 2. Density & Elastic Properties Explanation
Once the correct answer is explained:
*   State that these acoustic velocities ($V_L$ and $V_S$) can be used to calculate Young's Modulus ($E$), Shear Modulus ($G$), and Poisson's Ratio ($\nu$).
*   Explain that the **bulk density ($\rho$)** of the metal must be known to run these calculations.
*   Review laboratory methods to measure density, explicitly highlighting:
    1.  *Direct Geometric Measurement* (Mass / Volume).
    2.  *Hydrostatic Weighing (Archimedes' Principle)* (explain this as a highly accurate, easy-to-do alternative for irregular shapes, using `references/density_measurement.txt`).
