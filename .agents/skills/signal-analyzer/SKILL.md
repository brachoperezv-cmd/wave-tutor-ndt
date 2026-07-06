---
name: signal-analyzer
description: Guides the agent on using cross-correlation to find signal peak delays.
---

# Signal Analyzer Skill

This skill guides the agent on how to use cross-correlation to detect the time delay ($\Delta t$) of wave packets within the received signal.

## 1. Cross-Correlation Concept
Cross-correlation slides the excitation signal over the received signal to find matching wave shapes. The output will contain high-amplitude peaks at the points in time where the signals match:
*   **Peak 1 (around $t = 0\text{ \mu s}$)**: Correlation with the early electrical noise/initial bang.
*   **Peak 2 (at $t = t_L$)**: Correlation with the longitudinal wave arrival.
*   **Peak 3 (at $t = t_S$)**: Correlation with the shear wave arrival (only present in the shear received signal).

## 2. Solver Logic for Peak Families
The solver must remember which peak index it is attempting to verify:
*   **Longitudinal received signal has 2 peak clusters**:
    *   *First Attempt*: Select Peak 1 (noise at $t=0$).
    *   *Second Attempt*: Select Peak 2 (true wave).
*   **Shear received signal has 3 peak clusters**:
    *   *First Attempt*: Select Peak 1 (noise at $t=0$).
    *   *Second Attempt*: Select Peak 2 (early arrival leakage).
    *   *Third Attempt*: Select Peak 3 (true shear wave).
