---
name: sound-speed-calculator
description: Guides the agent on how to calculate sound speed and elastic moduli from signal delays and sample parameters.
---

# Sound Speed Calculator Skill

This skill handles the mathematical equations and unit conversions required to calculate ultrasonic velocities and material elastic constants.

## 1. Velocity Formulas
To calculate sound speed ($V$):
$$V = \frac{d}{\Delta t}$$

*   **Thickness ($d$)**: Must be converted from millimeters ($\text{mm}$) to meters ($\text{m}$):
    $$d_{\text{meters}} = \frac{d_{\text{mm}}}{1000}$$
*   **Time Delay ($\Delta t$)**: Must be converted from microseconds ($\mu\text{s}$) to seconds ($\text{s}$):
    $$\Delta t_{\text{seconds}} = \Delta t_{\mu\text{s}} \times 10^{-6}$$
*   **Velocity ($V$)**: Output speed in meters per second ($\text{m/s}$).

---

## 2. Elastic Properties Formulas
To calculate elastic properties, the material's bulk density ($\rho$) in $\text{kg/m}^3$ must be known.

### A. Shear Modulus ($G$)
$$G = \rho \times V_S^2$$
*   Divide the result by $10^9$ to output in Gigapascals ($\text{GPa}$).

### B. Poisson's Ratio ($\nu$)
$$\nu = \frac{V_L^2 - 2V_S^2}{2(V_L^2 - V_S^2)}$$
*   Output is a dimensionless ratio.

### C. Young's Modulus ($E$)
$$E = 2G \times (1 + \nu)$$
*   Output is in Gigapascals ($\text{GPa}$).
