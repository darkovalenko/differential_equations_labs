# Differential Equations: Computer Practicum

Laboratory works (computer practicums) for the "Differential Equations" course, 2nd year, specialty E7 Mathematics (AI), Taras Shevchenko National University of Kyiv. The code is written in **Python** and uses symbolic and numerical methods to analyze ODEs and systems of ODEs. Each lab folder contains the scripts and a PDF report.

## Technologies

* **SymPy**: symbolic computation (derivatives, Jacobian matrices, integration, solving equations).
* **NumPy**: numerical operations and grids.
* **SciPy**: numerical integration (`solve_ivp`) and root finding.
* **Matplotlib**: plots of direction fields, phase portraits and integral curves.

## Structure

### [Lab 1: Geometric Interpretation of a First-Order ODE](./Lab1)
* Direction fields, isoclines, monotonicity, extrema and inflection points; an autonomous equation and convergence of its solutions.
* Files: `t1.py`, `t2.py`, `differential_lab1.pdf`

### [Lab 3: Phase Portraits of Autonomous Systems on the Plane](./Lab3)
* A linear system (node, intersection and approach angles of trajectories) and a nonlinear system (equilibria, linearization, classification, vector field and phase trajectories).
* Files: `t01.py`, `t02.py`, `differential_lab3.pdf`

### [Lab 4: Cauchy Problem. Picard and Peano Theorems](./Lab4)
* Lipschitz constant, convergence interval, Picard successive approximations and error estimates; non-uniqueness of the solution and the integral funnel.
* Files: `t01.py`, `t02.py`, `differential_lab4.pdf`

## How to run

```bash
pip install numpy scipy matplotlib sympy
python Lab4/t01.py
```

Plot scripts open a window; add `--save` to write a PNG instead (where supported), e.g. `python Lab4/t02.py --save`.

Author: **Daryna Kovalenko**
