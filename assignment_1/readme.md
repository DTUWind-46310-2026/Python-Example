# Assignment 1

Extended and altered code based on the code from `exercise_1`.

## Comparison to Exercise 1

| Concept | Exercise 1 | Assignment 1 |
| --- | --- | --- |
| Aerodynamics | None | BEM with optional dynamic models |
| Structure | RigidStructure | + RigidPitchingStructure |
| Wind | Constant, shear, tower shadow | + Turbulent wind (Mann box) |
| Recorders | Position / velocity | + Power, thrust, induction, blade loads |
| Profiling | None | `timing.py` measures per-function performance |

## Running the code

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all cases:

```bash
python assignment_1.py
```

The turbulence case (case 4) requires a Mann turbulence box at `data/mann_box.nc`. If this file is missing, generate it first using the script in `turbulence_creation_py/turbulence.py`.
