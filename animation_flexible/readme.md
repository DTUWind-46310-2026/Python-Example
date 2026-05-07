# Example and Notes

Simulation Animation

https://github.com/user-attachments/assets/346a8f1d-2edf-4432-a186-81a05b1298b1

Mode Animation

https://github.com/user-attachments/assets/885a1afe-124c-4720-a88d-866b8a58e990


Provided functionalities are:

- `animate(...)` to animate the wind turbine response of a simulation
- `animate_modes(...)` to animate the modes of the wind turbine system

To use the animation functionality, you need

- animate_structure.py
- rotation.py
- data/blade_mode_shape.csv
- data/blade_data.csv

from this directory. Then, you may use `from animate_structure import animate, animate_modes`. To learn how to use `animate()` and `animate_modes`, look at its docstring and `tests.py` for `animate()` (for which the outputs are in data/).

You might need to install FFmpeg yourself. [https://matplotlib.org/stable/users/explain/animations/animations.html](https://matplotlib.org/stable/users/explain/animations/animations.html) gives an example at the bottom of the page.
