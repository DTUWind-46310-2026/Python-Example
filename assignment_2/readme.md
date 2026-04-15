# Assignment 2

To install the code, follow the instructions in the `readme.md` of `exercise_1` and adjust it for `assignment_2`. What `assignment_2.py` does and the main code changes compared to Assignment 1 are explained below.

## `assignemnt_2.py`

`py run assignment_2.py` will, in its current state, result in

1. Running an optimisation to find (TSR, $\theta$)$_{\text{opt}}$ for below rated conditions. These optimal values are then given to the aerodynamics and structural (for the pitch range) classes.
2. Run Task 1 by running a simulation with steps in wind speed from 4 to 25 m/s inflow. From each step, the pitch and $C_P$ are extracted.
3. Run Task 2 by running a turbulent wind simulation. The plot `controller_modes.pdf` shows a timeseries of when the pitch controller is active and when the generator torque controller switches between the optimal power tracking and constant power or torque.

## Comparison to Assignment 1

### `controller.py`

Contains classes

- `ControllerBase`: Defining the structure that all controller classes need to have.
- `PIController`: k-ω² below rated, constant-power or constant-torque above rated, and collective pitch PI
- `ConstantRotSpeedController`: Keeps the rotational speed of the rotor constant by always setting the generator torque to the aerodynamic torque. This class is used during an optimisation to find the optimal performance below rated.

### `aerodynamics.py`

The aerodynamics classes now hold information about the conditions for optimal aerodynamic behaviour, such as

- $C_{p,\text{max}}$
- $\text{TSR}_{\text{opt}}$
- $\omega_{\text{rated}}$
- $P_{\text{rated}}$
- $\theta_{\text{opt}}$

`AerodynamicsBase` is adjusted accordingly.

### `structure.py`

The structure classes now also contain information about the rotor inertia, the pitch limits, and the pitch actuator dynamics. Additionally, the rotational speed is calculated based on the difference in aerodynamic and generator torque.

### `wind.py`

Now contains class `WindSteps` that multiplies the wind speeds returned by a `base_wind: WindBase` with specified factors at specified times. This class is used for the optimisation and the first task.

### `simulation.py`

The simulation now receives a controller instance as well. The step order is `aerodynamics -> controller -> structure -> wind`. For `ConstantRotSpeedController` to work, the controller needs to be stepped after the aerodynamics but before the structure.
