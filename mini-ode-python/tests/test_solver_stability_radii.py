import math
import pytest
import mini_ode


@pytest.mark.parametrize(
    "solver, expected",
    [
        (mini_ode.EulerMethodSolver(step=0.1), 2.0),
        (mini_ode.RK4MethodSolver(step=0.1), 2.785293563),
        (
            mini_ode.ImplicitEulerMethodSolver(
                step=0.1,
                optimizer=mini_ode.optimizers.Newton(max_steps=10),
            ),
            math.inf,
        ),
        (
            mini_ode.GLRK4MethodSolver(
                step=0.1,
                optimizer=mini_ode.optimizers.Newton(max_steps=10),
            ),
            math.inf,
        ),
        (
            mini_ode.RKF45MethodSolver(
                rtol=0.001,
                atol=0.001,
                min_step=0.00001,
                safety_factor=0.9,
            ),
            3.677706621,
        ),
        (mini_ode.ROW1MethodSolver(step=0.1), math.inf),
    ],
)
def test_stability_radius(solver, expected):
    actual = solver.stability_radius

    if math.isinf(expected):
        assert math.isinf(actual)
    else:
        assert actual == pytest.approx(expected)
