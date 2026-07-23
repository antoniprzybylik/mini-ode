import pytest
import mini_ode

@pytest.mark.parametrize(
    "solver, expected",
    [
        (mini_ode.EulerMethodSolver(step=0.1), 0.5),
        (mini_ode.RK4MethodSolver(step=0.1), 0.6067708333333334),
        (
            mini_ode.ImplicitEulerMethodSolver(
                step=0.1,
                optimizer=mini_ode.optimizers.Newton(max_steps=10),
            ),
            2.0 / 3.0,
        ),
        (
            mini_ode.GLRK4MethodSolver(
                step=0.1,
                optimizer=mini_ode.optimizers.Newton(max_steps=10),
            ),
            0.6065573770491804,
        ),
        (
            mini_ode.RKF45MethodSolver(
                rtol=0.001,
                atol=0.001,
                min_step=0.00001,
                safety_factor=0.9,
            ),
            0.6065179286858975,
        ),
        (mini_ode.ROW1MethodSolver(step=0.1), 2.0 / 3.0),
    ],
)
def test_stability_function(solver, expected):
    assert solver.stability_function(-0.5) == pytest.approx(expected)


def test_stability_function_rejects_positive_values():
    solver = mini_ode.EulerMethodSolver(step=0.1)

    with pytest.raises(RuntimeError, match="Stability function is not defined for positive numbers"):
        solver.stability_function(1.0)
