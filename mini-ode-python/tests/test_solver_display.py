import mini_ode

def test_solver_display_euler():
    solver = mini_ode.EulerMethodSolver(step=0.1)
    assert str(solver) == "EulerMethodSolver(step=0.1)"

def test_solver_display_rk4():
    solver = mini_ode.RK4MethodSolver(step=0.1)
    assert str(solver) == "RK4MethodSolver(step=0.1)"

def test_solver_display_implicit_euler():
    solver = mini_ode.ImplicitEulerMethodSolver(step=0.1, optimizer=mini_ode.optimizers.Newton(max_steps=10))
    assert str(solver) == "ImplicitEulerMethodSolver(step=0.1, optimizer=Newton(max_steps=10))"

def test_solver_display_glrk4():
    solver = mini_ode.GLRK4MethodSolver(step=0.1, optimizer=mini_ode.optimizers.Newton(max_steps=10))
    assert str(solver) == "GLRK4MethodSolver(step=0.1, optimizer=Newton(max_steps=10))"

def test_solver_display_rkf45():
    solver = mini_ode.RKF45MethodSolver(rtol=0.001, atol=0.001, min_step=0.00001, safety_factor=0.9)
    assert str(solver) == "RKF45MethodSolver(rtol=0.001, atol=0.001, min_step=0.00001, safety_factor=0.9)"

def test_solver_display_row1():
    solver = mini_ode.ROW1MethodSolver(step=0.1)
    assert str(solver) == "ROW1MethodSolver(step=0.1)"
