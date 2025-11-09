import mini_ode

def test_solver_display_euler():
    solver = mini_ode.EulerMethodSolver(step=0.1)
    assert solver.__str__() == "EulerMethodSolver(step=0.1)"
    assert solver.__repr__() == "EulerMethodSolver(step=0.1)"

def test_solver_display_rk4():
    solver = mini_ode.RK4MethodSolver(step=0.1)
    assert solver.__str__() == "RK4MethodSolver(step=0.1)"
    assert solver.__repr__() == "RK4MethodSolver(step=0.1)"

def test_solver_display_implicit_euler():
    solver = mini_ode.ImplicitEulerMethodSolver(step=0.1, optimizer=mini_ode.optimizers.Newton(max_steps=10))
    assert solver.__str__() == "ImplicitEulerMethodSolver(step=0.1, optimizer=Newton(max_steps=10))"
    assert solver.__repr__() == "ImplicitEulerMethodSolver(step=0.1, optimizer=Newton(max_steps=10))"

def test_solver_display_glrk4():
    solver = mini_ode.GLRK4MethodSolver(step=0.1, optimizer=mini_ode.optimizers.Newton(max_steps=10))
    assert solver.__str__() == "GLRK4MethodSolver(step=0.1, optimizer=Newton(max_steps=10))"
    assert solver.__repr__() == "GLRK4MethodSolver(step=0.1, optimizer=Newton(max_steps=10))"

def test_solver_display_rkf45():
    solver = mini_ode.RKF45MethodSolver(rtol=0.001, atol=0.001, min_step=0.00001, safety_factor=0.9)
    assert solver.__str__() == "RKF45MethodSolver(rtol=0.001, atol=0.001, min_step=0.00001, safety_factor=0.9)"
    assert solver.__repr__() == "RKF45MethodSolver(rtol=0.001, atol=0.001, min_step=0.00001, safety_factor=0.9)"

def test_solver_display_row1():
    solver = mini_ode.ROW1MethodSolver(step=0.1)
    assert solver.__str__() == "ROW1MethodSolver(step=0.1)"
    assert solver.__repr__() == "ROW1MethodSolver(step=0.1)"
