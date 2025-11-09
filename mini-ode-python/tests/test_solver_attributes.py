import mini_ode

attributes = [
    "__class__",
    "__delattr__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__getstate__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__module__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "solve",
]

def test_euler_dir_attribute():
    solver = mini_ode.EulerMethodSolver(step=0.1)
    assert solver.__dir__() == attributes+["step"]

def test_rk4_dir_attribute():
    solver = mini_ode.RK4MethodSolver(step=0.1)
    assert solver.__dir__() == attributes+["step"]

def test_implicit_euler_dir_attribute():
    solver = mini_ode.ImplicitEulerMethodSolver(step=0.1, optimizer=mini_ode.optimizers.Newton(max_steps=10))
    assert solver.__dir__() == attributes+["step", "optimizer"]

def test_glrk4_dir_attribute():
    solver = mini_ode.GLRK4MethodSolver(step=0.1, optimizer=mini_ode.optimizers.Newton(max_steps=10))
    assert solver.__dir__() == attributes+["step", "optimizer"]

def test_rkf45_dir_attribute():
    solver = mini_ode.RKF45MethodSolver(rtol=1., atol=1., min_step=1., safety_factor=1.)
    assert solver.__dir__() == attributes+["rtol", "atol", "min_step", "safety_factor"]

def test_row1_dir_attribute():
    solver = mini_ode.ROW1MethodSolver(step=0.1)
    assert solver.__dir__() == attributes+["step"]
