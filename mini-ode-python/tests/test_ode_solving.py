import torch
import mini_ode


def test_euler_solver_exp_1d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0.])))

    solver = mini_ode.EulerMethodSolver(step=0.01)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,1)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1))).abs() <= 0.05).all()


def test_euler_solver_exp_2d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))

    solver = mini_ode.EulerMethodSolver(step=0.01)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1., 1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,2)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1).repeat([1, 2]))).abs() <= 0.05).all()


def test_rk4_solver_exp_1d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0.])))

    solver = mini_ode.RK4MethodSolver(step=0.01)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,1)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1))).abs() <= 0.05).all()


def test_rk4_solver_exp_2d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))

    solver = mini_ode.RK4MethodSolver(step=0.01)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1., 1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,2)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1).repeat([1, 2]))).abs() <= 0.05).all()


def test_implicit_euler_solver_exp_1d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0.])))

    solver = mini_ode.ImplicitEulerMethodSolver(step=0.01, optimizer=mini_ode.optimizers.Newton(max_steps=3))
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,1)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1))).abs() <= 0.05).all()


def test_implicit_euler_solver_exp_2d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))

    solver = mini_ode.ImplicitEulerMethodSolver(step=0.01, optimizer=mini_ode.optimizers.Newton(max_steps=3))
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1., 1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,2)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1).repeat([1, 2]))).abs() <= 0.05).all()


def test_glrk4_solver_exp_1d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0.])))

    solver = mini_ode.GLRK4MethodSolver(step=0.01, optimizer=mini_ode.optimizers.Newton(max_steps=3))
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,1)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1))).abs() <= 0.05).all()


def test_glrk4_solver_exp_2d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))

    solver = mini_ode.GLRK4MethodSolver(step=0.01, optimizer=mini_ode.optimizers.Newton(max_steps=3))
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1., 1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,2)
    assert ((ys - torch.exp(torch.linspace(0., 1., 101).unsqueeze(1).repeat([1, 2]))).abs() <= 0.05).all()


def test_rkf45_solver_exp_1d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0.])))

    solver = mini_ode.RKF45MethodSolver(0.00001, 0.00001, 1e-9, 0.9)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1.]))

    assert len(xs.shape) == 1
    assert ys.shape == (xs.shape[0],1)
    assert ((ys - torch.exp(xs.unsqueeze(1))).abs() <= 0.00003).all()


def test_rkf45_solver_exp_2d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))

    solver = mini_ode.RKF45MethodSolver(0.00001, 0.00001, 1e-9, 0.9)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1., 1.]))

    assert len(xs.shape) == 1
    assert ys.shape == (xs.shape[0],2)
    assert ((ys - torch.exp(xs.unsqueeze(1).repeat([1, 2]))).abs() <= 0.00003).all()


def test_row1_solver_exp_1d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0.])))

    solver = mini_ode.ROW1MethodSolver(step=0.01)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,1)
    assert ((ys - torch.exp(xs.unsqueeze(1))).abs() <= 0.05).all()


def test_row1_solver_exp_2d():
    def f(x: torch.Tensor, y: torch.Tensor):
        return y
    f_traced = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([0., 0.])))

    solver = mini_ode.ROW1MethodSolver(step=0.01)
    xs, ys = solver.solve(f_traced, (0., 1.), torch.tensor([1., 1.]))

    assert xs.shape == (101,)
    assert ys.shape == (101,2)
    assert ((ys - torch.exp(xs.unsqueeze(1).repeat([1, 2]))).abs() <= 0.05).all()
