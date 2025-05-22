# A minimalistic ODE solvers library built on top of PyTorch

## Supported methods

| **Solver**            | **Method**                                     | **Purpose / Use Case**                                       | **Implicit** | **Adaptive step** |
|------------------------|-----------------------------------------------|---------------------------------------------------------------|--------------|-------------------|
| `EulerMethodSolver`          | Euler method                                  | Simple and fast, good for teaching and very basic problems.   | `no`         | `no`              |
| `RK4MethodSolver`            | Fourth Order Runge-Kutta (RK4)                | Widely used general-purpose solver with fixed step size.      | `no`         | `no`              |
| `ImplicitEulerMethodSolver` | Implicit Euler method                         | Suitable for stiff or ill-conditioned problems.               | `yes`        | `no`              |
| `GLRK4MethodSolver`          | Gauss-Legendre Runge-Kutta (Order 4)          | High-precision solver for stiff or ill-conditioned problems.  | `yes`        | `no`              |
| `RKF45MethodSolver`          | Runge-Kutta-Fehlberg 4(5)                     | Adaptive step size, balances efficiency and accuracy.         | `no`         | `yes`             |
| `ROW1MethodSolver`           | First Order Rosenbrock-Wanner (ROW1) Method   | Efficient alternative to Implicit Euler for stiff systems.    | `semi`       | `no`              |

## Examples

Notebook `example.ipynb` has few examples of usage of `mini-ode` solvers.

## Building Rust crate

In order to build Rust crate go to `mini-ode/` directory and run command:

```bach
cargo build
```

## Building Python package

Python package can only be built in conda environment or in python virtual environment. In order to build the project activate conda environment or python virtual environment. Then, go to `mini-ode-python/` directory and run command:
```bash
LIBTORCH_USE_PYTORCH=1 maturin develop
```

Maturin will build the python package and install it in local environment.
