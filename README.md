# mini-ode

A minimalistic, multi-language library for solving Ordinary Differential Equations (ODEs). `mini-ode` is designed with a shared Rust core and a consistent interface for both **Rust** and **Python** users. It supports explicit, implicit, fixed step and adaptive step algorithms.

[![License: GPL-2.0](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)

## ✨ Features

- **Dual interface**: call the same solvers from Rust or Python
- **PyTorch-compatible**: define the derivative function using PyTorch
- **Multiple solver methods**: includes explicit, implicit, and adaptive-step solvers
- **Modular optimizers**: implicit solvers allow flexible optimizer configuration

## 🧠 Supported Solvers

| Solver Class               | Method                                 | Suitable For                                 | Implicit | Adaptive Step |
|----------------------------|----------------------------------------|----------------------------------------------|----------|----------------|
| `EulerMethodSolver`        | Euler                                  | Simple, fast, and educational use.           | ❌       | ❌             |
| `RK4MethodSolver`          | Runge-Kutta 4th Order (RK4)            | General-purpose with fixed step size.        | ❌       | ❌             |
| `ImplicitEulerMethodSolver`| Implicit Euler                         | Stiff or ill-conditioned problems.           | ✅       | ❌             |
| `GLRK4MethodSolver`        | Gauss-Legendre RK (Order 4)            | High-accuracy, stiff problems.               | ✅       | ❌             |
| `RKF45MethodSolver`        | Runge-Kutta-Fehlberg 4(5)              | Adaptive step size control.                  | ❌       | ✅             |
| `ROW1MethodSolver`         | Rosenbrock-Wanner (Order 1)            | Fast semi-implicit method for stiff systems. | semi  | ❌             |

## 📦 Building the Library

### Rust

To build the core Rust library:

```bash
cd mini-ode
cargo build --release
```

### Python

To build and install the Python package (in a virtual environment or Conda environment):

```bash
cd mini-ode-python
LIBTORCH_USE_PYTORCH=1 maturin develop
```

> This builds the Python bindings using [`maturin`](https://github.com/PyO3/maturin) and installs the package locally.

## 🐍 Python Usage Overview

To use `mini-ode` from Python:

1. Define the derivative function using `torch.Tensor` inputs.
2. **Trace** the function using `torch.jit.trace`.
3. Pass the traced function and initial conditions to a solver instance.
4. For implicit solvers, pass an optimizer at construction.

Example usage flow (not full code):

```python
import torch
import mini_ode

# 1. Define derivative function using PyTorch
def f(x: torch.Tensor, y: torch.Tensor):
    return y.flip(0) - torch.tensor([0, 1]) * (y.flip(0) ** 3)

# 2. Trace the function to TorchScript
traced_f = torch.jit.trace(f, (torch.tensor(0.), torch.tensor([[0., 0.]])))

# 3. Create a solver instance
solver = mini_ode.RK4MethodSolver(step=0.01)

# 4. Solve the ODE
xs, ys = solver.solve(traced_f, torch.tensor([0., 5.]), torch.tensor([1.0, 0.0]))
```

### 🔧 Using Optimizers (Implicit Solvers Only)

Some solvers like `GLRK4MethodSolver` or `ImplicitEulerMethodSolver` require an optimizer for nonlinear system solving:

```python
optimizer = mini_ode.optimizers.CG(
    max_steps=5,
    gtol=1e-8,
    linesearch_atol=1e-6
)

solver = mini_ode.GLRK4MethodSolver(step=0.2, optimizer=optimizer)
```

## 🦀 Rust Usage Overview

In Rust, solvers use the same logic as in Python - but you pass in a `tch::CModule` representing the TorchScripted derivative function.

```rust
use mini_ode::Solver;
use tch::{Tensor, CModule};

fn main() -> anyhow::Result<()> {
    let solver = Solver::Euler { step: 0.01 };
    let model = CModule::load("my_traced_function.pt")?;
    let x_span = Tensor::from_slice(&[0.0f64, 2.0]);
    let y0 = Tensor::from_slice(&[1.0f64, 0.0]);

    let (xs, ys) = solver.solve(model, x_span, y0)?;
    println!("{:?}", xs);
    Ok(())
}
```

## 📁 Project Structure

```
mini-ode/           # Core Rust implementation of solvers
mini-ode-python/    # Python bindings using PyO3 + maturin
example.ipynb       # Jupyter notebook demonstrating usage
```

## 📄 License

This project is licensed under the [GPL-2.0 License](LICENSE).

## 👤 Author

**Antoni Przybylik**  
📧 [antoni.przybylik@wp.pl](mailto:antoni.przybylik@wp.pl)  
🔗 [https://github.com/antoniprzybylik](https://github.com/antoniprzybylik)
