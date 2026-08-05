//! # mini-ode
//!
//! **mini-ode** is a minimalistic library for solving Ordinary Differential Equations (ODEs).
//!
//! The library supports explicit, implicit, fixed-step, and adaptive-step algorithms. The library
//! uses libtorch through [`tch`] bindings.
//!
//! ## Quick Start
//!
//! To begin, you'll need the [`tch`] crate (Rust bindings for PyTorch):
//!
//! ```toml
//! [dependencies]
//! mini-ode = "0.1"
//! tch = "0.15"
//! ```
//!
//! ### Defining Your ODE
//!
//! The derivative function must be defined using PyTorch operations and converted to TorchScript.
//! As an example, consider the undamped Duffing oscillator `y0'' = y0 - y0^3`, rewritten as a
//! first-order system `y0' = y1`, `y1' = y0 - y0^3`:
//!
//! ```rust
//! use tch::{Tensor, CModule};
//!
//! // Define the derivative function f(x, y) -> dy/dx
//! // x: scalar tensor (shape ())
//! // y: 1D tensor (shape (n,)) where n is state dimension
//! let y0 = Tensor::from_slice(&[1f64, 0f64]);
//! let mut closure = |inputs: &[Tensor]| {
//!     let _x = &inputs[0];
//!     let y = &inputs[1];
//!     let y0 = y.get(0);
//!     let y1 = y.get(1);
//!
//!     let dy0 = y1;
//!     let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);
//!
//!     vec![Tensor::stack(&[dy0, dy1], 0)]
//! };
//!
//! // Trace the function
//! let model = CModule::create_by_tracing(
//!     "ode_fn",
//!     "forward",
//!     &[Tensor::from(0.0f64), y0.shallow_clone()],
//!     &mut closure,
//! )?;
//! # use std::error::Error;
//! # Ok::<(), Box<dyn Error>>(())
//! ```
//!
//! ### Solving an ODE
//!
//! Once you have your TorchScript model, solve the ODE:
//!
//! ```rust
//! use mini_ode::Solver;
//! use tch::Tensor;
//!
//!# use tch::CModule;
//!#
//!# // Define the derivative function f(x, y) -> dy/dx
//!# // x: scalar tensor (shape ())
//!# // y: 1D tensor (shape (n,)) where n is state dimension
//!# let y0 = Tensor::from_slice(&[1f64, 0f64]);
//!# let mut closure = |inputs: &[Tensor]| {
//!#     let _x = &inputs[0];
//!#     let y = &inputs[1];
//!#     let y0 = y.get(0);
//!#     let y1 = y.get(1);
//!#
//!#     let dy0 = y1;
//!#     let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);
//!#
//!#     vec![Tensor::stack(&[dy0, dy1], 0)]
//!# };
//!#
//!# // Trace the function
//!# let model = CModule::create_by_tracing(
//!#     "ode_fn",
//!#     "forward",
//!#     &[Tensor::from(0.0f64), y0.shallow_clone()],
//!#     &mut closure,
//!# )?;
//!
//! // Create a solver with fixed step size
//! let solver = Solver::RK4 { step: 0.01 };
//!
//! // Define integration interval and initial condition
//! let x_span = (0.0, 5.0);
//! let y0 = Tensor::from_slice(&[1.0f64, 0.0]);
//!
//! // Solve and get results
//! let (xs, ys) = solver.solve(model, x_span, y0)?;
//! // xs: 1D tensor of x-values, shape (num_points,)
//! // ys: 2D tensor of y-values, shape (num_points, n)
//! # use std::error::Error;
//! # Ok::<(), Box<dyn Error>>(())
//! ```
//!
//! For adaptive-step solvers, configure tolerances:
//!
//! ```rust
//! use mini_ode::Solver;
//!
//! let solver = Solver::RKF45 {
//!     rtol: 1e-5,
//!     atol: 1e-5,
//!     min_step: 1e-9,
//!     safety_factor: 0.9,
//! };
//! ```
//!
//! For implicit solvers ([`Solver::ImplicitEuler`], [`Solver::GLRK4`]), you need to configure an optimizer:
//!
//! ```rust
//! use mini_ode::Solver;
//! use mini_ode::optimizers;
//! use std::sync::Arc;
//!
//! let optimizer = optimizers::CG::new(5, None, Some(1e-8));
//!
//! let solver = Solver::GLRK4 {
//!     step: 0.2,
//!     optimizer: Arc::new(optimizer),
//! };
//! ```
//!
//! ## Supported Solvers
//!
//! The library provides multiple solver implementations for different use cases:
//!
//! | Solver | Method | Implicit | Adaptive | Best For |
//! |--------|--------|----------|----------|----------|
//! | [`Solver::Euler`] | Euler | ❌ | ❌ | Simple, educational use |
//! | [`Solver::RK4`] | Runge-Kutta 4th Order | ❌ | ❌ | General-purpose, fixed step |
//! | [`Solver::ImplicitEuler`] | Implicit Euler | ✅ | ❌ | Stiff problems |
//! | [`Solver::GLRK4`] | Gauss-Legendre RK (Order 4) | ✅ | ❌ | High-accuracy, stiff systems |
//! | [`Solver::RKF45`] | Runge-Kutta-Fehlberg 4(5) | ❌ | ✅ | Adaptive step control |
//! | [`Solver::ROW1`] | Rosenbrock-Wanner (Order 1) | semi | ❌ | Fast semi-implicit, stiff |

#[cfg(test)]
mod tests;

pub(crate) mod utils;

pub mod optimizers;

mod solvers;
pub use solvers::Solver;
