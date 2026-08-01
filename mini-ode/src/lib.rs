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

use anyhow::anyhow;
use std::fmt;
use std::sync::Arc;
use tch::IndexOp;
use tch::Tensor;

pub mod optimizers;

#[cfg(feature = "warnings")]
use tracing::warn;

#[cfg(not(feature = "warnings"))]
macro_rules! warn {
    ($($arg:tt)*) => {};
}

#[cfg(test)]
mod tests;

/// Validates that a tensor contains only finite values.
/// Returns an error if any NaN or Inf values are detected.
fn validate_finite_tensor(tensor: &Tensor, context: &str) -> anyhow::Result<()> {
    if tensor.isfinite().f_all()?.f_int64_value(&[])? == 0 {
        anyhow::bail!(
            "Non-finite values (NaN/Inf) detected in {}: tensor shape {:?}",
            context,
            tensor.size()
        );
    }
    Ok(())
}

/// Validates that a scalar value is finite.
fn validate_finite_scalar(value: f64, context: &str) -> anyhow::Result<()> {
    if !value.is_finite() {
        anyhow::bail!("Non-finite value ({}) detected in {}", value, context);
    }
    Ok(())
}

pub enum Solver {
    Euler {
        step: f64,
    },
    RK4 {
        step: f64,
    },
    ImplicitEuler {
        step: f64,
        optimizer: Arc<dyn optimizers::Optimizer>,
    },
    GLRK4 {
        step: f64,
        optimizer: Arc<dyn optimizers::Optimizer>,
    },
    RKF45 {
        rtol: f64,
        atol: f64,
        min_step: f64,
        safety_factor: f64,
    },
    ROW1 {
        step: f64,
    },
}

impl Solver {
    pub fn solve(
        &self,
        f: tch::CModule,
        x_span: (f64, f64),
        y0: Tensor,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let kind = y0.kind();
        let device = y0.device();

        // Validate x_span
        if !x_span.0.is_finite() || !x_span.1.is_finite() {
            return Err(anyhow!("x_span must consist of finite values"));
        }
        if x_span.0 > x_span.1 {
            return Err(anyhow!("x_span is not a valid interval"));
        }

        // Validate solver parameters
        match self {
            Self::Euler { step }
            | Self::RK4 { step }
            | Self::ImplicitEuler { step, .. }
            | Self::GLRK4 { step, .. }
            | Self::ROW1 { step } => {
                if !step.is_finite() || *step <= 0.0 {
                    return Err(anyhow!(
                        "Step size must be a finite positive value, got {}",
                        step
                    ));
                }
            }

            Self::RKF45 {
                rtol,
                atol,
                min_step,
                safety_factor,
            } => {
                if !rtol.is_finite() || *rtol <= 0.0 {
                    return Err(anyhow!(
                        "rtol must be a finite positive value, got {}",
                        rtol
                    ));
                }

                if !atol.is_finite() || *atol <= 0.0 {
                    return Err(anyhow!(
                        "atol must be a finite positive value, got {}",
                        atol
                    ));
                }

                if !min_step.is_finite() || *min_step <= 0.0 {
                    return Err(anyhow!(
                        "min_step must be a finite positive value, got {}",
                        min_step
                    ));
                }

                if !safety_factor.is_finite() || *safety_factor <= 0.0 {
                    return Err(anyhow!(
                        "safety_factor must be a finite positive value, got {}",
                        safety_factor
                    ));
                }
            }
        }

        // Validate y0 - check it's finite
        validate_finite_tensor(&y0, "initial state y0")?;

        let y0_size = y0.size();

        if y0_size.len() != 1 {
            return Err(anyhow!(
                "y0 must be a one-dimensional tensor but it has {} dimensions",
                y0_size.len()
            ));
        }

        if kind != tch::Kind::Double
            && kind != tch::Kind::Float
            && kind != tch::Kind::BFloat16
            && kind != tch::Kind::Half
        {
            return Err(anyhow!("y0 is of unsupported kind {:?}", y0.kind()));
        }

        // Validate function f
        let dy = f.forward_ts(&[
            Tensor::from(x_span.0).to_kind(kind).to_device(device),
            y0.copy(),
        ])?;

        let dy_size = dy.size();

        if dy_size.len() != 1 {
            return Err(anyhow!(
                "Function `f` returns tensor of rank {}, expected one-dimensional tensor",
                dy_size.len()
            ));
        }

        if dy_size[0] != y0_size[0] {
            return Err(anyhow!(
                "Function `f` returns vector of length {}, expected vector of length {} (same as y0)",
                dy_size[0],
                y0_size[0]
            ));
        }

        if dy.device() != device {
            return Err(anyhow!(
                "Function `f` returns tensor on device {:?}, expected tensor to be on device {:?} (same as y0)",
                dy.device(),
                device
            ));
        }

        if dy.kind() != kind {
            return Err(anyhow!(
                "Function `f` returns tensor of kind {:?}, expected tensor to be of kind {:?} (same as y0)",
                dy.kind(),
                kind
            ));
        }

        // Validate derivative output is finite
        validate_finite_tensor(&dy, "derivative function output at initial point")?;

        match self {
            Self::Euler { step } => solve_euler(f, x_span, y0, *step),

            Self::RK4 { step } => solve_rk4(f, x_span, y0, *step),

            Self::ImplicitEuler { step, optimizer } => {
                solve_implicit_euler(f, x_span, y0, *step, optimizer.as_ref())
            }

            Self::GLRK4 { step, optimizer } => {
                solve_glrk4(f, x_span, y0, *step, optimizer.as_ref())
            }

            Self::RKF45 {
                rtol,
                atol,
                min_step,
                safety_factor,
            } => solve_rkf45(f, x_span, y0, *rtol, *atol, *min_step, *safety_factor),

            Self::ROW1 { step } => solve_row1(f, x_span, y0, *step),
        }
    }

    pub fn stability_function(&self, x: f64) -> anyhow::Result<f64> {
        if x > 0. {
            anyhow::bail!("Stability function is not defined for positive numbers.");
        }

        Ok(match self {
            Self::Euler { .. } => 1. + x,
            Self::RK4 { .. } => 1. + (1. + (1. / 2. + (1. / 6. + (1. / 24.) * x) * x) * x) * x,
            Self::ImplicitEuler { .. } => 1. / (1. - x),
            Self::GLRK4 { .. } => (1. + x / 2. + x * x / 12.) / (1. - x / 2. + x * x / 12.),
            Self::RKF45 { .. } => {
                1. + (1.
                    + (1. / 2.
                        + (1. / 6. + (1. / 24. + (1. / 120. + (1. / 2080.) * x) * x) * x) * x)
                        * x)
                    * x
            }
            Self::ROW1 { .. } => 1. / (1. - x),
        })
    }

    pub fn stability_constant(&self) -> f64 {
        match self {
            Self::Euler { .. } => 2f64,
            Self::RK4 { .. } => 2.785293563f64,
            Self::ImplicitEuler { .. } => f64::INFINITY,
            Self::GLRK4 { .. } => f64::INFINITY,
            Self::RKF45 { .. } => 3.677706621f64,
            Self::ROW1 { .. } => f64::INFINITY,
        }
    }
}

impl fmt::Display for Solver {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Solver::Euler { step } => write!(f, "Euler(step={})", step),
            Solver::RK4 { step } => write!(f, "RK4(step={})", step),
            Solver::ImplicitEuler { step, optimizer } => {
                write!(f, "ImplicitEuler(step={}, optimizer={})", step, optimizer)
            }
            Solver::GLRK4 { step, optimizer } => {
                write!(f, "GLRK4(step={}, optimizer={})", step, optimizer)
            }
            Solver::RKF45 {
                rtol,
                atol,
                min_step,
                safety_factor,
            } => write!(
                f,
                "RKF45(rtol={}, atol={}, min_step={}, safety_factor={})",
                rtol, atol, min_step, safety_factor
            ),
            Solver::ROW1 { step } => write!(f, "ROW1(step={})", step),
        }
    }
}

/// Solves ODE using Euler method
fn solve_euler(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        let dy = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;

        validate_finite_tensor(&dy, "derivative from f(x, y) in Euler step")?;

        let dy_size = dy.size();
        let dy_rank = dy_size.len();
        if dy_rank != 1 {
            anyhow::bail!(
                "Derivative CModule returned tensor of bad rank {}.",
                dy_rank
            );
        }
        if dy_size[0] != y0.size()[0] {
            anyhow::bail!(
                "Derivative CModule returned vector of bad length {}.",
                dy_size[0]
            );
        }

        // Compute next state
        y = &y + current_step * &dy;

        // Critical: validate new state is finite before proceeding
        validate_finite_tensor(&y, "state after Euler update (NaN/Inf propagating)")?;

        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };

        // Validate x remains finite
        let x_tensor = Tensor::from(x).to_kind(kind).to_device(device);
        validate_finite_tensor(&x_tensor, "integration variable x in Euler step")?;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "Euler: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "Euler: reached {} steps; consider increasing step size or switching to a higher-order solver",
                step_count
            );
            warned_many_steps = true;
        }
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}

/// Solves ODE using Runge-Kutta 4th order method
fn solve_rk4(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        // Stage k1
        let k1 = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;
        validate_finite_tensor(&k1, "RK4 stage k1")?;

        let k1_size = k1.size();
        if k1_size.len() != 1 || k1_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k1");
        }

        // Stage k2
        let x_half = x + 0.5 * current_step;
        let y_half: Tensor = &y + 0.5 * current_step * &k1;
        validate_finite_tensor(&y_half, "RK4 intermediate state for k2")?;

        let k2 = f.forward_ts(&[Tensor::from(x_half).to_kind(kind).to_device(device), y_half])?;
        validate_finite_tensor(&k2, "RK4 stage k2")?;

        let k2_size = k2.size();
        if k2_size.len() != 1 || k2_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k2");
        }

        // Stage k3
        let x_half_again = x + 0.5 * current_step;
        let y_half_again: Tensor = &y + 0.5 * current_step * &k2;
        validate_finite_tensor(&y_half_again, "RK4 intermediate state for k3")?;

        let k3 = f.forward_ts(&[
            Tensor::from(x_half_again).to_kind(kind).to_device(device),
            y_half_again,
        ])?;
        validate_finite_tensor(&k3, "RK4 stage k3")?;

        let k3_size = k3.size();
        if k3_size.len() != 1 || k3_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k3");
        }

        // Stage k4
        let x_full = x + current_step;
        let y_full = &y + current_step * &k3;
        validate_finite_tensor(&y_full, "RK4 intermediate state for k4")?;

        let k4 = f.forward_ts(&[Tensor::from(x_full).to_kind(kind).to_device(device), y_full])?;
        validate_finite_tensor(&k4, "RK4 stage k4")?;

        let k4_size = k4.size();
        if k4_size.len() != 1 || k4_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k4");
        }

        // Compute next state using weighted average of stages
        let step_div_6 = current_step / 6.0;
        let y_next = &y + step_div_6 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        // Critical validation after full RK4 step
        validate_finite_tensor(&y_next, "state after RK4 update (NaN/Inf propagating)")?;

        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        y = y_next;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "RK4: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "RK4: reached {} steps; consider increasing step size or switching to an adaptive solver",
                step_count
            );
            warned_many_steps = true;
        }
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}

/// Solves ODE using Implicit Euler method with gradient descent optimization
fn solve_implicit_euler(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
    optimizer: &dyn optimizers::Optimizer,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        let x_next = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        let y_prev = y.copy();

        // Create derivative function for current x
        let f_next_fn = |y_next: &Tensor| {
            let f_next = f
                .forward_ts(&[
                    Tensor::from(x_next).to_kind(kind).to_device(device),
                    y_next.copy(),
                ])
                .unwrap();
            let y_pred = &y_prev + current_step * &f_next;
            (y_next - &y_pred).pow_tensor_scalar(2).sum(y_next.kind())
        };

        // Initial guess based on explicit Euler
        let initial_guess = &y_prev.detach()
            + current_step
                * f.forward_ts(&[&Tensor::from(x).to_kind(kind).to_device(device), &y_prev])?;

        // Run optimizer (may fail gracefully internally)
        let y_next = optimizer
            .optimize(&f_next_fn, &initial_guess)
            .map_err(|err| anyhow!(format!("Implicit solver optimizer failed with: {}", err)))?;

        // Critical: validate optimizer output before accepting
        validate_finite_tensor(
            &y_next,
            "state after implicit solver optimization (NaN/Inf)",
        )?;

        y = y_next.copy();
        x = x_next;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "ImplicitEuler: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "ImplicitEuler: reached {} steps; consider increasing step size",
                step_count
            );
            warned_many_steps = true;
        }
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}

/// Solves ODE using Gauss-Legendre-Runge-Kutta 4th order method
fn solve_glrk4(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
    optimizer: &dyn optimizers::Optimizer,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();
    let y_length = y.size()[0];

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        let k = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;
        validate_finite_tensor(&k, "GLRK4 initial derivative")?;

        let k_size = k.size();
        if k_size.len() != 1 || k_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape in GLRK4");
        }

        const C1: f64 = 0.2113248654f64;
        const C2: f64 = 0.7886751346f64;
        const A11: f64 = 0.25;
        const A12: f64 = -0.03867513459f64;
        const A21: f64 = 0.5386751346f64;
        const A22: f64 = 0.25;

        // Initial guess for k1, k2
        let first_k1k2_guess = Tensor::f_cat(
            &[
                f.forward_ts(&[
                    Tensor::from(x + C1 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + C1 * current_step * &k,
                ])?,
                f.forward_ts(&[
                    Tensor::from(x + C2 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + C2 * current_step * &k,
                ])?,
            ],
            0,
        )?;

        // Define loss function for optimization
        let loss_fn = |k1k2_guess: &Tensor| {
            let diff1 = k1k2_guess.i(0..y_length)
                - f.forward_ts(&[
                    Tensor::from(x + C1 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + (A11 * k1k2_guess.i(0..y_length)
                        + A12 * k1k2_guess.i(y_length..2 * y_length))
                        * current_step,
                ])
                .unwrap();
            let diff2 = k1k2_guess.i(y_length..2 * y_length)
                - f.forward_ts(&[
                    Tensor::from(x + C2 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + (A21 * k1k2_guess.i(0..y_length)
                        + A22 * k1k2_guess.i(y_length..2 * y_length))
                        * current_step,
                ])
                .unwrap();
            diff1.dot(&diff1) + diff2.dot(&diff2)
        };

        // Run optimizer
        let k1k2 = optimizer
            .optimize(&loss_fn, &first_k1k2_guess)
            .map_err(|err| anyhow!(format!("GLRK4 optimizer failed with: {}", err)))?;

        // Validate optimizer output
        validate_finite_tensor(
            &k1k2,
            "GLRK4 stage coefficients after optimization (NaN/Inf)",
        )?;

        // Compute final state update
        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        y = &y
            + current_step
                * (0.5 * k1k2.f_i(0..y_length)? + 0.5 * k1k2.f_i(y_length..2 * y_length)?);

        // Validate final state
        validate_finite_tensor(&y, "state after GLRK4 update (NaN/Inf propagating)")?;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "GLRK4: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "GLRK4: reached {} steps; consider increasing step size",
                step_count
            );
            warned_many_steps = true;
        }
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}

/// One RKF45 step
fn rkf45_step(
    f: &tch::CModule,
    x: f64,
    y: &Tensor,
    step: f64,
    device: tch::Device,
    kind: tch::Kind,
    y0_length: i64,
) -> anyhow::Result<(Tensor, Tensor)> {
    // Stage k1
    let k1 = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;
    validate_finite_tensor(&k1, "RKF45 stage k1")?;

    let k1_size = k1.size();
    if k1_size.len() != 1 || k1_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k2
    let x_step = x + 0.25 * step;
    let y_step: Tensor = y + 0.25 * &step * &k1;
    validate_finite_tensor(&y_step, "RKF45 intermediate state for k2")?;

    let k2 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validate_finite_tensor(&k2, "RKF45 stage k2")?;

    let k2_size = k2.size();
    if k2_size.len() != 1 || k2_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k3
    let x_step = x + 0.375 * step;
    let y_step: Tensor = y + (0.09375 * &step * &k1) + (0.28125 * &step * &k2);
    validate_finite_tensor(&y_step, "RKF45 intermediate state for k3")?;

    let k3 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validate_finite_tensor(&k3, "RKF45 stage k3")?;

    let k3_size = k3.size();
    if k3_size.len() != 1 || k3_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k4
    let x_step = x + (12.0 / 13.0) * step;
    let y_step: Tensor = y
        + (1932.0 / 2197.0 * &step * &k1)
        + (-7200.0 / 2197.0 * &step * &k2)
        + (7296.0 / 2197.0 * &step * &k3);
    validate_finite_tensor(&y_step, "RKF45 intermediate state for k4")?;

    let k4 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validate_finite_tensor(&k4, "RKF45 stage k4")?;

    let k4_size = k4.size();
    if k4_size.len() != 1 || k4_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k5
    let x_step = x + step;
    let y_step: Tensor = y
        + (439.0 / 216.0 * &step * &k1)
        + (-8.0 * &step * &k2)
        + (3680.0 / 513.0 * &step * &k3)
        + (-845.0 / 4104.0 * &step * &k4);
    validate_finite_tensor(&y_step, "RKF45 intermediate state for k5")?;

    let k5 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validate_finite_tensor(&k5, "RKF45 stage k5")?;

    let k5_size = k5.size();
    if k5_size.len() != 1 || k5_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k6
    let x_step = x + 0.5 * step;
    let y_step: Tensor = y
        + (-8.0 / 27.0 * &step * &k1)
        + (2.0 * &step * &k2)
        + (-3544.0 / 2565.0 * &step * &k3)
        + (1859.0 / 4104.0 * &step * &k4)
        + (-11.0 / 40.0 * &step * &k5);
    validate_finite_tensor(&y_step, "RKF45 intermediate state for k6")?;

    let k6 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validate_finite_tensor(&k6, "RKF45 stage k6")?;

    let k6_size = k6.size();
    if k6_size.len() != 1 || k6_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Compute 4th and 5th order solutions
    let next_y4: Tensor = y + step
        * ((25.0 / 216.0 * &k1)
            + (1408.0 / 2565.0 * &k3)
            + (2197.0 / 4104.0 * &k4)
            + (-1.0 / 5.0 * &k5));

    let next_y5: Tensor = y + step
        * ((16.0 / 135.0 * &k1)
            + (6656.0 / 12825.0 * &k3)
            + (28561.0 / 56430.0 * &k4)
            + (-9.0 / 50.0 * &k5)
            + (2.0 / 55.0 * &k6));

    Ok((next_y4, next_y5))
}

/// Solves ODE using Runge-Kutta-Fehlberg 45 adaptive method
fn solve_rkf45(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    rtol: f64,
    atol: f64,
    min_step: f64,
    safety_factor: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut step = (x_end - x_start) * 0.1;

    let mut consecutive_rejections: u32 = 0;
    let mut total_rejections: u32 = 0;
    let mut total_accepted: u32 = 0;

    let mut warned_consec_rej = false;
    let mut warned_total_rej = false;
    let mut warned_tiny_step = false;

    const MAX_GROWTH: f64 = 5.;

    while x < x_end {
        let (next_y4, next_y5) = rkf45_step(&f, x, &y, step, device, kind, y0.size()[0])?;

        // Compute error estimate
        let d = (&next_y4 - &next_y5).f_abs()?;
        validate_finite_tensor(&d, "RKF45 error estimate difference")?;

        let e = next_y5.f_abs()? * rtol + atol;
        validate_finite_tensor(&e, "RKF45 error tolerance combination")?;

        // Compute step size adjustment
        let alpha = (e / d)
            .f_pow_tensor_scalar(0.2)?
            .f_min()?
            .f_double_value(&[])?;

        let condition = (safety_factor * alpha).clamp(0f64, MAX_GROWTH);

        if condition < 1f64 {
            // Step rejected - shrink and retry
            consecutive_rejections += 1;
            total_rejections += 1;

            // Warning for consecutive rejections
            if !warned_consec_rej && consecutive_rejections >= 20 {
                warn!(
                    "RKF45: {} consecutive rejected steps at x={:.3e}, step={:.3e}; problem may be stiff",
                    consecutive_rejections, x, step
                );
                warned_consec_rej = true;
            }

            // Warning for many total rejections
            if !warned_total_rej && total_rejections >= 1000 {
                warn!(
                    "RKF45: {} total rejected steps ({} accepted) at x={:.3e}; integration is inefficient",
                    total_rejections, total_accepted, x
                );
                warned_total_rej = true;
            }

            // Warning for very small step approaching min_step
            if !warned_tiny_step && step < min_step * 10.0 {
                warn!(
                    "RKF45: required very small step {:.3e} at x={:.3e} (min_step={:.3e}); solution may be inaccurate or problem is stiff",
                    step, x, min_step
                );
                warned_tiny_step = true;
            }

            step = step * condition;
            validate_finite_scalar(step, "RKF45 reduced step size")?;

            // Warning for step below min_step
            if step < min_step {
                return Err(anyhow!("Required step is smaller than minimal step"));
            }
        } else {
            // Accept the step
            consecutive_rejections = 0;
            total_accepted += 1;

            // At last step, special handling
            let remaining = x_end - x;
            if remaining < step {
                step = remaining;
                let (_next_y4, next_y5) = rkf45_step(&f, x, &y, step, device, kind, y0.size()[0])?;
                y = next_y5;
                x = x_end;
                all_x.push(x);
                all_y.push(y.copy());
                break;
            }

            y = next_y5;
            x = &x + &step;

            // Validate accepted state
            validate_finite_tensor(&y, "RKF45 accepted state (NaN/Inf)")?;
            validate_finite_scalar(x, "RKF45 updated integration variable")?;

            all_x.push(x);
            all_y.push(y.copy());

            step = step * condition;
            validate_finite_scalar(step, "RKF45 next step size")?;
        }
    }

    // Final efficiency summary
    if total_rejections > total_accepted * 2 && total_accepted > 0 {
        warn!(
            "RKF45: integration completed with {} rejected and {} accepted steps; consider relaxing tolerances or using an implicit solver for stiff problems",
            total_rejections, total_accepted
        );
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}

/// Solves ODE using first-order Rosenbrock method (Row1)
fn solve_row1(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut warned_large_matrix = false;
    let mut warned_large_inverse = false;
    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        let x_prev = x;
        let y_prev = y.copy();

        // Compute Jacobian
        let jacobian = compute_jacobian(
            |y| {
                f.forward_ts(&[
                    Tensor::from(x_prev).to_kind(kind).to_device(device),
                    y.copy(),
                ])
                .unwrap()
            },
            &y_prev,
        )?;

        // Validate Jacobian is finite
        validate_finite_tensor(&jacobian, "Jacobian matrix in ROW1 (NaN/Inf)")?;

        // Evaluate function at current point
        let f_current = f.forward_ts(&[
            Tensor::from(x_prev).to_kind(kind).to_device(device),
            y_prev.copy(),
        ])?;

        validate_finite_tensor(&f_current, "derivative function output in ROW1")?;

        let f_current_size = f_current.size();
        let f_current_rank = f_current_size.len();
        if f_current_rank != 1 {
            anyhow::bail!(
                "Derivative CModule returned tensor of bad rank {}.",
                f_current_rank
            );
        }
        if f_current_size[0] != y0.size()[0] {
            anyhow::bail!(
                "Derivative CModule returned vector of bad length {}.",
                f_current_size[0]
            );
        }

        // Compute (I - h*J)^(-1) * f
        let n = jacobian.size()[0];
        let eye = Tensor::f_eye(n, (jacobian.kind(), jacobian.device()))?;
        let step_j = current_step * &jacobian;
        let matrix_to_invert = eye - step_j;

        // Warn about ill-conditioning before inversion (one-shot)
        let matrix_norm = matrix_to_invert.f_norm()?.f_double_value(&[])?;
        if !warned_large_matrix && matrix_norm > 1e12 {
            warn!(
                "ROW1: linear system matrix has large norm {:.3e} at x={:.3e}; solution may be unstable",
                matrix_norm, x_prev
            );
            warned_large_matrix = true;
        }

        let inv_matrix = matrix_to_invert.f_inverse()?;

        validate_finite_tensor(&inv_matrix, "inverse matrix (I - h*J)^(-1) in ROW1")?;

        // Warn about inverse magnitude (one-shot)
        let inv_norm = inv_matrix.f_norm()?.f_double_value(&[])?;
        if !warned_large_inverse && inv_norm > 1e10 {
            warn!(
                "ROW1: inverse matrix has large norm {:.3e} at x={:.3e}; Jacobian may be ill-conditioned",
                inv_norm, x_prev
            );
            warned_large_inverse = true;
        }

        let delta_y = inv_matrix.f_matmul(&f_current)?;
        validate_finite_tensor(&delta_y, "Newton correction step in ROW1")?;

        let y_next = y_prev + current_step * delta_y;

        // Critical validation after ROW1 update
        validate_finite_tensor(&y_next, "state after ROW1 update (NaN/Inf)")?;

        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        validate_finite_scalar(x, "ROW1 updated integration variable")?;

        y = y_next.detach().copy();

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "ROW1: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "ROW1: reached {} steps; consider increasing step size",
                step_count
            );
            warned_many_steps = true;
        }
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}

/// Computes the Jacobian matrix of a function f at point x
fn compute_jacobian<F>(f: F, x: &Tensor) -> anyhow::Result<Tensor>
where
    F: Fn(&Tensor) -> Tensor,
{
    if x.dim() != 1 {
        return Err(anyhow!(
            "Jacobian input tensor must be one-dimensional, got {} dimensions",
            x.dim()
        ));
    }

    let x_with_grad = x.detach().copy().set_requires_grad(true);

    let y = f(&x_with_grad);

    if y.dim() != 1 {
        return Err(anyhow!(
            "Jacobian output tensor must be one-dimensional, got {} dimensions",
            y.dim()
        ));
    }

    if y.isfinite().f_all()?.f_int64_value(&[])? == 0 {
        return Err(anyhow!(
            "Jacobian function returned tensor containing non-finite values"
        ));
    }

    let y_size = y.size()[0];
    let mut grads = Vec::with_capacity(y_size as usize);

    for i in 0..y_size {
        let yi = y.i(i);

        let grad = Tensor::f_run_backward(&[yi], &[&x_with_grad], true, false)?
            .first()
            .ok_or_else(|| anyhow!("Failed to compute Jacobian gradient"))?
            .copy();

        if grad.size() != x.size() {
            return Err(anyhow!(
                "Jacobian gradient has shape {:?}, expected shape {:?}",
                grad.size(),
                x.size()
            ));
        }

        if grad.isfinite().f_all()?.f_int64_value(&[])? == 0 {
            return Err(anyhow!("Jacobian computation produced non-finite values"));
        }

        grads.push(grad);
    }

    let jacobian = Tensor::f_stack(&grads, 0)?;

    if jacobian.size() != vec![y_size, x.size()[0]] {
        return Err(anyhow!(
            "Jacobian has shape {:?}, expected shape [{}, {}]",
            jacobian.size(),
            y_size,
            x.size()[0]
        ));
    }

    Ok(jacobian)
}
