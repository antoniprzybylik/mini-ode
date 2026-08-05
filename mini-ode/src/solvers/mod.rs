use anyhow::anyhow;
use std::fmt;
use std::sync::Arc;
use tch::Tensor;

use crate::utils::validation;

use crate::optimizers;

mod explicit;
use explicit::solve_euler;
use explicit::solve_rk4;
use explicit::solve_rkf45;
mod implicit;
use implicit::solve_glrk4;
use implicit::solve_implicit_euler;
use implicit::solve_row1;

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
        validation::validate_finite_tensor(&y0, "initial state y0")?;

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
        validation::validate_finite_tensor(&dy, "derivative function output at initial point")?;

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
