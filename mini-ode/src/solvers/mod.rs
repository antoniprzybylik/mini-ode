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

/// Main enumeration of all supported ODE integration methods.
///
/// Each variant represents a different numerical algorithm with specific configuration parameters.
/// Choose based on problem characteristics:
///
/// - **Non-stiff problems**: [`Euler`](Self::Euler) for simplicity, [`RK4`](Self::RK4) for accuracy
/// - **Stiff problems**: [`ImplicitEuler`](Self::ImplicitEuler) or [`GLRK4`](Self::GLRK4)
/// - **Unknown step-size requirements**: [`RKF45`](Self::RKF45) with automatic error control
pub enum Solver {
    /// Forward Euler method (first-order explicit).
    ///
    /// The simplest ODE integrator: `y_{n+1} = y_n + h * f(x_n, y_n)`
    ///
    /// **Characteristics:**
    /// - Order: 1 (global error ~O(h), local truncation error ~O(h²))
    /// - Radius of absolute stability on the negative real axis: 2
    /// - Cost: One RHS evaluation per step
    ///
    /// **Best for:** Simple non-stiff systems where low accuracy is acceptable.
    Euler {
        /// Fixed step size for integration.
        ///
        /// Must be a finite positive value. Smaller steps improve accuracy but increase
        /// computation time linearly.
        step: f64,
    },

    /// Fourth-order Runge-Kutta method (RK4).
    ///
    /// The workhorse of ODE integration: uses four stage evaluations per step to achieve
    /// fourth-order accuracy. Formula involves weighted average of slopes at intermediate points.
    ///
    /// **Characteristics:**
    /// - Order: 4 (global error ~O(h⁴), local truncation error ~O(h⁵))
    /// - Radius of absolute stability on the negative real axis: approximately 2.785
    /// - Cost: Four RHS evaluations per step
    ///
    /// Halving the step size reduces the asymptotic global discretization error by
    /// approximately 16×.
    ///
    /// **Best for:** General-purpose integration of non-stiff systems where a good
    /// balance between accuracy and computational cost is required.
    RK4 {
        /// Fixed step size for integration.
        ///
        /// Must be a finite positive value. Fourth-order convergence means that,
        /// asymptotically, halving the step reduces the global discretization error
        /// by approximately 16×.
        step: f64,
    },

    /// Implicit (backward) Euler method (first-order).
    ///
    /// Defines the next state implicitly: `y_{n+1} = y_n + h * f(x_{n+1}, y_{n+1})`
    ///
    /// Each step requires solving a nonlinear system. This implementation uses the
    /// configured optimizer/nonlinear solver to perform this solve.
    ///
    /// **Characteristics:**
    /// - Order: 1 (global error ~O(h), local truncation error ~O(h²))
    /// - Stability: A-stable and L-stable
    /// - Cost: Nonlinear solve iterations plus RHS evaluations per step
    ///
    /// **Best for:** Stiff problems where stability is more important than high-order
    /// accuracy. Much slower per step than explicit methods but allows larger steps.
    ImplicitEuler {
        /// Fixed step size for integration.
        step: f64,
        /// Optimizer used to solve the implicit equation at each step.
        ///
        /// Common choices:
        /// - [`optimizers::Newton`] for well-conditioned problems
        /// - [`optimizers::CG`] for memory efficiency
        optimizer: Arc<dyn optimizers::Optimizer>,
    },

    /// Fourth-order Gauss-Legendre Runge-Kutta method (collocation).
    ///
    /// An implicit Runge-Kutta method based on Gaussian quadrature collocation points.
    /// Offers excellent stability and accuracy properties for stiff systems.
    ///
    /// **Characteristics:**
    /// - Order: 4 (global error ~O(h⁴), local truncation error ~O(h⁵))
    /// - Stability: A-stable
    /// - Cost: Nonlinear solves involving multiple implicit stages
    /// - Additional property: Symplectic for Hamiltonian systems
    ///
    /// **Best for:** High-accuracy integration of stiff systems where preservation of
    /// qualitative structure is important.
    GLRK4 {
        /// Fixed step size for integration.
        step: f64,
        /// Optimizer used to solve the implicit system at each step.
        ///
        /// Common choices:
        /// - [`optimizers::Newton`] for well-conditioned problems
        /// - [`optimizers::CG`] for memory efficiency
        optimizer: Arc<dyn optimizers::Optimizer>,
    },

    /// Runge-Kutta-Fehlberg method with adaptive step control (4th/5th order).
    ///
    /// Embedded Runge-Kutta pair that computes fourth- and fifth-order solutions
    /// simultaneously. The fifth-order solution is accepted, while the difference
    /// between the fourth- and fifth-order approximations provides an estimate of
    /// the local truncation error, which is used to adapt the step size automatically.
    ///
    /// **Characteristics:**
    /// - Orders: 4th-order error estimate and 5th-order solution
    /// - Cost: Six RHS evaluations per attempted step
    /// - Adaptive step control based on user-provided tolerances
    ///
    /// **Best for:** Problems where the appropriate step size is unknown or the
    /// solution smoothness varies throughout the integration interval.
    ///
    /// **Note:** Explicit adaptive methods do not remove stiffness limitations;
    /// stiff problems may still require extremely small step sizes.
    RKF45 {
        /// Relative error tolerance for step adaptation.
        ///
        /// Step size adjusted so that estimated error satisfies:
        /// `|error| <= max(rtol * |y|, atol)`
        rtol: f64,
        /// Absolute error tolerance for step adaptation.
        ///
        /// Controls accuracy when state values are close to zero.
        atol: f64,
        /// Minimum allowable step size.
        ///
        /// Prevents step size from becoming too small (infinite loop protection).
        min_step: f64,
        /// Safety factor for step size adjustments.
        ///
        /// Typically 0.8-0.9; conservative factors reduce step rejections.
        safety_factor: f64,
    },

    /// Rosenbrock-Wanner method (linearly implicit).
    ///
    /// A semi-implicit method that uses Jacobian information to transform the nonlinear
    /// implicit solve into a single linear solve.
    ///
    /// This implementation uses `γ = 1`, i.e. it solves `(I - hJ) k = f(x, y)`, which
    /// gives the same stability function as implicit Euler, `R(z) = 1/(1 - z)`. As a
    /// result it is L-stable. The difference from implicit Euler is nonlinear behavior
    /// and computational cost (a single linear solve instead of a nonlinear iteration),
    /// not the linear stability region.
    ///
    /// **Characteristics:**
    /// - Order: 1 (global error ~O(h), local truncation error ~O(h²))
    /// - Stability: L-stable (same stability function as implicit Euler)
    /// - Cost: One Jacobian evaluation and one linear solve per step
    /// - Avoids the nonlinear iterations required by fully implicit methods
    ///
    /// **Best for:** Moderately stiff problems where explicit methods require
    /// prohibitively small steps and full nonlinear implicit methods are too expensive.
    ROW1 {
        /// Fixed step size for integration.
        step: f64,
    },
}

impl Solver {
    /// Solve the ODE initial value problem `dy/dx = f(x, y)` over the specified interval.
    ///
    /// This method performs comprehensive input validation before integration begins.
    /// If validation passes, the appropriate solver algorithm is dispatched based on
    /// the `Solver` variant.
    ///
    /// # Arguments
    ///
    /// * `f` - TorchScript module implementing the derivative function `f(x, y) -> dy/dx`.
    ///         Must accept scalar `x` and 1D tensor `y`, return 1D tensor of same dimension.
    /// * `x_span` - Integration interval as `(start, end)` tuple. Both values must be finite
    ///              with `start <= end`.
    /// * `y0` - Initial state as 1D tensor of shape `(n,)` where `n` is system dimension.
    ///
    /// # Returns
    ///
    /// On success, returns `(xs, ys)` where:
    /// - `xs`: 1D tensor of integration points (shape `(num_points,)`)
    /// - `ys`: 2D tensor of states at each point (shape `(num_points, n)`)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `x_span` contains non-finite values or `start > end`
    /// - Step size/tolerance parameters are non-finite or non-positive
    /// - `y0` is not 1D, contains non-finite values, or has unsupported dtype
    /// - Derivative function `f` output dimension mismatches `y0`
    /// - Device/dtype mismatch between `y0` and `f` output
    /// - Integration fails numerically (NaN/Inf produced)
    ///
    /// # Example
    ///
    /// ```rust
    /// use mini_ode::Solver;
    /// use tch::{Tensor, CModule};
    ///
    ///# let y0 = Tensor::from_slice(&[1f64, 0f64]);
    ///# let mut closure = |inputs: &[Tensor]| {
    ///#     let _x = &inputs[0];
    ///#     let y = &inputs[1];
    ///#     let y0 = y.get(0);
    ///#     let y1 = y.get(1);
    ///#
    ///#     let dy0 = y1;
    ///#     let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);
    ///#
    ///#     vec![Tensor::stack(&[dy0, dy1], 0)]
    ///# };
    ///#
    ///# // Trace the function
    ///# let model = CModule::create_by_tracing(
    ///#     "ode_fn",
    ///#     "forward",
    ///#     &[Tensor::from(0.0f64), y0.shallow_clone()],
    ///#     &mut closure,
    ///# )?;
    /// let solver = Solver::RK4 { step: 0.01 };
    /// let x_span = (0.0, 10.0);
    /// let y0 = Tensor::from_slice(&[1.0, 0.0]);
    ///
    /// let (xs, ys) = solver.solve(model, x_span, y0)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    /// Compute the stability function (amplification factor) for this solver.
    ///
    /// The stability function `R(z)` describes how errors propagate for the test equation
    /// `y' = λy` where `z = hλ`. A solver is absolutely stable when `|R(z)| ≤ 1`.
    ///
    /// # Arguments
    ///
    /// * `x` - The stability variable `z = hλ` (must be non-positive for meaningful results).
    ///
    /// # Returns
    ///
    /// The stability function value `R(x)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x > 0` because stability functions are conventionally analyzed
    /// for `x ≤ 0` (left half-plane).
    ///
    /// # Mathematical Background
    ///
    /// For each solver:
    /// - **Euler**: `R(z) = 1 + z`
    /// - **RK4**: `R(z) = 1 + z + z²/2 + z³/6 + z⁴/24` (Taylor polynomial)
    /// - **Implicit Euler**: `R(z) = 1/(1 - z)` (A-stable)
    /// - **GLRK4**: `R(z) = (1 + z/2 + z²/12)/(1 - z/2 + z²/12)` (A-stable rational)
    /// - **RKF45**: Stability polynomial determined by the Fehlberg tableau
    /// - **ROW1**: `R(z) = 1/(1 - z)` (same as implicit Euler)
    ///
    /// # Example
    ///
    /// ```rust
    /// use mini_ode::Solver;
    ///
    /// let solver = Solver::Euler { step: 0.1 };
    /// let stability = solver.stability_function(-0.5)?;
    /// assert!((stability - 0.5).abs() < 1e-10);  // R(-0.5) = 1 - 0.5 = 0.5
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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

    /// Return the radius of absolute stability for this solver.
    ///
    /// This is the extent of the stability region along the negative real axis.
    /// Explicit methods have a finite radius, while A-stable methods have an
    /// unbounded stability region.
    ///
    /// # Values by Solver
    ///
    /// | Solver | Radius of absolute stability | Type |
    /// |--------|------------------------------|------|
    /// | Euler | 2.0 | Explicit |
    /// | RK4 | ~2.785 | Explicit |
    /// | Implicit Euler | ∞ | A-stable/L-stable |
    /// | GLRK4 | ∞ | A-stable |
    /// | RKF45 | ~3.678 (for this Fehlberg tableau) | Explicit (adaptive) |
    /// | ROW1 | ∞ | L-stable (γ = 1, same as implicit Euler) |
    ///
    /// # Practical Meaning
    ///
    /// A larger radius allows larger step sizes for stable integration. For stiff
    /// systems where eigenvalues have large negative real parts, only A-stable
    /// methods (unbounded radius) remain stable regardless of step size.
    pub fn stability_radius(&self) -> f64 {
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
    /// Format the solver with its configuration parameters.
    ///
    /// Useful for logging, debugging, and displaying solver choice to users.
    /// Example output: `"RK4(step=0.01)"` or `"RKF45(rtol=1e-5, atol=1e-5, min_step=1e-9, safety_factor=0.9)"`
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
