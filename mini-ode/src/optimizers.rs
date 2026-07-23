//! Nonlinear optimization algorithms which are required by some ODE solvers
//!
//! The user may create objects containing optimizer configuration and pass it to ODE solver.

use anyhow::anyhow;
use std::fmt;
use tch::{IndexOp, Tensor};

#[cfg(feature = "warnings")]
use tracing::warn;

#[cfg(not(feature = "warnings"))]
macro_rules! warn {
    ($($arg:tt)*) => {};
}

/// Optimizer interface common for any optimizer in the library
pub trait Optimizer: Send + Sync + fmt::Display {
    /// Solves the problem of optimization of function `function` starting from point `x0`
    ///
    /// # Arguments
    /// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
    ///   representing the objective value to minimize.
    /// * `x0` - Initial guess, 1D tensor accepted by `function`.
    ///
    /// # Returns
    /// Optimal `x`, or error if optimization fails.
    ///
    /// # Panics
    /// May panic if libtorch tensor operations fail.
    fn optimize(&self, function: &dyn Fn(&Tensor) -> Tensor, x0: &Tensor)
    -> anyhow::Result<Tensor>;
}

/// Helper function to validate that optimizer output is finite
fn validate_optimizer_output(tensor: &Tensor, optimizer_name: &str) -> anyhow::Result<()> {
    if tensor.isfinite().f_all()?.f_int64_value(&[])? == 0 {
        anyhow::bail!(
            "Optimizer {} produced non-finite result (NaN/Inf)",
            optimizer_name
        );
    }
    Ok(())
}

/// Newton optimization algorithm
///
/// This struct configures the Newton method, a second-order optimizer that uses the
/// Hessian matrix for quadratic approximations.
///
/// # Fields
/// * `max_steps` - Maximum number of optimization steps.
/// * `gtol` - Optional tolerance for gradient norm (stop if ||grad|| < gtol).
/// * `ftol` - Optional tolerance for change in objective value (stop if |f - prev_f| < ftol).
pub struct Newton {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // minimum change in the objective function between iterations
    ftol: Option<f64>,
}

/// Broyden-Fletcher-Goldfarb-Shanno optimization algorithm
///
/// This struct configures the BFGS quasi-Newton method, which approximates the inverse
/// Hessian using rank-2 updates. It is as memory-intensive as regular Newton method
/// (O(n^2) storage) but it does not require double differentiation.
///
/// # Fields
/// * `max_steps` - Maximum number of optimization steps.
/// * `gtol` - Optional tolerance for gradient norm (stop if ||grad|| < gtol).
/// * `ftol` - Optional tolerance for change in objective value (stop if |f - prev_f| < ftol).
pub struct BFGS {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // minimum change in the objective function between iterations
    ftol: Option<f64>,
}

/// Halley optimization algorithm
///
/// This struct configures the Halley method, a third-order optimizer that uses
/// tensor of third order derivatives.
///
/// # Fields
/// * `max_steps` - Maximum number of optimization steps.
/// * `gtol` - Optional tolerance for gradient norm (stop if ||grad|| < gtol).
/// * `ftol` - Optional tolerance for change in objective value (stop if |f - prev_f| < ftol).
pub struct Halley {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // minimum change in the objective function between iterations
    ftol: Option<f64>,
}

/// Conjugate Gradient optimization algorithm
///
/// This struct configures the nonlinear conjugate gradient method with Polak-Ribiere+
/// (PR+) beta and orthogonality-based restarts. It is gradient-only (first-order) and
/// memory-efficient, suitable for large-scale problems.
///
/// # Fields
/// * `max_steps` - Maximum number of optimization steps.
/// * `gtol` - Optional tolerance for gradient norm (stop if ||grad|| < gtol).
/// * `ftol` - Optional tolerance for change in objective value (stop if |f - prev_f| < ftol).
pub struct CG {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // Minimum change in the objective function between iterations
    ftol: Option<f64>,
}

/// Computes the gradient of `function` at `x` using automatic differentiation.
///
/// # Arguments
/// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
/// * `x` - Evaluation point (1D tensor).
///
/// # Returns
/// Gradient tensor at `x`.
pub(crate) fn differentiate(
    function: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
) -> anyhow::Result<Tensor> {
    let x_with_grad = x.f_detach()?.copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    if y.size() != [] as [i64; 0] {
        return Err(anyhow!(
            "Bad shape of `y`. Expected [], but got {:?}",
            y.size()
        ));
    }

    if !y.requires_grad() {
        return Ok(tch::Tensor::f_zeros(x.size(), (x.kind(), x.device()))?);
    }

    let gradient = tch::Tensor::f_run_backward(&[y], &[x_with_grad], false, false)?[0].copy();

    // Note: We don't validate here because gradients can legitimately be non-finite
    // during exploration; caller will validate final result

    Ok(gradient)
}

/// Computes the gradient and Hessian of `function` at `x` using automatic differentiation.
///
/// # Arguments
/// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
/// * `x` - Evaluation point (1D tensor).
/// # Returns
/// Tuple `(grad, hessian)`, both detached tensors. `grad` is 1D, `hessian` is 2D.
pub(crate) fn gradient_and_hessian(
    function: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_with_grad = x.f_detach()?.copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    if y.size() != [] as [i64; 0] {
        return Err(anyhow!(
            "Bad shape of `y`. Expected [], but got {:?}",
            y.size()
        ));
    }

    if !y.requires_grad() {
        return Ok((
            tch::Tensor::f_zeros(x.size(), (x.kind(), x.device()))?,
            tch::Tensor::f_zeros([x.size()[0], x.size()[0]], (x.kind(), x.device()))?,
        ));
    }

    // keep_graph = true (this graph is needed for some functions during second differentiation)
    // create_graph = true (allow calculating second derivatives)
    let grad = Tensor::f_run_backward(&[y], &[&x_with_grad], true, true)?[0].copy();
    let grad_len = grad.size()[0];
    let grad_kind = grad.kind();
    let grad_device = grad.device();

    // If gradient is constant, immediately return gradient and zero hessian
    // It is not possible to differentiate constants in torch
    if !grad.requires_grad() {
        return Ok((
            grad,
            Tensor::f_zeros([grad_len, grad_len], (grad_kind, grad_device))?,
        ));
    }

    let mut vectors = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        // keep_graph = true (we need to run backward pass multiple times - in each iteration of the loop)
        // create_graph = false (we don't need to differentiate three times)
        vectors.append(&mut Tensor::f_run_backward(
            &[grad.i(i)],
            &[&x_with_grad],
            true,
            false,
        )?);
    }

    // Detach autograd computation graph
    let grad = grad.f_detach()?;
    // Stack slices of the Hessian matrix and detach autograd computation graph
    let hessian = Tensor::f_stack(&vectors, 0)?.f_detach()?;

    Ok((grad, hessian))
}

/// Computes the gradient, Hessian and third derivatives tensor of `function` at `x` using automatic differentiation.
///
/// # Arguments
/// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
/// * `x` - Evaluation point (1D tensor).
/// # Returns
/// Tuple `(grad, hessian, d3_tensor)`, both detached tensors. `grad` is 1D, `hessian` is 2D, `d3_tensor` is 3D.
pub(crate) fn derivative_tensors_123(
    function: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let x_with_grad = x.f_detach()?.copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    if y.size() != [] as [i64; 0] {
        return Err(anyhow!(
            "Bad shape of `y`. Expected [], but got {:?}",
            y.size()
        ));
    }

    if !y.requires_grad() {
        return Ok((
            tch::Tensor::f_zeros(x.size(), (x.kind(), x.device()))?,
            tch::Tensor::f_zeros([x.size()[0], x.size()[0]], (x.kind(), x.device()))?,
            tch::Tensor::f_zeros(
                [x.size()[0], x.size()[0], x.size()[0]],
                (x.kind(), x.device()),
            )?,
        ));
    }

    // keep_graph = true (this graph is needed for some functions during second differentiation)
    // create_graph = true (allow calculating second derivatives)
    let grad = Tensor::f_run_backward(&[y], &[&x_with_grad], true, true)?[0].copy();
    let grad_len = grad.size()[0];
    let grad_kind = grad.kind();
    let grad_device = grad.device();

    // If gradient is constant, immediately return gradient zero hessian and
    // zero tensor of third order derivatives
    // It is not possible to differentiate constants in torch
    if !grad.requires_grad() {
        return Ok((
            grad,
            Tensor::f_zeros([grad_len, grad_len], (grad_kind, grad_device))?,
            Tensor::f_zeros([grad_len, grad_len, grad_len], (grad_kind, grad_device))?,
        ));
    }

    let mut vectors = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        // keep_graph = true (we need to run backward pass multiple times - in each iteration of the loop)
        // create_graph = true (we need to differentiate three times)
        vectors.append(&mut Tensor::f_run_backward(
            &[grad.i(i)],
            &[&x_with_grad],
            true,
            true,
        )?);
    }

    // Stack slices of the Hessian matrix
    let hessian = Tensor::f_stack(&vectors, 0)?;

    // If gradient is constant, immediately return gradient zero hessian and
    // zero tensor of third order derivatives
    // It is not possible to differentiate constants in torch
    if !hessian.requires_grad() {
        return Ok((
            grad,
            hessian,
            Tensor::f_zeros([grad_len, grad_len, grad_len], (grad_kind, grad_device))?,
        ));
    }

    let mut vectors2 = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        let mut vectors1 = Vec::<Tensor>::with_capacity(grad_len as usize);
        for j in 0..grad_len {
            vectors1.append(&mut Tensor::f_run_backward(
                &[hessian.i((i, j))],
                &[&x_with_grad],
                true,
                false,
            )?);
        }
        vectors2.push(Tensor::f_stack(&vectors1, 0)?);
    }

    // Detach autograd computation graph
    let grad = grad.f_detach()?;
    let hessian = hessian.f_detach()?;
    // Stack slices of the tensor of third derivatives and detach autograd computation graph
    let d3_tensor = Tensor::f_stack(&vectors2, 0)?.f_detach()?;

    Ok((grad, hessian, d3_tensor))
}

/// Minimum step value.
const P0: f64 = 0.0000000001f64;

/// Golden ratio squared (phi^2)
const PHI2: f64 = 2.618033988749894848207f64;
/// Reciprocal of golden ratio (1/phi)
const RPHI: f64 = 0.618033988749894848207f64;

/// Performs a golden section line search to find a step size that approximately minimizes
/// `function` along `direction` from `x0`, subject to tolerance `atol`.
///
/// # Arguments
/// * `x0` - Starting point (1D tensor).
/// * `direction` - Search direction (1D tensor).
/// * `function` - Objective function.
/// * `atol` - Absolute tolerance for step size convergence.
///
/// # Returns
/// Optimal step tensor.
fn choose_step_golden_section(
    x0: &Tensor,
    direction: &Tensor,
    function: &dyn Fn(&Tensor) -> Tensor,
    atol: f64,
) -> anyhow::Result<Tensor> {
    let (mut x1, mut x2, mut x3, mut x4): (f64, f64, f64, f64);
    let (fx1, mut fx3, mut fx4): (f64, f64, f64);

    fx1 = function(&x0).f_double_value(&[])?;

    x1 = 0.;
    // Heuristics: Try to set x2 based on atol value. If we succeed, we can
    //             skip some forward search iterations.
    let fx_guess = function(&(x0 + direction * atol * 15.)).f_double_value(&[])?;
    x2 = if !fx_guess.is_finite() || fx_guess > fx1 {
        P0
    } else {
        atol * 15.
    };
    // Forward search - continues even if non-finite encountered (gentle handling)
    let mut fx = function(&(x0 + direction * x2)).f_double_value(&[])?;
    let mut forward_iters: u32 = 0;
    while fx <= fx1 {
        let new_x2 = x1 + (x2 - x1) * PHI2;
        fx = function(&(x0 + direction * new_x2)).f_double_value(&[])?;
        if !fx.is_finite() {
            break;
        }
        x2 = new_x2;
        forward_iters += 1;
    }

    x3 = x2 - (x2 - x1) * RPHI;
    x4 = x1 + (x2 - x1) * RPHI;
    fx3 = function(&(x0 + direction * x3)).f_double_value(&[])?;
    fx4 = function(&(x0 + direction * x4)).f_double_value(&[])?;

    let mut refine_iters: u32 = 0;
    while x2 - x1 > atol && refine_iters < 500 {
        if fx3 < fx4 {
            x2 = x4;

            fx4 = fx3;
            x3 = x2 - (x2 - x1) * RPHI;
            x4 = x1 + (x2 - x1) * RPHI;
            fx3 = function(&(x0 + direction * x3)).f_double_value(&[])?;
        } else {
            x1 = x3;

            fx3 = fx4;
            x3 = x2 - (x2 - x1) * RPHI;
            x4 = x1 + (x2 - x1) * RPHI;
            fx4 = function(&(x0 + direction * x4)).f_double_value(&[])?;
        }
        refine_iters += 1;
    }

    // Warnings after line search completes
    if forward_iters >= 100 {
        warn!(
            "golden section line search: forward search took {} iterations without bracketing a minimum; direction may be poor",
            forward_iters
        );
    }

    if refine_iters >= 200 {
        warn!(
            "golden section line search: refinement took {} iterations; atol may be too small or objective flat",
            refine_iters
        );
    }

    Ok(direction * ((x1 + x2) / 2.))
}

/// Performs a backtracking line search to find a step size satisfying the Armijo condition.
///
/// # Arguments
/// * `x0` - Starting point (1D tensor).
/// * `direction` - Descent direction (1D tensor).
/// * `function` - Objective function.
/// * `grad` - Gradient at `x0`.
/// * `alpha` - Armijo parameter (0 < alpha < 1, 0.1 is recommended).
/// * `beta` - Backtracking factor (0 < beta < 1, 0.9 is recommended).
///
/// # Returns
/// Step tensor.
fn choose_step_backtracking(
    x0: &Tensor,
    direction: &Tensor,
    function: &dyn Fn(&Tensor) -> Tensor,
    grad: &Tensor,
    alpha: f64,
    beta: f64,
) -> anyhow::Result<Tensor> {
    let fx0 = function(&x0).f_double_value(&[])?;

    let mut t = 1f64;
    let mut backtrack_iters: u32 = 0;

    while {
        let fx = function(&(x0 + direction * t)).f_double_value(&[])?;

        if !fx.is_finite() {
            true
        } else {
            fx > fx0
                + grad
                    .f_reshape([-1])?
                    .f_dot(&direction.f_reshape([-1])?)?
                    .f_double_value(&[])?
                    * alpha
                    * t
        }
    } {
        t *= beta;
        backtrack_iters += 1;
        if t < 1e-30 {
            break;
        }
    }

    // Warnings after line search completes
    if backtrack_iters >= 100 {
        warn!(
            "backtracking line search: {} iterations without satisfying Armijo condition; direction may not be a descent direction",
            backtrack_iters
        );
    }

    if t < 1e-30 {
        warn!(
            "backtracking line search: step size collapsed to {:.3e}; optimizer may be stuck",
            t
        );
    }

    Ok(direction.copy() * t)
}

impl CG {
    pub fn new(max_steps: usize, gtol: Option<f64>, ftol: Option<f64>) -> Self {
        Self {
            max_steps,
            gtol,
            ftol,
        }
    }
}

impl Optimizer for CG {
    /// Creates a new CG optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `max_steps` - Maximum iterations.
    /// * `gtol` - Optional gradient tolerance.
    /// * `ftol` - Optional function value change tolerance.
    ///
    /// # Returns
    /// Configured CG instance.
    fn optimize(
        &self,
        function: &dyn Fn(&Tensor) -> Tensor,
        x0: &Tensor,
    ) -> anyhow::Result<Tensor> {
        // Ensure that rank of the initital guess tensor is 1
        if x0.size().len() != 1 {
            return Err(anyhow!("`x0` must have rank 1"));
        }

        let mut prev3_step_norm = 0f64;
        let mut prev2_step_norm = 0f64;
        let mut prev_step_norm = 0f64;

        let mut prev_grad = Tensor::f_zeros_like(&x0)?;
        let mut prev_direction = Tensor::f_zeros_like(&x0)?;
        let mut prev_y: Option<Tensor> = None;
        let mut x = x0.copy();

        let mut warned_nonfinite_grad = false;
        let mut warned_beta_clamp = false;
        let mut warned_nonfinite_iter = false;

        for step_num in 0..self.max_steps {
            let grad = match differentiate(function, &x) {
                Ok(grad) => grad,
                Err(e) => {
                    return Err(anyhow!(
                        "Runtime error: Differentiation failed in CG optimizer: {}",
                        e
                    ));
                }
            };

            // Warning: non-finite gradient
            if !warned_nonfinite_grad && grad.isfinite().f_all()?.f_int64_value(&[])? == 0 {
                warn!("CG: non-finite gradient detected; function may be ill-defined");
                warned_nonfinite_grad = true;
            }

            // Stop if gradient is smaller than `gtol`
            if let Some(gtol) = self.gtol {
                if grad.norm().f_double_value(&[])? < gtol {
                    // Final result validation
                    validate_optimizer_output(&x, "CG")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if grad.norm().f_double_value(&[])? == 0. {
                    // Final result validation
                    validate_optimizer_output(&x, "CG")?;
                    return Ok(x);
                }
            }

            // Calculate direction with PR+ and orthogonality-based restart
            let direction = match step_num {
                0 => -&grad,
                _ => {
                    let orthogonality_measure = grad
                        .f_reshape([-1])?
                        .f_dot(&prev_grad.f_reshape([-1])?)?
                        .f_abs()?
                        / grad.f_reshape([-1])?.f_dot(&grad.f_reshape([-1])?)?;
                    if orthogonality_measure.f_double_value(&[])? > 0.2 {
                        // Restart
                        -&grad
                    } else {
                        let beta = grad
                            .f_reshape([-1])?
                            .f_dot(&(&grad - &prev_grad).f_reshape([-1])?)?
                            / prev_grad
                                .f_reshape([-1])?
                                .f_dot(&prev_grad.f_reshape([-1])?)?;
                        // Clamp beta to be nonnegative (PR+)
                        let beta = if beta.f_double_value(&[])? > 0. {
                            beta
                        } else {
                            tch::Tensor::f_zeros_like(&beta)?
                        };
                        // Clamp beta to not be too large (this may result in numerical instability)
                        let beta = if beta.f_double_value(&[])? > 1e12 {
                            if !warned_beta_clamp {
                                warn!("CG: beta clamped to 1e12; optimizer may be diverging");
                                warned_beta_clamp = true;
                            }
                            tch::Tensor::f_ones_like(&beta)? * 1e12
                        } else {
                            beta
                        };

                        -&grad + beta * &prev_direction
                    }
                }
            };

            // Calculate linesearch_atol based on previous step norms
            let linesearch_atol =
                P0.max(prev_step_norm.min(prev2_step_norm).min(prev3_step_norm) / 1000.);

            // Choose step in direction `direction`
            // Note: golden section handles non-finite gracefully during search
            let step = choose_step_golden_section(&x, &direction, &function, linesearch_atol)?;

            // Update previous step norms
            prev3_step_norm = prev2_step_norm;
            prev2_step_norm = prev_step_norm;
            prev_step_norm = step.f_norm()?.f_double_value(&[])?;

            // Apply step
            x = x + step;

            // Warning: non-finite iterate
            if !warned_nonfinite_iter && x.isfinite().f_all()?.f_int64_value(&[])? == 0 {
                warn!("CG: non-finite iterate detected; step size may be too large");
                warned_nonfinite_iter = true;
            }

            // Stop if change in function value is smaller than `ftol`
            let y = function(&x);
            if let (Some(prev_y), Some(ftol)) = (prev_y, self.ftol) {
                if (&prev_y - &y).f_double_value(&[])? < ftol {
                    // Final result validation
                    validate_optimizer_output(&x, "CG")?;
                    return Ok(x);
                }
            }
            prev_y = Some(y);

            // Update previous gradient value and previous direction value
            prev_grad = grad;
            prev_direction = direction;
        }

        // Final result validation
        validate_optimizer_output(&x, "CG")?;
        Ok(x)
    }
}

impl fmt::Display for CG {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut string = String::from("CG(");

        string = string + "max_steps=" + self.max_steps.to_string().as_str();
        if let Some(gtol) = self.gtol {
            string = string + ", gtol=" + gtol.to_string().as_str();
        }
        if let Some(ftol) = self.ftol {
            string = string + ", ftol=" + ftol.to_string().as_str();
        }
        string = string + ")";

        write!(f, "{}", string)
    }
}

impl BFGS {
    /// Creates a new BFGS optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `max_steps` - Maximum iterations.
    /// * `gtol` - Optional gradient tolerance.
    /// * `ftol` - Optional function value change tolerance.
    ///
    /// # Returns
    /// Configured BFGS instance.
    pub fn new(max_steps: usize, gtol: Option<f64>, ftol: Option<f64>) -> Self {
        Self {
            max_steps,
            gtol,
            ftol,
        }
    }
}

impl Optimizer for BFGS {
    fn optimize(
        &self,
        function: &dyn Fn(&Tensor) -> Tensor,
        x0: &Tensor,
    ) -> anyhow::Result<Tensor> {
        // Ensure that rank of the initital guess tensor is 1
        if x0.size().len() != 1 {
            return Err(anyhow!("`x0` must have rank 1"));
        }

        // Determine the device and kind for use in the function
        let kind = x0.kind();
        let device = x0.device();

        let mut prev3_step_norm = 0f64;
        let mut prev2_step_norm = 0f64;
        let mut prev_step_norm = 0f64;

        let x0_length = x0.size()[0];
        let identity = match Tensor::f_eye(x0_length, (kind, device)) {
            Ok(matrix) => matrix,
            // BFGS requires a lot of resources.
            // Give knowledgable error message to the user
            // when BFGS fails due to unsufficient memory.
            Err(tch::TchError::Torch(_)) => {
                return Err(anyhow!(
                    "Could not allocate {}x{} matrix. Maybe try less resourcefull algorithm.",
                    x0_length,
                    x0_length
                ));
            }
            e => e.unwrap(),
        };
        let mut x = x0.copy();
        let mut appr_inv_h = identity.copy();
        let mut curr_grad = match differentiate(function, &x) {
            Ok(grad) => grad,
            Err(e) => {
                return Err(anyhow!(
                    "Runtime error: Differentiation failed in BFGS optimizer: {}",
                    e
                ));
            }
        };
        let mut curr_y = function(&x);

        // Ensure that output of `function` is a scalar
        if curr_y.size() != Vec::<i64>::new() {
            return Err(anyhow!("Output of function `function` must be scalar"));
        }

        let mut warned_inv_hess_large = false;

        for _ in 0..self.max_steps {
            // Check for stop condition
            if let Some(gtol) = self.gtol {
                if curr_grad.f_norm()?.f_double_value(&[])? < gtol {
                    // Final result validation
                    validate_optimizer_output(&x, "BFGS")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.f_norm()?.f_double_value(&[])? == 0. {
                    // Final result validation
                    validate_optimizer_output(&x, "BFGS")?;
                    return Ok(x);
                }
            }

            // Calculate step direction base on the gradient and approximate hessian
            let direction = (-appr_inv_h.f_mm(&curr_grad.f_reshape([-1, 1])?)?).f_reshape([-1])?;

            // Calculate linesearch_atol based on previous step norms
            let linesearch_atol =
                P0.max(prev_step_norm.min(prev2_step_norm).min(prev3_step_norm) / 100.);

            // Choose optimal step in given direction using line search
            // Line search handles non-finite gracefully during search
            let step = choose_step_golden_section(&x, &direction, function, linesearch_atol)?;

            // Update previous step norms
            prev3_step_norm = prev2_step_norm;
            prev2_step_norm = prev_step_norm;
            prev_step_norm = step.f_norm()?.f_double_value(&[])?;

            // Apply step
            x = x + &step;

            // Check for stop contition
            let y = function(&x);
            if let Some(ftol) = self.ftol {
                if (curr_y.f_double_value(&[])? - y.f_double_value(&[])?) < ftol {
                    // Final result validation
                    validate_optimizer_output(&x, "BFGS")?;
                    return Ok(x);
                }
            }
            curr_y = y;

            let grad = match differentiate(function, &x) {
                Ok(grad) => grad,
                Err(e) => {
                    return Err(anyhow!(
                        "Runtime error: Differentiation failed in BFGS optimizer: {}",
                        e
                    ));
                }
            };
            let gdiff = &grad - &curr_grad;

            // Use Powell's dampening for gamma computation
            // This prevents gamma from blowing up. Normal formula for gamma is 1/step.dot(gdiff)
            let gamma = {
                let delta = 0.0001;

                let sty = step.f_dot(&gdiff)?.f_double_value(&[])?;
                let step_norm_sq = step.f_dot(&step)?.f_double_value(&[])?;

                let theta = if sty >= delta * step_norm_sq {
                    1.
                } else {
                    let numerator = (1. - delta) * step_norm_sq;
                    let denominator = step_norm_sq - sty;

                    if denominator.abs() < 1e-10 {
                        1.
                    } else {
                        (numerator / denominator).min(1.)
                    }
                };

                let projection_factor = if step_norm_sq < 1e-10 {
                    0.
                } else {
                    sty / step_norm_sq
                };
                let gdiff_prime = &gdiff * theta + &step * ((1. - theta) * projection_factor);
                let sty_prime = step.f_dot(&gdiff_prime)?.f_double_value(&[])?;

                if sty_prime.abs() < 1e-10 {
                    1. / (delta * step_norm_sq + 1e-10)
                } else {
                    1. / sty_prime
                }
            };

            // Compute approximation of inverse Hessian
            appr_inv_h = (&identity
                - gamma * step.f_reshape([-1, 1])?.f_mm(&gdiff.f_reshape([1, -1])?)?)
            .f_mm(&appr_inv_h)?
            .f_mm(
                &(&identity - gamma * gdiff.f_reshape([-1, 1])?.f_mm(&step.f_reshape([1, -1])?)?),
            )? + gamma * step.f_reshape([-1, 1])?.f_mm(&step.f_reshape([1, -1])?)?;

            // Warning: large inverse Hessian norm
            let inv_h_norm = appr_inv_h.f_norm()?.f_double_value(&[])?;
            if !warned_inv_hess_large && inv_h_norm > 1e10 {
                warn!(
                    "BFGS: inverse Hessian approximation norm reached {:.3e}; problem may be ill-conditioned",
                    inv_h_norm
                );
                warned_inv_hess_large = true;
            }

            curr_grad = grad;
        }

        // Final result validation
        validate_optimizer_output(&x, "BFGS")?;
        Ok(x)
    }
}

impl fmt::Display for BFGS {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut string = String::from("BFGS(");

        string = string + "max_steps=" + self.max_steps.to_string().as_str();
        if let Some(gtol) = self.gtol {
            string = string + ", gtol=" + gtol.to_string().as_str();
        }
        if let Some(ftol) = self.ftol {
            string = string + ", ftol=" + ftol.to_string().as_str();
        }

        string = string + ")";

        write!(f, "{}", string)
    }
}

impl Newton {
    /// Creates a new Newton optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `max_steps` - Maximum iterations.
    /// * `gtol` - Optional gradient tolerance.
    /// * `ftol` - Optional function value change tolerance.
    ///
    /// # Returns
    /// Configured Newton instance.
    pub fn new(max_steps: usize, gtol: Option<f64>, ftol: Option<f64>) -> Self {
        Self {
            max_steps,
            gtol,
            ftol,
        }
    }
}

impl Optimizer for Newton {
    fn optimize(
        &self,
        function: &dyn Fn(&Tensor) -> Tensor,
        x0: &Tensor,
    ) -> anyhow::Result<Tensor> {
        // Ensure that rank of the initital guess tensor is 1
        if x0.size().len() != 1 {
            return Err(anyhow!("`x0` must have rank 1"));
        }

        // Determine the device and kind for use in the function
        let kind = x0.kind();
        let device = x0.device();

        let x0_length = x0.size()[0];

        // Test for sufficient resources for storing Hessian
        let _ = match Tensor::f_eye(x0_length, (kind, device)) {
            Ok(matrix) => matrix,
            // Give knowledgable error message to the user
            // when there is unsufficient memory.
            Err(tch::TchError::Torch(_)) => {
                return Err(anyhow!(
                    "Could not allocate {}x{} matrix. Maybe try less resourcefull algorithm.",
                    x0_length,
                    x0_length
                ));
            }
            e => e.unwrap(),
        };

        let mut x = x0.copy();
        let mut curr_y = function(&x);

        // Ensure that output of `function` is a scalar
        if curr_y.size() != Vec::<i64>::new() {
            return Err(anyhow!("Output of function `function` must be scalar"));
        }

        let mut warned_damping_moderate = false;
        let mut warned_damping_severe = false;

        for _ in 0..self.max_steps {
            let (curr_grad, curr_hessian) = match gradient_and_hessian(function, &x) {
                Ok(gh) => gh,
                Err(e) => {
                    return Err(anyhow!(
                        "Runtime error: Differentiation failed in Newton optimizer: {}",
                        e
                    ));
                }
            };

            // Check for stop condition
            if let Some(gtol) = self.gtol {
                if curr_grad.f_norm()?.f_double_value(&[])? < gtol {
                    // Final result validation
                    validate_optimizer_output(&x, "Newton")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.f_norm()?.f_double_value(&[])? == 0. {
                    // Final result validation
                    validate_optimizer_output(&x, "Newton")?;
                    return Ok(x);
                }
            }

            // Calculate step direction
            let negative_grad = -curr_grad.f_reshape([-1, 1])?; // Negative gradient direction
            let mut lambda = (negative_grad.f_norm()?.f_double_value(&[])? * 1e-3).max(1e-8); // Initial dampening factor
            let direction = loop {
                // We damp hessian until it is positive definite.
                // For non-positive definite Hessian, Newton method may give unwanted results.
                let damped_hessian =
                    &curr_hessian + Tensor::f_eye(x0_length, (kind, device))? * lambda;

                // Try to perform Banach-Cholesky decomposition of damped hessian
                match damped_hessian.f_linalg_cholesky(false) {
                    Ok(lower_triangular) => {
                        // Hessian is positive-definite. Solve system with Banach-Cholesky decomposition
                        let y = lower_triangular.f_linalg_solve_triangular(
                            &negative_grad,
                            false,
                            true,
                            false,
                        )?;
                        break lower_triangular
                            .f_transpose(0, 1)?
                            .f_linalg_solve_triangular(&y, true, true, false)?
                            .reshape([-1]);
                    }
                    Err(_) => {
                        // Hessian is not positive-definite. Try increasing dampening factor.
                        lambda *= 10.;

                        // Warnings for damping levels
                        if !warned_damping_moderate && lambda >= 1e3 && lambda < 1e7 {
                            warn!(
                                "Newton: Hessian required damping factor {:.3e}; problem may be ill-conditioned",
                                lambda
                            );
                            warned_damping_moderate = true;
                        }

                        if !warned_damping_severe && lambda >= 1e10 {
                            warn!(
                                "Newton: Hessian damping factor reached {:.3e}; falling back to pseudoinverse (Hessian is severely ill-conditioned)",
                                lambda
                            );
                            warned_damping_severe = true;
                        }

                        if lambda > 1e10 {
                            // Dampening factor (lambda) exceeded maximum value. Fallback to pseudoinverse.
                            break curr_hessian
                                .f_linalg_pinv(1e-14, false)?
                                .f_mm(&negative_grad)?
                                .f_reshape([-1])?;
                        }
                    }
                }
            };

            // Choose optimal step in given direction using line search
            // Backtracking handles non-finite gracefully during search
            let step = choose_step_backtracking(&x, &direction, function, &curr_grad, 0.1, 0.9)?;

            // Apply step
            x = x + &step;

            // Check for stop contition
            let y = function(&x);
            if let Some(ftol) = self.ftol {
                if (curr_y.f_double_value(&[])? - y.f_double_value(&[])?) < ftol {
                    // Final result validation
                    validate_optimizer_output(&x, "Newton")?;
                    return Ok(x);
                }
            }
            curr_y = y;
        }

        // Final result validation
        validate_optimizer_output(&x, "Newton")?;
        Ok(x)
    }
}

impl fmt::Display for Newton {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut string = String::from("Newton(");

        string = string + "max_steps=" + self.max_steps.to_string().as_str();
        if let Some(gtol) = self.gtol {
            string = string + ", gtol=" + gtol.to_string().as_str();
        }
        if let Some(ftol) = self.ftol {
            string = string + ", ftol=" + ftol.to_string().as_str();
        }

        string = string + ")";

        write!(f, "{}", string)
    }
}

impl Halley {
    /// Creates a new Halley optimizer with the given parameters.
    ///
    /// # Arguments
    /// * `max_steps` - Maximum iterations.
    /// * `gtol` - Optional gradient tolerance.
    /// * `ftol` - Optional function value change tolerance.
    ///
    /// # Returns
    /// Configured Halley instance.
    pub fn new(max_steps: usize, gtol: Option<f64>, ftol: Option<f64>) -> Self {
        Self {
            max_steps,
            gtol,
            ftol,
        }
    }
}

impl Optimizer for Halley {
    fn optimize(
        &self,
        function: &dyn Fn(&Tensor) -> Tensor,
        x0: &Tensor,
    ) -> anyhow::Result<Tensor> {
        // Ensure that rank of the initital guess tensor is 1
        if x0.size().len() != 1 {
            return Err(anyhow!("`x0` must have rank 1"));
        }

        // Determine the device and kind for use in the function
        let kind = x0.kind();
        let device = x0.device();

        let x0_length = x0.size()[0];

        // Test for sufficient resources for storing tensor of third order derivatives
        let _ = match Tensor::f_zeros([x0_length, x0_length, x0_length], (kind, device)) {
            Ok(matrix) => matrix,
            // Give knowledgable error message to the user
            // when there is unsufficient memory.
            Err(tch::TchError::Torch(_)) => {
                return Err(anyhow!(
                    "Could not allocate {}x{}x{} tensor. Maybe try less resourcefull algorithm.",
                    x0_length,
                    x0_length,
                    x0_length
                ));
            }
            e => e.unwrap(),
        };

        let mut x = x0.copy();
        let mut curr_y = function(&x);

        // Ensure that output of `function` is a scalar
        if curr_y.size() != Vec::<i64>::new() {
            return Err(anyhow!("Output of function `function` must be scalar"));
        }

        let mut warned_pinv_large = false;

        for _ in 0..self.max_steps {
            let (curr_grad, curr_hessian, curr_d3_tensor) =
                match derivative_tensors_123(function, &x) {
                    Ok(ghd3) => ghd3,
                    Err(e) => {
                        return Err(anyhow!(
                            "Runtime error: Differentiation failed in Halley optimizer: {}",
                            e
                        ));
                    }
                };

            // Check for stop condition
            if let Some(gtol) = self.gtol {
                if curr_grad.f_norm()?.f_double_value(&[])? < gtol {
                    // Final result validation
                    validate_optimizer_output(&x, "Halley")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.f_norm()?.f_double_value(&[])? == 0. {
                    // Final result validation
                    validate_optimizer_output(&x, "Halley")?;
                    return Ok(x);
                }
            }

            // Calculate step direction
            let hessian_pinv = curr_hessian.f_linalg_pinv(1e-14, false)?;

            // Warning: large pseudoinverse norm
            let pinv_norm = hessian_pinv.f_norm()?.f_double_value(&[])?;
            if !warned_pinv_large && pinv_norm > 1e8 {
                warn!(
                    "Halley: Hessian pseudoinverse norm is {:.3e}; Hessian may be ill-conditioned",
                    pinv_norm
                );
                warned_pinv_large = true;
            }

            let neg_newton_dir = hessian_pinv.f_mm(&curr_grad.f_reshape([-1, 1])?)?;
            let direction = -hessian_pinv
                .f_mm(
                    &(curr_grad.f_reshape([-1, 1])?
                        + curr_d3_tensor
                            .f_matmul(&neg_newton_dir)?
                            .f_reshape([x0_length, x0_length])?
                            .f_mm(&neg_newton_dir)?
                            * 0.5),
                )?
                .f_reshape([-1])?;

            // Choose optimal step in given direction using line search
            // Backtracking handles non-finite gracefully during search
            let step = choose_step_backtracking(&x, &direction, function, &curr_grad, 0.1, 0.9)?;

            // Apply step
            x = x + &step;

            // Check for stop contition
            let y = function(&x);
            if let Some(ftol) = self.ftol {
                if (curr_y.f_double_value(&[])? - y.f_double_value(&[])?) < ftol {
                    // Final result validation
                    validate_optimizer_output(&x, "Halley")?;
                    return Ok(x);
                }
            }
            curr_y = y;
        }

        // Final result validation
        validate_optimizer_output(&x, "Halley")?;
        Ok(x)
    }
}

impl fmt::Display for Halley {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut string = String::from("Halley(");

        string = string + "max_steps=" + self.max_steps.to_string().as_str();
        if let Some(gtol) = self.gtol {
            string = string + ", gtol=" + gtol.to_string().as_str();
        }
        if let Some(ftol) = self.ftol {
            string = string + ", ftol=" + ftol.to_string().as_str();
        }

        string = string + ")";

        write!(f, "{}", string)
    }
}
