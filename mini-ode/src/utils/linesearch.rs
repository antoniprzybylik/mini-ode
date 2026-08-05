use crate::utils::warnings::warn;
use tch::Tensor;

/// Minimum step value.
pub(crate) const P0: f64 = 0.0000000001f64;

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
pub(crate) fn choose_step_golden_section(
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
pub(crate) fn choose_step_backtracking(
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
