use anyhow::anyhow;
use std::fmt;
use tch::Tensor;

use super::Optimizer;

use crate::utils::differentiation;
use crate::utils::linesearch;
use crate::utils::validation;
use crate::utils::warnings::warn;

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
        let mut curr_grad = match differentiation::differentiate(function, &x) {
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
                    validation::validate_optimizer_output(&x, "BFGS")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.f_norm()?.f_double_value(&[])? == 0. {
                    // Final result validation
                    validation::validate_optimizer_output(&x, "BFGS")?;
                    return Ok(x);
                }
            }

            // Calculate step direction base on the gradient and approximate hessian
            let direction = (-appr_inv_h.f_mm(&curr_grad.f_reshape([-1, 1])?)?).f_reshape([-1])?;

            // Calculate linesearch_atol based on previous step norms
            let linesearch_atol =
                linesearch::P0.max(prev_step_norm.min(prev2_step_norm).min(prev3_step_norm) / 100.);

            // Choose optimal step in given direction using line search
            // Line search handles non-finite gracefully during search
            let step =
                linesearch::choose_step_golden_section(&x, &direction, function, linesearch_atol)?;

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
                    validation::validate_optimizer_output(&x, "BFGS")?;
                    return Ok(x);
                }
            }
            curr_y = y;

            let grad = match differentiation::differentiate(function, &x) {
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
        validation::validate_optimizer_output(&x, "BFGS")?;
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
