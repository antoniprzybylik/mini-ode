use anyhow::anyhow;
use std::fmt;
use tch::Tensor;

use super::Optimizer;

use crate::utils::differentiation;
use crate::utils::linesearch;
use crate::utils::validation;
use crate::utils::warnings::warn;

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
            let grad = match differentiation::differentiate(function, &x) {
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
                    validation::validate_optimizer_output(&x, "CG")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if grad.norm().f_double_value(&[])? == 0. {
                    // Final result validation
                    validation::validate_optimizer_output(&x, "CG")?;
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
            let linesearch_atol = linesearch::P0
                .max(prev_step_norm.min(prev2_step_norm).min(prev3_step_norm) / 1000.);

            // Choose step in direction `direction`
            // Note: golden section handles non-finite gracefully during search
            let step =
                linesearch::choose_step_golden_section(&x, &direction, &function, linesearch_atol)?;

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
                    validation::validate_optimizer_output(&x, "CG")?;
                    return Ok(x);
                }
            }
            prev_y = Some(y);

            // Update previous gradient value and previous direction value
            prev_grad = grad;
            prev_direction = direction;
        }

        // Final result validation
        validation::validate_optimizer_output(&x, "CG")?;
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
