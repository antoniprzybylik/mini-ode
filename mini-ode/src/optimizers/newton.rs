use anyhow::anyhow;
use std::fmt;
use tch::Tensor;

use super::Optimizer;

use crate::utils::differentiation;
use crate::utils::linesearch;
use crate::utils::validation;
use crate::utils::warnings::warn;

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
            let (curr_grad, curr_hessian) =
                match differentiation::gradient_and_hessian(function, &x) {
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
                    validation::validate_optimizer_output(&x, "Newton")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.f_norm()?.f_double_value(&[])? == 0. {
                    // Final result validation
                    validation::validate_optimizer_output(&x, "Newton")?;
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
            let step = linesearch::choose_step_backtracking(
                &x, &direction, function, &curr_grad, 0.1, 0.9,
            )?;

            // Apply step
            x = x + &step;

            // Check for stop contition
            let y = function(&x);
            if let Some(ftol) = self.ftol {
                if (curr_y.f_double_value(&[])? - y.f_double_value(&[])?) < ftol {
                    // Final result validation
                    validation::validate_optimizer_output(&x, "Newton")?;
                    return Ok(x);
                }
            }
            curr_y = y;
        }

        // Final result validation
        validation::validate_optimizer_output(&x, "Newton")?;
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
