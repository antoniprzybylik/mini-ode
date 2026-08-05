use anyhow::anyhow;
use std::fmt;
use tch::Tensor;

use super::Optimizer;

use crate::utils::differentiation;
use crate::utils::linesearch;
use crate::utils::validation;
use crate::utils::warnings::warn;

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
                match differentiation::derivative_tensors_123(function, &x) {
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
                    validation::validate_optimizer_output(&x, "Halley")?;
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.f_norm()?.f_double_value(&[])? == 0. {
                    // Final result validation
                    validation::validate_optimizer_output(&x, "Halley")?;
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
                    validation::validate_optimizer_output(&x, "Halley")?;
                    return Ok(x);
                }
            }
            curr_y = y;
        }

        // Final result validation
        validation::validate_optimizer_output(&x, "Halley")?;
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
