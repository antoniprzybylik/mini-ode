//! Nonlinear optimization algorithms which are required by some ODE solvers
//!
//! The user may create objects containing optimizer configuration and pass it to ODE solver.

use anyhow::anyhow;
use std::fmt;
use tch::{IndexOp, Tensor};

/// Optimizer interface common for any optimizer in the library
pub trait Optimizer: Send + Sync + fmt::Display {
    /// Solves the problem of optimization of function `function` starting from point `x0`
    fn optimize(&self, function: &dyn Fn(&Tensor) -> Tensor, x0: &Tensor)
    -> anyhow::Result<Tensor>;
}

/// Newton optimization algorithm
pub struct Newton {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // minimum change in the objective function between iterations
    ftol: Option<f64>,
}

/// Broyden-Fletcher-Goldfarb-Shanno optimization algorithm
pub struct BFGS {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // minimum change in the objective function between iterations
    ftol: Option<f64>,
}

/// Conjugate Gradient optimization algorithm
pub struct CG {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // Minimum change in the objective function between iterations
    ftol: Option<f64>,
}

fn differentiate(function: &dyn Fn(&Tensor) -> Tensor, x: &Tensor) -> Tensor {
    let x_with_grad = x.detach().copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    tch::Tensor::run_backward(&[y], &[x_with_grad], false, false)[0].copy()
}

fn gradient_and_hessian(function: &dyn Fn(&Tensor) -> Tensor, x: &Tensor) -> (Tensor, Tensor) {
    let x_with_grad = x.detach().copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    // keep_graph = true (this graph is needed for some functions during second differentiation)
    // create_graph = true (allow calculating second derivatives)
    let grad = Tensor::run_backward(&[y], &[&x_with_grad], true, true)[0].copy();
    let grad_len = grad.size()[0];

    let mut vectors = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        // keep_graph = true (we need to run backward pass multiple times - in each iteration of the loop)
        // create_graph = false (we don't need to differentiate three times)
        vectors.append(&mut Tensor::run_backward(
            &[grad.i(i)],
            &[&x_with_grad],
            true,
            false,
        ));
    }

    (grad.detach(), Tensor::stack(&vectors, 0).detach())
}

const P0: f64 = 0.0000000001f64;

const PHI2: f64 = 2.618033988749894848207f64;
const RPHI: f64 = 0.618033988749894848207f64;

fn choose_step_golden_section(
    x0: &Tensor,
    direction: &Tensor,
    function: &dyn Fn(&Tensor) -> Tensor,
    atol: f64,
) -> Tensor {
    let (mut x1, mut x2, mut x3, mut x4): (f64, f64, f64, f64);
    let (fx1, mut fx3, mut fx4): (f64, f64, f64);

    fx1 = function(&x0).double_value(&[]);

    x1 = 0.;
    // Heuristics: Try to set x2 based on atol value. If we succeed, we can
    //             skip some forward search iterations.
    x2 = if function(&(x0 + direction * atol * 15.)).double_value(&[]) <= fx1 {
        atol * 15.
    } else {
        P0
    };
    // Forward search
    while function(&(x0 + direction * x2)).double_value(&[]) <= fx1 {
        x2 = &x1 + (&x2 - &x1) * PHI2;
    }

    x3 = x2 - (x2 - x1) * RPHI;
    x4 = x1 + (x2 - x1) * RPHI;
    fx3 = function(&(x0 + direction * x3)).double_value(&[]);
    fx4 = function(&(x0 + direction * x4)).double_value(&[]);
    while x2 - x1 > atol {
        if fx3 < fx4 {
            x2 = x4;

            fx4 = fx3;
            x3 = x2 - (x2 - x1) * RPHI;
            x4 = x1 + (x2 - x1) * RPHI;
            fx3 = function(&(x0 + direction * x3)).double_value(&[]);
        } else {
            x1 = x3;

            fx3 = fx4;
            x3 = x2 - (x2 - x1) * RPHI;
            x4 = x1 + (x2 - x1) * RPHI;
            fx4 = function(&(x0 + direction * x4)).double_value(&[]);
        }
    }

    direction * ((x1 + x2) / 2.)
}

fn choose_step_backtracking(
    x0: &Tensor,
    direction: &Tensor,
    function: &dyn Fn(&Tensor) -> Tensor,
    grad: &Tensor,
    alpha: f64,
    beta: f64,
) -> Tensor {
    let fx0 = function(&x0).double_value(&[]);

    let mut t = 1f64;
    while function(&(x0 + direction * t)).double_value(&[])
        > fx0 + grad.squeeze().dot(&direction.squeeze()).double_value(&[]) * alpha * t
    {
        t *= beta;
    }

    direction.copy() * t
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

        let mut prev_grad = Tensor::zeros_like(&x0);
        let mut prev_direction = Tensor::zeros_like(&x0);
        let mut prev_y: Option<Tensor> = None;
        let mut x = x0.copy();

        for step_num in 0..self.max_steps {
            let grad = differentiate(function, &x);

            // Stop if gradient is smaller than `gtol`
            if let Some(gtol) = self.gtol {
                if grad.norm().double_value(&[]) < gtol {
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if grad.norm().double_value(&[]) == 0. {
                    return Ok(x);
                }
            }

            // Calculate direction with PR+ and orthogonality-based restart
            let direction = match step_num {
                0 => -&grad,
                _ => {
                    let orthogonality_measure = grad.squeeze().dot(&prev_grad.squeeze()).abs()
                        / grad.squeeze().dot(&grad.squeeze());
                    if orthogonality_measure.double_value(&[]) > 0.2 {
                        // Restart
                        -&grad
                    } else {
                        let beta = grad.squeeze().dot(&(&grad - &prev_grad).squeeze())
                            / prev_grad.squeeze().dot(&prev_grad.squeeze());
                        // Clamp beta to be nonnegative (PR+)
                        let beta = if beta.double_value(&[]) > 0. {
                            beta
                        } else {
                            tch::Tensor::zeros_like(&beta)
                        };
                        // Clamp beta to not be too large (this may result in numerical instability)
                        let beta = if beta.double_value(&[]) > 1000000000000. {
                            tch::Tensor::ones_like(&beta) * 1000000000000.
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
            let step = choose_step_golden_section(&x, &direction, &function, linesearch_atol);

            // Update previous step norms
            prev3_step_norm = prev2_step_norm;
            prev2_step_norm = prev_step_norm;
            prev_step_norm = step.norm().double_value(&[]);

            // Apply step
            x = x + step;

            // Stop if change in function value is smaller than `ftol`
            let y = function(&x);
            if let (Some(prev_y), Some(ftol)) = (prev_y, self.ftol) {
                if (&prev_y - &y).double_value(&[]) < ftol {
                    return Ok(x);
                }
            }
            prev_y = Some(y);

            // Update previous gradient value and previous direction value
            prev_grad = grad;
            prev_direction = direction;
        }

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
        let mut curr_grad = differentiate(function, &x);
        let mut curr_y = function(&x);

        // Ensure that output of `function` is a scalar
        if curr_y.size() != Vec::<i64>::new() {
            return Err(anyhow!("Output of function `function` must be scalar"));
        }

        for _ in 0..self.max_steps {
            // Check for stop condition
            if let Some(gtol) = self.gtol {
                if curr_grad.norm().double_value(&[]) < gtol {
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.norm().double_value(&[]) == 0. {
                    return Ok(x);
                }
            }

            // Calculate step direction base on the gradient and approximate hessian
            let direction = (-appr_inv_h.mm(&curr_grad.unsqueeze(1))).squeeze();

            // Calculate linesearch_atol based on previous step norms
            let linesearch_atol =
                P0.max(prev_step_norm.min(prev2_step_norm).min(prev3_step_norm) / 100.);

            // Choose optimal step in given direction using line search
            let step = choose_step_golden_section(&x, &direction, function, linesearch_atol);

            // Update previous step norms
            prev3_step_norm = prev2_step_norm;
            prev2_step_norm = prev_step_norm;
            prev_step_norm = step.norm().double_value(&[]);

            // Apply step
            x = x + &step;

            // Check for stop contition
            let y = function(&x);
            if let Some(ftol) = self.ftol {
                if (curr_y.double_value(&[]) - y.double_value(&[])) < ftol {
                    return Ok(x);
                }
            }
            curr_y = y;

            let grad = differentiate(function, &x);
            let gdiff = &grad - &curr_grad;

            // Use Powell's dampening for gamma computation
            // This prevents gamma from blowing up. Normal formula for gamma is 1/step.dot(gdiff)
            let gamma = {
                let delta = 0.0001;

                let sty = step.dot(&gdiff).double_value(&[]);
                let step_norm_sq = step.dot(&step).double_value(&[]);

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
                let sty_prime = step.dot(&gdiff_prime).double_value(&[]);

                if sty_prime.abs() < 1e-10 {
                    1. / (delta * step_norm_sq + 1e-10)
                } else {
                    1. / sty_prime
                }
            };

            // Compute approximation of inverse Hessian
            appr_inv_h = (&identity - gamma * step.reshape([-1, 1]).mm(&gdiff.reshape([1, -1])))
                .mm(&appr_inv_h)
                .mm(&(&identity - gamma * gdiff.reshape([-1, 1]).mm(&step.reshape([1, -1]))))
                + gamma * step.reshape([-1, 1]).mm(&step.reshape([1, -1]));

            curr_grad = grad;
        }

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

        for _ in 0..self.max_steps {
            let (curr_grad, curr_hessian) = gradient_and_hessian(function, &x);

            // Check for stop condition
            if let Some(gtol) = self.gtol {
                if curr_grad.norm().double_value(&[]) < gtol {
                    return Ok(x);
                }
            } else {
                // This check is necessary. Continuation of the algorithm
                // with gradient equal to exactly zero leads to NaN appearing
                // in the result.
                if curr_grad.norm().double_value(&[]) == 0. {
                    return Ok(x);
                }
            }

            // Calculate step direction
            let negative_grad = -curr_grad.unsqueeze(1); // Negative gradient direction
            let mut lambda = (negative_grad.norm().double_value(&[]) * 1e-3).max(1e-8); // Initial dampening factor
            let direction = loop {
                // We damp hessian until it is positive definite.
                // For non-positive definite Hessian, Newton method may give unwanted results.
                let damped_hessian =
                    &curr_hessian + Tensor::eye(x0_length, (kind, device)) * lambda;

                // Try to perform Banach-Cholesky decomposition of damped hessian
                match damped_hessian.f_linalg_cholesky(false) {
                    Ok(lower_triangular) => {
                        // Hessian is positive-definite. Solve system with Banach-Cholesky decomposition
                        let y = lower_triangular.linalg_solve_triangular(
                            &negative_grad,
                            false,
                            true,
                            false,
                        );
                        break lower_triangular
                            .transpose(0, 1)
                            .linalg_solve_triangular(&y, true, true, false)
                            .squeeze();
                    }
                    Err(_) => {
                        // Hessian is not positive-definite. Try increasing dampening factor.
                        lambda *= 10.;
                        if lambda > 1e10 {
                            // Dampening factor (lambda) exceeded maximum value. Fallback to pseudoinverse.
                            break curr_hessian
                                .linalg_pinv(1e-14, false)
                                .mm(&negative_grad)
                                .squeeze();
                        }
                    }
                }
            };

            // Choose optimal step in given direction using line search
            let step = choose_step_backtracking(&x, &direction, function, &curr_grad, 0.1, 0.9);

            // Apply step
            x = x + &step;

            // Check for stop contition
            let y = function(&x);
            if let Some(ftol) = self.ftol {
                if (curr_y.double_value(&[]) - y.double_value(&[])) < ftol {
                    return Ok(x);
                }
            }
            curr_y = y;
        }

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
