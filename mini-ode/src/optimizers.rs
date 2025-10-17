//! Nonlinear optimization algorithms which are required by some ODE solvers
//!
//! The user may create objects containing optimizer configuration and pass it to ODE solver.

use anyhow::anyhow;
use tch::Tensor;

/// Optimizer interface common for any optimizer in the library
pub trait Optimizer: Send + Sync {
    /// Solves the problem of optimization of function `function` starting from point `x0`
    fn optimize(&self, function: &dyn Fn(&Tensor) -> Tensor, x0: &Tensor)
    -> anyhow::Result<Tensor>;
}

/// Broyden-Fletcher-Goldfarb-Shanno optimization algorithm
pub struct BFGS {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // minimum change in the objective function between iterations
    ftol: Option<f64>,
    // Absolute error tolerance for line search
    linesearch_atol: f64,
}

/// Conjugate Gradient optimization algorithm
pub struct CG {
    // Maximum number of optimization steps
    max_steps: usize,
    // Minimum gradient
    gtol: Option<f64>,
    // Minimum change in the objective function between iterations
    ftol: Option<f64>,
    // Absolute error tolerance for line search
    linesearch_atol: f64,
}

fn differentiate(function: &dyn Fn(&Tensor) -> Tensor, x: &Tensor) -> Tensor {
    let x_with_grad = x.detach().copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    tch::Tensor::run_backward(&[y], &[x_with_grad], false, false)[0].copy()
}

const P0: f64 = 0.000001f64;

const PHI2: f64 = 2.618033988749894848207f64;
const RPHI: f64 = 0.618033988749894848207f64;

fn choose_step(
    x0: &Tensor,
    direction: &Tensor,
    function: &dyn Fn(&Tensor) -> Tensor,
    atol: f64,
) -> Tensor {
    let (mut x1, mut x2, mut x3, mut x4): (Tensor, Tensor, Tensor, Tensor);
    let (fx1, mut fx3, mut fx4): (Tensor, Tensor, Tensor);

    let kind = x0.kind();

    fx1 = function(&x0);

    x1 = Tensor::from(0.).to_kind(kind);
    x2 = Tensor::from(P0).to_kind(kind);
    while function(&(x0 + direction * &x2)).double_value(&[]) <= fx1.double_value(&[]) {
        x2 = &x1 + (&x2 - &x1) * PHI2;
    }

    x3 = x2.copy() - (x2.copy() - &x1) * RPHI;
    x4 = x1.copy() + (x2.copy() - &x1) * RPHI;
    fx3 = function(&(x0 + direction * &x3));
    fx4 = function(&(x0 + direction * &x4));
    while (x1.copy() - &x2).abs().double_value(&[]) > atol {
        if fx3.double_value(&[]) < fx4.double_value(&[]) {
            x2 = x4.copy();

            fx4 = fx3.copy();
            x3 = x2.copy() - (x2.copy() - &x1) * RPHI;
            x4 = x1.copy() + (x2.copy() - &x1) * RPHI;
            fx3 = function(&(x0 + direction * &x3));
        } else {
            x1 = x3.copy();

            fx3 = fx4.copy();
            x3 = x2.copy() - (x2.copy() - &x1) * RPHI;
            x4 = x1.copy() + (x2.copy() - &x1) * RPHI;
            fx4 = function(&(x0 + direction * &x4));
        }
    }

    direction * ((&x1 + &x2) / 2.)
}

impl CG {
    pub fn new(
        max_steps: usize,
        gtol: Option<f64>,
        ftol: Option<f64>,
        linesearch_atol: Option<f64>,
    ) -> Self {
        Self {
            max_steps,
            gtol,
            ftol,
            linesearch_atol: if let Some(linesearch_atol) = linesearch_atol {
                linesearch_atol
            } else {
                P0
            },
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

        let iters_to_reset = x0.size().len();
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

            // Calculate direction according to the Polak-Ribiere formula
            let direction = match step_num % iters_to_reset {
                0 => -&grad,
                _ => {
                    let beta = grad.squeeze().dot(&(&grad - &prev_grad).squeeze())
                        / prev_grad.squeeze().dot(&prev_grad.squeeze());

                    -&grad + beta * &prev_direction
                }
            };

            // Choose step in direction `direction`
            let step = choose_step(&x, &direction, &function, self.linesearch_atol);

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

impl BFGS {
    pub fn new(
        max_steps: usize,
        gtol: Option<f64>,
        ftol: Option<f64>,
        linesearch_atol: Option<f64>,
    ) -> Self {
        Self {
            max_steps,
            gtol,
            ftol,
            linesearch_atol: if let Some(linesearch_atol) = linesearch_atol {
                linesearch_atol
            } else {
                P0
            },
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

            // Choose optimal step in given direction using line search
            let step = choose_step(&x, &direction, function, self.linesearch_atol);

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
