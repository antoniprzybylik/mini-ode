use anyhow::anyhow;
use tch::IndexOp;
use tch::Tensor;
use std::sync::Arc;

pub mod optimizers;

pub enum Solver {
    Euler { step: f64 },
    RK4 { step: f64 },
    ImplicitEuler { step: f64, optimizer: Arc<dyn optimizers::Optimizer> },
    GLRK4 { step: f64, optimizer: Arc<dyn optimizers::Optimizer> },
    RKF45 { rtol: f64, atol: f64, min_step: f64, safety_factor: f64 },
    ROW1 { step: f64 }
}

impl Solver {
    pub fn solve(
        &self,
        f: tch::CModule,
        x_span: Tensor,
        y0: Tensor
    ) -> anyhow::Result<(Tensor, Tensor)> {
        if x_span.size() != [2] {
            return Err(anyhow!("x_span must be of shape [2] but it has shape {:?}", x_span.size().as_slice()));
        }
        if y0.size().len() != 1 {
            return Err(anyhow!("y0 must be a one-dimensional tensor but it has {} dimensions", y0.size().len()));
        }
        if x_span.device() != y0.device() {
            return Err(anyhow!("x_span and y0 must reside on the same device. Device of x_span is {:?}. Device of y0 is {:?}", x_span.device(), y0.device()));
        }
        if x_span.kind() != tch::Kind::Double && x_span.kind() != tch::Kind::Float && x_span.kind() != tch::Kind::BFloat16 && x_span.kind() != tch::Kind::Half {
            return Err(anyhow!("x_span is of unsupported kind {:?}", x_span.kind()));
        }
        if y0.kind() != tch::Kind::Double && y0.kind() != tch::Kind::Float && y0.kind() != tch::Kind::BFloat16 && y0.kind() != tch::Kind::Half {
            return Err(anyhow!("y0 is of unsupported kind {:?}", y0.kind()));
        }
        if x_span.kind() != y0.kind() {
            return Err(anyhow!("x_span and y0 must be of the same kind. Kind of x_span is {:?}. Kind of y0 is {:?}", x_span.kind(), y0.kind()));
        }

        match self {
            Self::Euler { step } => solve_euler(f, x_span, y0, *step),
            Self::RK4 { step } => solve_rk4(f, x_span, y0, *step),
            Self::ImplicitEuler { step, optimizer } => solve_implicit_euler(f, x_span, y0, *step, optimizer.as_ref()),
            Self::GLRK4 { step, optimizer } => solve_glrk4(f, x_span, y0, *step, optimizer.as_ref()),
            Self::RKF45 { rtol, atol, min_step, safety_factor } => solve_rkf45(f, x_span, y0, *rtol, *atol, *min_step, *safety_factor),
            Self::ROW1 { step } => solve_row1(f, x_span, y0, *step)
        }
    }
}

/// Solves ODE using Euler method
fn solve_euler(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.double_value(&[]) < current_step {
            current_step = remaining.double_value(&[]);
        }

        let dy = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;
        let dy_rank = dy.size().len();
        if dy_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", dy_rank);
        }
        
        y = &y + current_step * &dy;
        x = &x + current_step;

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Runge-Kutta 4th order method
fn solve_rk4(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.double_value(&[]) < current_step {
            current_step = remaining.double_value(&[]);
        }

        let k1 = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;
        let k1_rank = k1.size().len();
        if k1_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k1_rank);
        }

        let x_half: Tensor = &x + 0.5 * current_step;
        let y_half: Tensor = &y + 0.5 * current_step * &k1;
        let k2 = f.forward_ts(&[x_half.squeeze(), y_half.squeeze()])?;
        let k2_rank = k2.size().len();
        if k2_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k2_rank);
        }

        let x_half_again: Tensor = &x + 0.5 * current_step;
        let y_half_again: Tensor = &y + 0.5 * current_step * &k2;
        let k3 = f.forward_ts(&[x_half_again.squeeze(), y_half_again.squeeze()])?;
        let k3_rank = k3.size().len();
        if k3_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k3_rank);
        }

        let x_full = &x + current_step;
        let y_full = &y + current_step * &k3;
        let k4 = f.forward_ts(&[x_full.squeeze(), y_full.squeeze()])?;
        let k4_rank = k4.size().len();
        if k4_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k4_rank);
        }

        let step_div_6 = current_step / 6.0;
        let y_next = &y + step_div_6 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        x = &x + current_step;
        y = y_next;

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Implicit Euler method with gradient descent optimization
fn solve_implicit_euler(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: f64,
    optimizer: &dyn optimizers::Optimizer,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.double_value(&[]) < current_step {
            current_step = remaining.double_value(&[]);
        }

        let x_next = &x + current_step;
        let y_prev = y.copy();

        let y_next = optimizer.optimize(
            &|y_next: &Tensor| {
                let f_next = f
                    .forward_ts(&[x_next.squeeze().copy(), y_next.squeeze().copy()])
                    .unwrap();
                let y_pred = &y_prev.squeeze() + current_step * &f_next;
                (y_next - &y_pred).pow_tensor_scalar(2).sum(y_next.kind())
            },
            &(&y_prev.detach().squeeze()
                + current_step * f.forward_ts(&[&x.squeeze(), &y_prev.squeeze()])?),
        ).map_err( |err| {
            anyhow!(format!("Optimizer failed with: {}", err))
        })?;

        y = y_next.unsqueeze(0);
        x = x_next.copy();

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Gauss-Legendre-Runge-Kutta 4th order method
fn solve_glrk4(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: f64,
    optimizer: &dyn optimizers::Optimizer,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.double_value(&[]) < current_step {
            current_step = remaining.double_value(&[]);
        }

        let k = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;
        let k_rank = k.size().len();
        if k_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k_rank);
        }

        const C1: f64 = 0.2113248654f64;
        const C2: f64 = 0.7886751346f64;
        const A11: f64 = 0.25;
        const A12: f64 = -0.03867513459f64;
        const A21: f64 = 0.5386751346f64;
        const A22: f64 = 0.25;

        let first_k1k2_guess = Tensor::cat(
            &[
                f.forward_ts(&[
                    &x.squeeze() + C1 * current_step,
                    &y.squeeze() + C1 * current_step * &k,
                ])?,
                f.forward_ts(&[
                    &x.squeeze() + C2 * current_step,
                    &y.squeeze() + C2 * current_step * &k,
                ])?,
            ],
            0,
        );
        let k1k2 = optimizer.optimize(
            &|k1k2_guess| {
                let diff1 = k1k2_guess.i(0..=1)
                    - f.forward_ts(&[
                        &x.squeeze() + C1 * current_step,
                        &y.squeeze()
                            + (A11 * k1k2_guess.i(0..=1) + A12 * k1k2_guess.i(2..=3))
                                * current_step,
                    ])
                    .unwrap();
                let diff2 = k1k2_guess.i(2..=3)
                    - f.forward_ts(&[
                        &x.squeeze() + C2 * current_step,
                        &y.squeeze()
                            + (A21 * k1k2_guess.i(0..=1) + A22 * k1k2_guess.i(2..=3))
                                * current_step,
                    ])
                    .unwrap();

                diff1.dot(&diff1) + diff2.dot(&diff2)
            },
            &first_k1k2_guess,
        ).map_err( |err| {
            anyhow!(format!("Optimizer failed with: {}", err))
        })?;
        assert!(k1k2.size().len() == 1);
        assert!(k1k2.size()[0] == 4);

        x = &x + current_step;
        y = &y + current_step * (0.5 * k1k2.i(0..=1) + 0.5 * k1k2.i(2..=3));

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Runge-Kutta-Fehlberg 45 adaptive method
fn solve_rkf45(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    rtol: f64,
    atol: f64,
    min_step: f64,
    safety_factor: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut step = (&x_end - &x_start) * 0.1;
    let safety_factor_tensor = Tensor::from(safety_factor);

    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.lt_tensor(&step) == Tensor::from(true) {
            step = remaining.copy();
        }

        let k1 = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;
        let k1_rank = k1.size().len();
        if k1_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k1_rank);
        }

        let k2 = {
            let x_step: Tensor = &x + 0.25 * &step;
            let y_step: Tensor = &y + 0.25 * &step * &k1;
            let k2_unchecked = f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?;
            let k2_rank = k2_unchecked.size().len();
            if k2_rank != 1 {
                anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k2_rank);
            }

            k2_unchecked
        };

        let k3 = {
            let x_step: Tensor = &x + 0.375 * &step;
            let y_step: Tensor = &y + (0.09375 * &step * &k1) + (0.28125 * &step * &k2);
            let k3_unchecked = f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?;
            let k3_rank = k3_unchecked.size().len();
            if k3_rank != 1 {
                anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k3_rank);
            }

            k3_unchecked
        };

        let k4 = {
            let x_step: Tensor = &x + (12.0 / 13.0) * &step;
            let y_step: Tensor = &y
                + (1932.0 / 2197.0 * &step * &k1)
                + (-7200.0 / 2197.0 * &step * &k2)
                + (7296.0 / 2197.0 * &step * &k3);
            let k4_unchecked = f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?;
            let k4_rank = k4_unchecked.size().len();
            if k4_rank != 1 {
                anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k4_rank);
            }

            k4_unchecked
        };

        let k5 = {
            let x_step: Tensor = &x + &step;
            let y_step: Tensor = &y
                + (439.0 / 216.0 * &step * &k1)
                + (-8.0 * &step * &k2)
                + (3680.0 / 513.0 * &step * &k3)
                + (-845.0 / 4104.0 * &step * &k4);
            let k5_unchecked = f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?;
            let k5_rank = k5_unchecked.size().len();
            if k5_rank != 1 {
                anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k5_rank);
            }

            k5_unchecked
        };

        let k6 = {
            let x_step: Tensor = &x + 0.5 * &step;
            let y_step: Tensor = &y
                + (-8.0 / 27.0 * &step * &k1)
                + (2.0 * &step * &k2)
                + (-3544.0 / 2565.0 * &step * &k3)
                + (1859.0 / 4104.0 * &step * &k4)
                + (-11.0 / 40.0 * &step * &k5);
            let k6_unchecked = f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?;
            let k6_rank = k6_unchecked.size().len();
            if k6_rank != 1 {
                anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", k6_rank);
            }

            k6_unchecked
        };

        let next_y4: Tensor = &y
            + &step
                * ((25.0 / 216.0 * &k1)
                    + (1408.0 / 2565.0 * &k3)
                    + (2197.0 / 4104.0 * &k4)
                    + (-1.0 / 5.0 * &k5));
        let next_y5: Tensor = &y
            + &step
                * ((16.0 / 135.0 * &k1)
                    + (6656.0 / 12825.0 * &k3)
                    + (28561.0 / 56430.0 * &k4)
                    + (-9.0 / 50.0 * &k5)
                    + (2.0 / 55.0 * &k6));

        let d = (&next_y4 - &next_y5).abs();
        let e = next_y5.abs() * rtol + atol;

        let alpha_tensor = (e / d).sqrt().min();
        let condition = &safety_factor_tensor * &alpha_tensor;

        let condition_met = condition.lt(1.0);
        let condition_met_bool: bool = condition_met == Tensor::from(true);

        if condition_met_bool {
            step = &step * &condition;
            if step.double_value(&[]) < min_step {
                return Err(anyhow!("Required step is smaller than minimal step"));
            }
        } else {
            y = next_y4;
            x = &x + &step;
            all_x.push(x.copy());
            all_y.push(y.copy());

            let new_step = &step * &condition;
            let max_step = &step * 5.0;
            step = new_step.fmin(&max_step);
        }
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using first-order Rosenbrock method (Row1)
fn solve_row1(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        let mut current_step = step;
        if remaining.double_value(&[]) < step {
            current_step = remaining.double_value(&[]);
        }

        let x_prev = x.copy();
        let y_prev = y.copy().squeeze();

        let jacobian = compute_jacobian(
            |y| {
                f.forward_ts(&[x_prev.squeeze().copy(), y.copy()])
                    .unwrap()
                    .squeeze()
            },
            &y_prev,
        );
        let f_current = f
            .forward_ts(&[x_prev.squeeze().copy(), y_prev.copy()])?;
        let f_current_rank = f_current.size().len();
        if f_current_rank != 1 {
            anyhow::bail!("Derivative CModule returned tensor of bad rank {}.", f_current_rank);
        }

        let n = jacobian.size()[0];
        let eye = Tensor::eye(n, (tch::Kind::Float, jacobian.device()));
        let step_j = current_step * &jacobian;
        let inv_matrix = (eye - step_j).inverse();

        let delta_y = inv_matrix.matmul(&f_current);
        let y_next = y_prev + current_step * delta_y;

        x = &x_prev + current_step;
        y = y_next.unsqueeze(0);

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Computes the Jacobian matrix of a function f at point x
fn compute_jacobian<F>(f: F, x: &Tensor) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    assert_eq!(x.dim(), 1, "x must be 1-dimensional");
    let mut x_with_grad = x.detach().copy().set_requires_grad(true);
    let y = f(&x_with_grad);
    assert_eq!(y.dim(), 1, "y must be 1-dimensional");

    let y_size = y.size()[0];
    let mut grads = Vec::new();

    for i in 0..y_size {
        let yi = y.i(i);
        //yi.backward();
        //let grad = x_with_grad.grad().copy();
        let grad = Tensor::run_backward(&[yi], &[&x_with_grad], true, false)[0].copy();
        grads.push(grad.unsqueeze(0));
        x_with_grad.zero_grad();
    }

    Tensor::cat(&grads, 0)
}
