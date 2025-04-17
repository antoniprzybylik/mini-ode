use anyhow::anyhow;
use tch::IndexOp;
use tch::Tensor;

mod optimizers;

pub use optimizers::BFGS;
pub use optimizers::CG;
pub use optimizers::Optimizer;

/// Solves ODE using Euler method
pub fn solve_euler(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    if x_span.size() != [2] {
        return Err(anyhow!("x_span must be of shape [2]"));
    }
    if y0.size().len() != 1 {
        return Err(anyhow!("y0 must be a one-dimensional tensor"));
    }
    if step.size().len() != 0 {
        return Err(anyhow!("step must be a zero-dimensional tensor"));
    }

    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.lt_tensor(&current_step) == Tensor::from_slice(&[true]) {
            current_step = remaining.copy();
        }

        let dy = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;
        y = &y + &current_step * &dy;
        x = &x + &current_step;

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Runge-Kutta 4th order method
pub fn solve_rk4(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    if x_span.size() != [2] {
        return Err(anyhow!("x_span must be of shape [2]"));
    }
    if y0.size().len() != 1 {
        return Err(anyhow!("y0 must be a one-dimensional tensor"));
    }
    if step.size().len() != 0 {
        return Err(anyhow!("step must be a zero-dimensional tensor"));
    }

    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.lt_tensor(&current_step) == Tensor::from_slice(&[true]) {
            current_step = remaining.copy();
        }

        let k1 = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;

        let x_half: Tensor = &x + 0.5 * &current_step;
        let y_half: Tensor = &y + 0.5 * &current_step * &k1;
        let k2 = f.forward_ts(&[x_half.squeeze(), y_half.squeeze()])?;

        let x_half_again: Tensor = &x + 0.5 * &current_step;
        let y_half_again: Tensor = &y + 0.5 * &current_step * &k2;
        let k3 = f.forward_ts(&[x_half_again.squeeze(), y_half_again.squeeze()])?;

        let x_full = &x + &current_step;
        let y_full = &y + &current_step * &k3;
        let k4 = f.forward_ts(&[x_full.squeeze(), y_full.squeeze()])?;

        let step_div_6 = &current_step / 6.0;
        let y_next = &y + &step_div_6 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        x = &x + &current_step;
        y = y_next;

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Implicit Euler method with gradient descent optimization
pub fn solve_implicit_euler(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: Tensor,
    optimizer: &dyn Optimizer,
) -> anyhow::Result<(Tensor, Tensor)> {
    if x_span.size() != [2] {
        return Err(anyhow!("x_span must be of shape [2]"));
    }
    if y0.size().len() != 1 {
        return Err(anyhow!("y0 must be a one-dimensional tensor"));
    }
    if step.size().len() != 0 {
        return Err(anyhow!("step must be a zero-dimensional tensor"));
    }

    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.lt_tensor(&current_step) == Tensor::from_slice(&[true]) {
            current_step = remaining.copy();
        }

        let x_next = &x + &current_step;
        let y_prev = y.copy();

        let y_next = optimizer.optimize(
            &|y_next: &Tensor| {
                let f_next = f
                    .forward_ts(&[x_next.squeeze().copy(), y_next.squeeze().copy()])
                    .unwrap();
                let y_pred = &y_prev.squeeze() + &current_step * &f_next;
                (y_next - &y_pred).pow_tensor_scalar(2).sum(y_next.kind())
            },
            &(&y_prev.detach().squeeze()
                + &current_step * f.forward_ts(&[&x.squeeze(), &y_prev.squeeze()])?),
        );

        y = y_next.unsqueeze(0);
        x = x_next.copy();

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Gauss-Legendre-Runge-Kutta 4th order method
pub fn solve_glrk4(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: Tensor,
    optimizer: &dyn Optimizer,
) -> anyhow::Result<(Tensor, Tensor)> {
    if x_span.size() != [2] {
        return Err(anyhow!("x_span must be of shape [2]"));
    }
    if y0.size().len() != 1 {
        return Err(anyhow!("y0 must be a one-dimensional tensor"));
    }
    if step.size().len() != 0 {
        return Err(anyhow!("step must be a zero-dimensional tensor"));
    }

    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    let mut current_step = step;
    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        if remaining.lt_tensor(&current_step) == Tensor::from_slice(&[true]) {
            current_step = remaining.copy();
        }

        let k = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;

        const c1: f64 = 0.2113248654f64;
        const c2: f64 = 0.7886751346f64;
        const a11: f64 = 0.25;
        const a12: f64 = -0.03867513459f64;
        const a21: f64 = 0.5386751346f64;
        const a22: f64 = 0.25;

        let first_k1k2_guess = Tensor::cat(
            &[
                f.forward_ts(&[
                    &x.squeeze() + c1 * &current_step,
                    &y.squeeze() + c1 * &current_step * &k,
                ])?,
                f.forward_ts(&[
                    &x.squeeze() + c2 * &current_step,
                    &y.squeeze() + c2 * &current_step * &k,
                ])?,
            ],
            0,
        );
        let k1k2 = optimizer.optimize(
            &|k1k2_guess| {
                let diff1 = k1k2_guess.i((0..=1,))
                    - f.forward_ts(&[
                        &x.squeeze() + c1 * &current_step,
                        &y.squeeze()
                            + (a11 * k1k2_guess.i((0..=1)) + a12 * k1k2_guess.i((2..=3)))
                                * &current_step,
                    ])
                    .unwrap();
                let diff2 = k1k2_guess.i((2..=3))
                    - f.forward_ts(&[
                        &x.squeeze() + c2 * &current_step,
                        &y.squeeze()
                            + (a21 * k1k2_guess.i((0..=1)) + a22 * k1k2_guess.i((2..=3)))
                                * &current_step,
                    ])
                    .unwrap();

                diff1.dot(&diff1) + diff2.dot(&diff2)
            },
            &first_k1k2_guess,
        );
        assert!(k1k2.size().len() == 1);
        assert!(k1k2.size()[0] == 4);

        x = &x + &current_step;
        y = &y + &current_step * (0.5 * k1k2.i((0..=1)) + 0.5 * k1k2.i((2..=3)));

        all_x.push(x.copy());
        all_y.push(y.copy());
    }

    Ok((Tensor::cat(&all_x, 0), Tensor::cat(&all_y, 0)))
}

/// Solves ODE using Runge-Kutta-Fehlberg 45 adaptive method
pub fn solve_rkf45(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    rtol: Tensor,
    atol: Tensor,
    min_step: Tensor,
    safety_factor: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    if x_span.size() != [2] {
        return Err(anyhow!("x_span must be of shape [2]"));
    }
    if y0.size().len() != 1 {
        return Err(anyhow!("y0 must be a one-dimensional tensor"));
    }
    if rtol.size().len() != 0 || atol.size().len() != 0 || min_step.size().len() != 0 {
        return Err(anyhow!("rtol, atol, and min_step must be scalar tensors"));
    }

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
        if remaining.lt_tensor(&step) == Tensor::from_slice(&[true]) {
            step = remaining.copy();
        }

        let k1 = f.forward_ts(&[x.squeeze().copy(), y.squeeze().copy()])?;

        let k2 = {
            let x_step: Tensor = &x + 0.25 * &step;
            let y_step: Tensor = &y + 0.25 * &step * &k1;
            f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?
        };

        let k3 = {
            let x_step: Tensor = &x + 0.375 * &step;
            let y_step: Tensor = &y + (0.09375 * &step * &k1) + (0.28125 * &step * &k2);
            f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?
        };

        let k4 = {
            let x_step: Tensor = &x + (12.0 / 13.0) * &step;
            let y_step: Tensor = &y
                + (1932.0 / 2197.0 * &step * &k1)
                + (-7200.0 / 2197.0 * &step * &k2)
                + (7296.0 / 2197.0 * &step * &k3);
            f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?
        };

        let k5 = {
            let x_step: Tensor = &x + &step;
            let y_step: Tensor = &y
                + (439.0 / 216.0 * &step * &k1)
                + (-8.0 * &step * &k2)
                + (3680.0 / 513.0 * &step * &k3)
                + (-845.0 / 4104.0 * &step * &k4);
            f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?
        };

        let k6 = {
            let x_step: Tensor = &x + 0.5 * &step;
            let y_step: Tensor = &y
                + (-8.0 / 27.0 * &step * &k1)
                + (2.0 * &step * &k2)
                + (-3544.0 / 2565.0 * &step * &k3)
                + (1859.0 / 4104.0 * &step * &k4)
                + (-11.0 / 40.0 * &step * &k5);
            f.forward_ts(&[x_step.squeeze(), y_step.squeeze()])?
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
        let e = next_y5.abs() * &rtol + &atol;

        let alpha_tensor = (e / d).sqrt().min();
        let condition = &safety_factor_tensor * &alpha_tensor;

        let condition_met = condition.lt(1.0);
        let condition_met_bool: bool = condition_met == Tensor::from(true);

        if condition_met_bool {
            step = &step * &condition;
            if step.lt_tensor(&min_step) == Tensor::from_slice(&[true]) {
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
pub fn solve_row1(
    f: tch::CModule,
    x_span: Tensor,
    y0: Tensor,
    step: Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    if x_span.size() != [2] {
        return Err(anyhow!("x_span must be of shape [2]"));
    }
    if y0.size().len() != 1 {
        return Err(anyhow!("y0 must be a one-dimensional tensor"));
    }
    if step.size().len() != 0 {
        return Err(anyhow!("step must be a zero-dimensional tensor"));
    }

    let x_start = x_span.i(0);
    let x_end = x_span.i(1);

    let mut x = x_start.unsqueeze(0);
    let mut y = y0.unsqueeze(0);

    let mut all_x = vec![x.copy()];
    let mut all_y = vec![y.copy()];

    while x.lt_tensor(&x_end) == Tensor::from_slice(&[true]) {
        let remaining = &x_end - &x.squeeze();
        let mut current_step = step.copy();
        if remaining.lt_tensor(&current_step) == Tensor::from_slice(&[true]) {
            current_step = remaining.copy();
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
            .forward_ts(&[x_prev.squeeze().copy(), y_prev.copy()])?
            .squeeze();

        let n = jacobian.size()[0];
        let eye = Tensor::eye(n, (tch::Kind::Float, jacobian.device()));
        let step_j = &current_step * &jacobian;
        let inv_matrix = (eye - step_j).inverse();

        let delta_y = inv_matrix.matmul(&f_current);
        let y_next = y_prev + &current_step.squeeze() * delta_y;

        x = &x_prev + &current_step;
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
