use tch::Tensor;

use crate::utils::validation;
use crate::utils::warnings::warn;

/// Solves ODE using Runge-Kutta 4th order method
pub(crate) fn solve_rk4(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        // Stage k1
        let k1 = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;
        validation::validate_finite_tensor(&k1, "RK4 stage k1")?;

        let k1_size = k1.size();
        if k1_size.len() != 1 || k1_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k1");
        }

        // Stage k2
        let x_half = x + 0.5 * current_step;
        let y_half: Tensor = &y + 0.5 * current_step * &k1;
        validation::validate_finite_tensor(&y_half, "RK4 intermediate state for k2")?;

        let k2 = f.forward_ts(&[Tensor::from(x_half).to_kind(kind).to_device(device), y_half])?;
        validation::validate_finite_tensor(&k2, "RK4 stage k2")?;

        let k2_size = k2.size();
        if k2_size.len() != 1 || k2_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k2");
        }

        // Stage k3
        let x_half_again = x + 0.5 * current_step;
        let y_half_again: Tensor = &y + 0.5 * current_step * &k2;
        validation::validate_finite_tensor(&y_half_again, "RK4 intermediate state for k3")?;

        let k3 = f.forward_ts(&[
            Tensor::from(x_half_again).to_kind(kind).to_device(device),
            y_half_again,
        ])?;
        validation::validate_finite_tensor(&k3, "RK4 stage k3")?;

        let k3_size = k3.size();
        if k3_size.len() != 1 || k3_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k3");
        }

        // Stage k4
        let x_full = x + current_step;
        let y_full = &y + current_step * &k3;
        validation::validate_finite_tensor(&y_full, "RK4 intermediate state for k4")?;

        let k4 = f.forward_ts(&[Tensor::from(x_full).to_kind(kind).to_device(device), y_full])?;
        validation::validate_finite_tensor(&k4, "RK4 stage k4")?;

        let k4_size = k4.size();
        if k4_size.len() != 1 || k4_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape at stage k4");
        }

        // Compute next state using weighted average of stages
        let step_div_6 = current_step / 6.0;
        let y_next = &y + step_div_6 * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        // Critical validation after full RK4 step
        validation::validate_finite_tensor(
            &y_next,
            "state after RK4 update (NaN/Inf propagating)",
        )?;

        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        y = y_next;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "RK4: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "RK4: reached {} steps; consider increasing step size or switching to an adaptive solver",
                step_count
            );
            warned_many_steps = true;
        }
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}
