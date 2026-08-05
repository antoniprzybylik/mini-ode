use tch::Tensor;

use crate::utils::validation;
use crate::utils::warnings::warn;

/// Solves ODE using Euler method
pub(crate) fn solve_euler(
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

        let dy = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;

        validation::validate_finite_tensor(&dy, "derivative from f(x, y) in Euler step")?;

        let dy_size = dy.size();
        let dy_rank = dy_size.len();
        if dy_rank != 1 {
            anyhow::bail!(
                "Derivative CModule returned tensor of bad rank {}.",
                dy_rank
            );
        }
        if dy_size[0] != y0.size()[0] {
            anyhow::bail!(
                "Derivative CModule returned vector of bad length {}.",
                dy_size[0]
            );
        }

        // Compute next state
        y = &y + current_step * &dy;

        // Critical: validate new state is finite before proceeding
        validation::validate_finite_tensor(&y, "state after Euler update (NaN/Inf propagating)")?;

        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };

        // Validate x remains finite
        let x_tensor = Tensor::from(x).to_kind(kind).to_device(device);
        validation::validate_finite_tensor(&x_tensor, "integration variable x in Euler step")?;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "Euler: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "Euler: reached {} steps; consider increasing step size or switching to a higher-order solver",
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
