use anyhow::anyhow;
use tch::Tensor;

use crate::optimizers;
use crate::utils::validation;
use crate::utils::warnings::warn;

/// Solves ODE using Implicit Euler method with gradient descent optimization
pub(crate) fn solve_implicit_euler(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    step: f64,
    optimizer: &dyn optimizers::Optimizer,
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

        let x_next = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        let y_prev = y.copy();

        // Create derivative function for current x
        let f_next_fn = |y_next: &Tensor| {
            let f_next = f
                .forward_ts(&[
                    Tensor::from(x_next).to_kind(kind).to_device(device),
                    y_next.copy(),
                ])
                .unwrap();
            let y_pred = &y_prev + current_step * &f_next;
            (y_next - &y_pred).pow_tensor_scalar(2).sum(y_next.kind())
        };

        // Initial guess based on explicit Euler
        let initial_guess = &y_prev.detach()
            + current_step
                * f.forward_ts(&[&Tensor::from(x).to_kind(kind).to_device(device), &y_prev])?;

        // Run optimizer (may fail gracefully internally)
        let y_next = optimizer
            .optimize(&f_next_fn, &initial_guess)
            .map_err(|err| anyhow!(format!("Implicit solver optimizer failed with: {}", err)))?;

        // Critical: validate optimizer output before accepting
        validation::validate_finite_tensor(
            &y_next,
            "state after implicit solver optimization (NaN/Inf)",
        )?;

        y = y_next.copy();
        x = x_next;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "ImplicitEuler: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "ImplicitEuler: reached {} steps; consider increasing step size",
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
