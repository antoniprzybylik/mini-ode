use anyhow::anyhow;
use tch::IndexOp;
use tch::Tensor;

use crate::optimizers;
use crate::utils::validation;
use crate::utils::warnings::warn;

/// Solves ODE using Gauss-Legendre-Runge-Kutta 4th order method
pub(crate) fn solve_glrk4(
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
    let y_length = y.size()[0];

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

        let k = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;
        validation::validate_finite_tensor(&k, "GLRK4 initial derivative")?;

        let k_size = k.size();
        if k_size.len() != 1 || k_size[0] != y0.size()[0] {
            anyhow::bail!("Derivative CModule returned tensor of wrong shape in GLRK4");
        }

        const C1: f64 = 0.2113248654f64;
        const C2: f64 = 0.7886751346f64;
        const A11: f64 = 0.25;
        const A12: f64 = -0.03867513459f64;
        const A21: f64 = 0.5386751346f64;
        const A22: f64 = 0.25;

        // Initial guess for k1, k2
        let first_k1k2_guess = Tensor::f_cat(
            &[
                f.forward_ts(&[
                    Tensor::from(x + C1 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + C1 * current_step * &k,
                ])?,
                f.forward_ts(&[
                    Tensor::from(x + C2 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + C2 * current_step * &k,
                ])?,
            ],
            0,
        )?;

        // Define loss function for optimization
        let loss_fn = |k1k2_guess: &Tensor| {
            let diff1 = k1k2_guess.i(0..y_length)
                - f.forward_ts(&[
                    Tensor::from(x + C1 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + (A11 * k1k2_guess.i(0..y_length)
                        + A12 * k1k2_guess.i(y_length..2 * y_length))
                        * current_step,
                ])
                .unwrap();
            let diff2 = k1k2_guess.i(y_length..2 * y_length)
                - f.forward_ts(&[
                    Tensor::from(x + C2 * current_step)
                        .to_kind(kind)
                        .to_device(device),
                    &y + (A21 * k1k2_guess.i(0..y_length)
                        + A22 * k1k2_guess.i(y_length..2 * y_length))
                        * current_step,
                ])
                .unwrap();
            diff1.dot(&diff1) + diff2.dot(&diff2)
        };

        // Run optimizer
        let k1k2 = optimizer
            .optimize(&loss_fn, &first_k1k2_guess)
            .map_err(|err| anyhow!(format!("GLRK4 optimizer failed with: {}", err)))?;

        // Validate optimizer output
        validation::validate_finite_tensor(
            &k1k2,
            "GLRK4 stage coefficients after optimization (NaN/Inf)",
        )?;

        // Compute final state update
        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        y = &y
            + current_step
                * (0.5 * k1k2.f_i(0..y_length)? + 0.5 * k1k2.f_i(y_length..2 * y_length)?);

        // Validate final state
        validation::validate_finite_tensor(&y, "state after GLRK4 update (NaN/Inf propagating)")?;

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "GLRK4: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "GLRK4: reached {} steps; consider increasing step size",
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
