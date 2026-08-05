use tch::Tensor;

use crate::utils::differentiation;
use crate::utils::validation;
use crate::utils::warnings::warn;

/// Solves ODE using first-order Rosenbrock method (Row1)
pub(crate) fn solve_row1(
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

    let mut warned_large_matrix = false;
    let mut warned_large_inverse = false;
    let mut warned_large_norm = false;
    let mut warned_many_steps = false;

    let n_steps = ((x_end - x_start) / step).ceil() as usize;
    for step_no in 0..n_steps {
        let current_step = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end - (x_start + step_no as f64 * step)
        } else {
            step
        };

        let x_prev = x;
        let y_prev = y.copy();

        // Compute Jacobian
        let jacobian = differentiation::compute_jacobian(
            |y| {
                f.forward_ts(&[
                    Tensor::from(x_prev).to_kind(kind).to_device(device),
                    y.copy(),
                ])
                .unwrap()
            },
            &y_prev,
        )?;

        // Validate Jacobian is finite
        validation::validate_finite_tensor(&jacobian, "Jacobian matrix in ROW1 (NaN/Inf)")?;

        // Evaluate function at current point
        let f_current = f.forward_ts(&[
            Tensor::from(x_prev).to_kind(kind).to_device(device),
            y_prev.copy(),
        ])?;

        validation::validate_finite_tensor(&f_current, "derivative function output in ROW1")?;

        let f_current_size = f_current.size();
        let f_current_rank = f_current_size.len();
        if f_current_rank != 1 {
            anyhow::bail!(
                "Derivative CModule returned tensor of bad rank {}.",
                f_current_rank
            );
        }
        if f_current_size[0] != y0.size()[0] {
            anyhow::bail!(
                "Derivative CModule returned vector of bad length {}.",
                f_current_size[0]
            );
        }

        // Compute (I - h*J)^(-1) * f
        let n = jacobian.size()[0];
        let eye = Tensor::f_eye(n, (jacobian.kind(), jacobian.device()))?;
        let step_j = current_step * &jacobian;
        let matrix_to_invert = eye - step_j;

        // Warn about ill-conditioning before inversion (one-shot)
        let matrix_norm = matrix_to_invert.f_norm()?.f_double_value(&[])?;
        if !warned_large_matrix && matrix_norm > 1e12 {
            warn!(
                "ROW1: linear system matrix has large norm {:.3e} at x={:.3e}; solution may be unstable",
                matrix_norm, x_prev
            );
            warned_large_matrix = true;
        }

        let inv_matrix = matrix_to_invert.f_inverse()?;

        validation::validate_finite_tensor(&inv_matrix, "inverse matrix (I - h*J)^(-1) in ROW1")?;

        // Warn about inverse magnitude (one-shot)
        let inv_norm = inv_matrix.f_norm()?.f_double_value(&[])?;
        if !warned_large_inverse && inv_norm > 1e10 {
            warn!(
                "ROW1: inverse matrix has large norm {:.3e} at x={:.3e}; Jacobian may be ill-conditioned",
                inv_norm, x_prev
            );
            warned_large_inverse = true;
        }

        let delta_y = inv_matrix.f_matmul(&f_current)?;
        validation::validate_finite_tensor(&delta_y, "Newton correction step in ROW1")?;

        let y_next = y_prev + current_step * delta_y;

        // Critical validation after ROW1 update
        validation::validate_finite_tensor(&y_next, "state after ROW1 update (NaN/Inf)")?;

        x = if x_start + (step_no + 1) as f64 * step > x_end {
            x_end
        } else {
            x_start + (step_no + 1) as f64 * step
        };
        validation::validate_finite_scalar(x, "ROW1 updated integration variable")?;

        y = y_next.detach().copy();

        all_x.push(x);
        all_y.push(y.copy());

        let y_norm = y.f_norm()?.f_double_value(&[])?;

        if !warned_large_norm && y_norm > 1e10 {
            warn!(
                "ROW1: solution norm exceeded {:.1e} at x={:.3e}; the solution may be diverging.",
                1e10, x
            );
            warned_large_norm = true;
        }

        let step_count = step_no + 1;
        if !warned_many_steps && step_count >= 100_000 {
            warn!(
                "ROW1: reached {} steps; consider increasing step size",
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
