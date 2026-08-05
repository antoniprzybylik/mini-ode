use anyhow::anyhow;
use tch::Tensor;

use crate::utils::validation;
use crate::utils::warnings::warn;

/// One RKF45 step
fn rkf45_step(
    f: &tch::CModule,
    x: f64,
    y: &Tensor,
    step: f64,
    device: tch::Device,
    kind: tch::Kind,
    y0_length: i64,
) -> anyhow::Result<(Tensor, Tensor)> {
    // Stage k1
    let k1 = f.forward_ts(&[Tensor::from(x).to_kind(kind).to_device(device), y.copy()])?;
    validation::validate_finite_tensor(&k1, "RKF45 stage k1")?;

    let k1_size = k1.size();
    if k1_size.len() != 1 || k1_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k2
    let x_step = x + 0.25 * step;
    let y_step: Tensor = y + 0.25 * &step * &k1;
    validation::validate_finite_tensor(&y_step, "RKF45 intermediate state for k2")?;

    let k2 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validation::validate_finite_tensor(&k2, "RKF45 stage k2")?;

    let k2_size = k2.size();
    if k2_size.len() != 1 || k2_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k3
    let x_step = x + 0.375 * step;
    let y_step: Tensor = y + (0.09375 * &step * &k1) + (0.28125 * &step * &k2);
    validation::validate_finite_tensor(&y_step, "RKF45 intermediate state for k3")?;

    let k3 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validation::validate_finite_tensor(&k3, "RKF45 stage k3")?;

    let k3_size = k3.size();
    if k3_size.len() != 1 || k3_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k4
    let x_step = x + (12.0 / 13.0) * step;
    let y_step: Tensor = y
        + (1932.0 / 2197.0 * &step * &k1)
        + (-7200.0 / 2197.0 * &step * &k2)
        + (7296.0 / 2197.0 * &step * &k3);
    validation::validate_finite_tensor(&y_step, "RKF45 intermediate state for k4")?;

    let k4 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validation::validate_finite_tensor(&k4, "RKF45 stage k4")?;

    let k4_size = k4.size();
    if k4_size.len() != 1 || k4_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k5
    let x_step = x + step;
    let y_step: Tensor = y
        + (439.0 / 216.0 * &step * &k1)
        + (-8.0 * &step * &k2)
        + (3680.0 / 513.0 * &step * &k3)
        + (-845.0 / 4104.0 * &step * &k4);
    validation::validate_finite_tensor(&y_step, "RKF45 intermediate state for k5")?;

    let k5 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validation::validate_finite_tensor(&k5, "RKF45 stage k5")?;

    let k5_size = k5.size();
    if k5_size.len() != 1 || k5_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Stage k6
    let x_step = x + 0.5 * step;
    let y_step: Tensor = y
        + (-8.0 / 27.0 * &step * &k1)
        + (2.0 * &step * &k2)
        + (-3544.0 / 2565.0 * &step * &k3)
        + (1859.0 / 4104.0 * &step * &k4)
        + (-11.0 / 40.0 * &step * &k5);
    validation::validate_finite_tensor(&y_step, "RKF45 intermediate state for k6")?;

    let k6 = f.forward_ts(&[Tensor::from(x_step).to_kind(kind).to_device(device), y_step])?;
    validation::validate_finite_tensor(&k6, "RKF45 stage k6")?;

    let k6_size = k6.size();
    if k6_size.len() != 1 || k6_size[0] != y0_length {
        anyhow::bail!("Derivative CModule returned tensor of bad shape in RKF45");
    }

    // Compute 4th and 5th order solutions
    let next_y4: Tensor = y + step
        * ((25.0 / 216.0 * &k1)
            + (1408.0 / 2565.0 * &k3)
            + (2197.0 / 4104.0 * &k4)
            + (-1.0 / 5.0 * &k5));

    let next_y5: Tensor = y + step
        * ((16.0 / 135.0 * &k1)
            + (6656.0 / 12825.0 * &k3)
            + (28561.0 / 56430.0 * &k4)
            + (-9.0 / 50.0 * &k5)
            + (2.0 / 55.0 * &k6));

    Ok((next_y4, next_y5))
}

/// Solves ODE using Runge-Kutta-Fehlberg 45 adaptive method
pub(crate) fn solve_rkf45(
    f: tch::CModule,
    x_span: (f64, f64),
    y0: Tensor,
    rtol: f64,
    atol: f64,
    min_step: f64,
    safety_factor: f64,
) -> anyhow::Result<(Tensor, Tensor)> {
    let device = y0.device();
    let kind = y0.kind();

    let x_start = x_span.0;
    let x_end = x_span.1;

    let mut x = x_start;
    let mut y = y0.copy();

    let mut all_x = vec![x];
    let mut all_y = vec![y.copy()];

    let mut step = (x_end - x_start) * 0.1;

    let mut consecutive_rejections: u32 = 0;
    let mut total_rejections: u32 = 0;
    let mut total_accepted: u32 = 0;

    let mut warned_consec_rej = false;
    let mut warned_total_rej = false;
    let mut warned_tiny_step = false;

    const MAX_GROWTH: f64 = 5.;

    while x < x_end {
        let (next_y4, next_y5) = rkf45_step(&f, x, &y, step, device, kind, y0.size()[0])?;

        // Compute error estimate
        let d = (&next_y4 - &next_y5).f_abs()?;
        validation::validate_finite_tensor(&d, "RKF45 error estimate difference")?;

        let e = next_y5.f_abs()? * rtol + atol;
        validation::validate_finite_tensor(&e, "RKF45 error tolerance combination")?;

        // Compute step size adjustment
        let alpha = (e / d)
            .f_pow_tensor_scalar(0.2)?
            .f_min()?
            .f_double_value(&[])?;

        let condition = (safety_factor * alpha).clamp(0f64, MAX_GROWTH);

        if condition < 1f64 {
            // Step rejected - shrink and retry
            consecutive_rejections += 1;
            total_rejections += 1;

            // Warning for consecutive rejections
            if !warned_consec_rej && consecutive_rejections >= 20 {
                warn!(
                    "RKF45: {} consecutive rejected steps at x={:.3e}, step={:.3e}; problem may be stiff",
                    consecutive_rejections, x, step
                );
                warned_consec_rej = true;
            }

            // Warning for many total rejections
            if !warned_total_rej && total_rejections >= 1000 {
                warn!(
                    "RKF45: {} total rejected steps ({} accepted) at x={:.3e}; integration is inefficient",
                    total_rejections, total_accepted, x
                );
                warned_total_rej = true;
            }

            // Warning for very small step approaching min_step
            if !warned_tiny_step && step < min_step * 10.0 {
                warn!(
                    "RKF45: required very small step {:.3e} at x={:.3e} (min_step={:.3e}); solution may be inaccurate or problem is stiff",
                    step, x, min_step
                );
                warned_tiny_step = true;
            }

            step = step * condition;
            validation::validate_finite_scalar(step, "RKF45 reduced step size")?;

            // Warning for step below min_step
            if step < min_step {
                return Err(anyhow!("Required step is smaller than minimal step"));
            }
        } else {
            // Accept the step
            consecutive_rejections = 0;
            total_accepted += 1;

            // At last step, special handling
            let remaining = x_end - x;
            if remaining < step {
                step = remaining;
                let (_next_y4, next_y5) = rkf45_step(&f, x, &y, step, device, kind, y0.size()[0])?;
                y = next_y5;
                x = x_end;
                all_x.push(x);
                all_y.push(y.copy());
                break;
            }

            y = next_y5;
            x = &x + &step;

            // Validate accepted state
            validation::validate_finite_tensor(&y, "RKF45 accepted state (NaN/Inf)")?;
            validation::validate_finite_scalar(x, "RKF45 updated integration variable")?;

            all_x.push(x);
            all_y.push(y.copy());

            step = step * condition;
            validation::validate_finite_scalar(step, "RKF45 next step size")?;
        }
    }

    // Final efficiency summary
    if total_rejections > total_accepted * 2 && total_accepted > 0 {
        warn!(
            "RKF45: integration completed with {} rejected and {} accepted steps; consider relaxing tolerances or using an implicit solver for stiff problems",
            total_rejections, total_accepted
        );
    }

    Ok((
        Tensor::f_from_slice(&all_x)?
            .to_kind(kind)
            .to_device(device),
        Tensor::f_stack(&all_y, 0)?,
    ))
}
