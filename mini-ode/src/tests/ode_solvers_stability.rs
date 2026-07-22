use crate::Solver;
use crate::optimizers;
use std::sync::Arc;
use tch::{CModule, Tensor};

fn solution_is_bounded(ys: &Tensor, bound: f64) -> bool {
    if ys.isfinite().f_all().unwrap().f_int64_value(&[]).unwrap() == 0 {
        return false;
    }
    let max_abs = ys.abs().max().double_value(&[]);
    max_abs <= bound
}

fn make_linear_model(lambda: f64, y0_len: usize) -> CModule {
    let y0 = Tensor::zeros([y0_len as i64], (tch::Kind::Float, tch::Device::Cpu));
    let mut closure = move |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        vec![y.g_mul_scalar(lambda)]
    };
    CModule::create_by_tracing(
        "linear_ode",
        "forward",
        &[Tensor::from(0.0f64), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap()
}

fn make_van_der_pol_model(mu: f64) -> CModule {
    let y0 = Tensor::from_slice(&[2.0f64, 0.0f64]);
    let one = Tensor::from(1.0f64);
    let mut closure = move |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];

        let y0c = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1.shallow_clone();
        let dy1 = ((&one - (&y0c * &y0c)).g_mul_scalar(mu) * &y1) - &y0c;

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    CModule::create_by_tracing(
        "vanderpol",
        "forward",
        &[Tensor::from(0.0f64), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap()
}

#[test]
fn test_euler_stability_inside_region() {
    let lambda: f64 = -10.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 1.0;

    let solver = Solver::Euler { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "Euler should be stable for λh = -1.0 (inside [-2, 0])"
    );
}

#[test]
fn test_euler_stability_near_limit_inside() {
    let lambda: f64 = -19.5;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 1.0;

    let solver = Solver::Euler { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "Euler should be stable at λh = -1.95 (inside [-2, 0])"
    );
}

#[test]
fn test_euler_instability_near_limit_outside() {
    let lambda: f64 = -20.5;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 10.0;

    let solver = Solver::Euler { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        !solution_is_bounded(&ys, bound),
        "Euler should blow up at λh = -2.05 (outside [-2, 0])"
    );
}

#[test]
fn test_euler_instability_far_outside() {
    let lambda: f64 = -500.0;
    let step = 0.05;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 1e10;

    let solver = Solver::Euler { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();

    assert!(
        !solution_is_bounded(&ys, bound),
        "Euler should blow up for λh = -25 (far outside [-2, 0])"
    );
}

#[test]
fn test_euler_instability_van_der_pol_stiff() {
    let mu: f64 = 1000.0;
    let step = 0.05;
    let model = make_van_der_pol_model(mu);
    let y0 = Tensor::from_slice(&[2.0f64, 0.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 1e10;

    let solver = Solver::Euler { step };
    match solver.solve(model, x_span, y0) {
        Err(msg) => assert!(msg.to_string().contains("Non-finite")),
        Ok((_xs, ys)) => assert!(
            !solution_is_bounded(&ys, bound),
            "Euler should blow up on stiff van der Pol (μ={mu}, h={step})"
        ),
    };
}

#[test]
fn test_rk4_stability_inside_region() {
    let lambda: f64 = -10.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;

    let solver = Solver::RK4 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "RK4 should be stable for λh = -1.0 (inside ≈[-2.785, 0])"
    );
}

#[test]
fn test_rk4_stability_near_limit_inside() {
    let lambda: f64 = -27.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 5.0;

    let solver = Solver::RK4 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "RK4 should be stable at λh = -2.7 (inside ≈[-2.785, 0])"
    );
}

#[test]
fn test_rk4_instability_near_limit_outside() {
    let lambda: f64 = -28.5;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 5.0;

    let solver = Solver::RK4 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        !solution_is_bounded(&ys, bound),
        "RK4 should blow up at λh = -2.85 (outside ≈[-2.785, 0])"
    );
}

#[test]
fn test_rk4_instability_far_outside() {
    let lambda: f64 = -500.0;
    let step = 0.05;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 10.0;

    let solver = Solver::RK4 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();

    assert!(
        !solution_is_bounded(&ys, bound),
        "RK4 should blow up for λh = -25 (far outside ≈[-2.785, 0])"
    );
}

#[test]
fn test_rk4_stability_between_euler_and_rk4_limits() {
    let lambda: f64 = -24.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;

    let solver = Solver::RK4 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "RK4 should remain stable for λh = -2.4 (limit is ≈-2.785)"
    );
}

#[test]
fn test_rk4_instability_van_der_pol_stiff() {
    let mu: f64 = 1000.0;
    let step = 0.05;
    let model = make_van_der_pol_model(mu);
    let y0 = Tensor::from_slice(&[2.0f64, 0.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 50.0;

    let solver = Solver::RK4 { step };
    match solver.solve(model, x_span, y0) {
        Err(msg) => assert!(msg.to_string().contains("Non-finite")),
        Ok((_xs, ys)) => assert!(
            !solution_is_bounded(&ys, bound),
            "RK4 should blow up on stiff van der Pol (μ={mu}, h={step})"
        ),
    };
}

#[test]
fn test_implicit_euler_stability_inside_region() {
    let lambda: f64 = -10.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::ImplicitEuler { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "Implicit Euler should be stable for λh = -1.0 (A-stable)"
    );
}

#[test]
fn test_implicit_euler_stability_far_outside() {
    let lambda: f64 = -500.0;
    let step = 0.05;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 10.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::ImplicitEuler { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "Implicit Euler should remain stable for λh = -25 (A-stable)"
    );
}

#[test]
fn test_implicit_euler_stability_extreme_stiffness() {
    let lambda: f64 = -100_000.0;
    let step = 0.01;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 5.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::ImplicitEuler { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "Implicit Euler should remain stable for λh = -1000 (A-stable)"
    );
}

#[test]
fn test_implicit_euler_stability_van_der_pol_stiff() {
    let mu: f64 = 1000.0;
    let step = 0.05;
    let model = make_van_der_pol_model(mu);
    let y0 = Tensor::from_slice(&[2.0f64, 0.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 50.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::ImplicitEuler { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "Implicit Euler should remain stable on stiff van der Pol"
    );
}

#[test]
fn test_glrk4_stability_inside_region() {
    let lambda: f64 = -10.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::GLRK4 { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "GLRK4 should be stable for λh = -1.0 (A-stable)"
    );
}

#[test]
fn test_glrk4_stability_far_outside() {
    let lambda: f64 = -500.0;
    let step = 0.05;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 10.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::GLRK4 { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "GLRK4 should remain stable for λh = -25 (A-stable)"
    );
}

#[test]
fn test_glrk4_stability_extreme_stiffness() {
    let lambda: f64 = -100_000.0;
    let step = 0.01;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 5.0;
    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));

    let solver = Solver::GLRK4 { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "GLRK4 should remain stable for λh = -1000 (A-stable)"
    );
}

#[test]
fn test_glrk4_stability_van_der_pol_stiff() {
    let mu: f64 = 25.0;
    let step = 0.05;
    let model = make_van_der_pol_model(mu);
    let y0 = Tensor::from_slice(&[2.0f64, 0.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 50.0;
    let optimizer = Arc::new(optimizers::Halley::new(200, Some(1e-9), Some(1e-9)));

    let solver = Solver::GLRK4 { step, optimizer };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "GLRK4 should remain stable on stiff van der Pol"
    );
}

#[test]
fn test_row1_stability_inside_region() {
    let lambda: f64 = -10.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;

    let solver = Solver::ROW1 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "ROW1 should be stable for λh = -1.0 (A-stable)"
    );
}

#[test]
fn test_row1_stability_far_outside() {
    let lambda: f64 = -500.0;
    let step = 0.05;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 10.0;

    let solver = Solver::ROW1 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "ROW1 should remain stable for λh = -25 (A-stable)"
    );
}

#[test]
fn test_row1_stability_extreme_stiffness() {
    let lambda: f64 = -100_000.0;
    let step = 0.01;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 1.0f64);
    let bound = 5.0;

    let solver = Solver::ROW1 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "ROW1 should remain stable for λh = -1000 (A-stable)"
    );
}

#[test]
fn test_row1_stability_van_der_pol_stiff() {
    let mu: f64 = 1000.0;
    let step = 0.05;
    let model = make_van_der_pol_model(mu);
    let y0 = Tensor::from_slice(&[2.0f64, 0.0f64]);
    let x_span = (0.0f64, 5.0f64);
    let bound = 50.0;

    let solver = Solver::ROW1 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "ROW1 should remain stable on stiff van der Pol"
    );
}

#[test]
fn test_rkf45_stability_inside_region() {
    let lambda: f64 = -10.0;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;

    let solver = Solver::RKF45 {
        rtol: 1e-6,
        atol: 1e-9,
        min_step: 1e-10,
        safety_factor: 0.9,
    };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "RKF45 should be stable for λh = -1.0"
    );
}

#[test]
fn test_row1_stability_between_euler_and_rk4_limits() {
    let lambda: f64 = -24.0;
    let step = 0.1;
    let model = make_linear_model(lambda, 1);
    let y0 = Tensor::from_slice(&[1.0f64]);
    let x_span = (0.0f64, 2.0f64);
    let bound = 2.0;

    let solver = Solver::ROW1 { step };
    let (_xs, ys) = solver.solve(model, x_span, y0).unwrap();
    assert!(
        solution_is_bounded(&ys, bound),
        "ROW1 should remain stable for λh = -2.4 (A-stable)"
    );
}
