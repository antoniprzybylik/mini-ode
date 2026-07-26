use crate::Solver;
use crate::optimizers;
use std::sync::Arc;
use tch::{CModule, Tensor};

fn ode_exact_solution(t: f64) -> (f64, f64) {
    assert!(t >= 0f64 && t <= 6.5f64);

    use super::ode_solving_data::ode_exact_solution_array::ODE_EXACT_SOLUTION_ARRAY;
    let ys = ODE_EXACT_SOLUTION_ARRAY;

    let idx = (t / 0.001f64) as usize;
    let res = t - (idx as f64) * 0.001f64;
    let interpolated_value = (
        ys[idx].0 * (1. - res) + ys[idx + 1].0 * res,
        ys[idx].1 * (1. - res) + ys[idx + 1].1 * res,
    );

    interpolated_value
}

#[test]
fn test_solver_euler_case1() {
    use super::ode_solving_data::euler_case1_data::EULER_CASE1_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&EULER_CASE1_DATA);

    let solver = Solver::Euler { step: 0.1 };

    let y0 = Tensor::from_slice(&[0.0f64, 2.0f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.5);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .collect::<Vec<f64>>(),
    );
    assert!((xs - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys - test_ys).abs().max().double_value(&[]) <= 8e-1);
}

#[test]
fn test_solver_euler_case2() {
    use super::ode_solving_data::euler_case2_data::EULER_CASE2_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&EULER_CASE2_DATA);

    let solver = Solver::Euler { step: 0.1 };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.5);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .collect::<Vec<f64>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 1e-4);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 0.5 && (exact_solution.1 - y[1]).abs() <= 0.5
            })
    );
}

#[test]
fn test_solver_euler_case3_interval_is_not_a_multiplicity_of_step() {
    use super::ode_solving_data::euler_case3_data::EULER_CASE3_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&EULER_CASE3_DATA);

    let solver = Solver::Euler { step: 0.1 };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.45);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..65)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .chain([6.45].into_iter())
            .collect::<Vec<f64>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 1e-4);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 0.5 && (exact_solution.1 - y[1]).abs() <= 0.5
            })
    );
}

#[test]
fn test_solver_rk4_case1() {
    use super::ode_solving_data::rk4_case1_data::RK4_CASE1_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&RK4_CASE1_DATA);

    let solver = Solver::RK4 { step: 0.1 };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.5);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .collect::<Vec<f64>>(),
    );
    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 8e-5);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 2e-3 && (exact_solution.1 - y[1]).abs() <= 2e-3
            })
    );
}

#[test]
fn test_solver_implicit_euler_case1() {
    use super::ode_solving_data::implicit_euler_case1_data::IMPLICIT_EULER_CASE1_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&IMPLICIT_EULER_CASE1_DATA);

    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));
    let solver = Solver::ImplicitEuler {
        step: 0.1,
        optimizer,
    };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.5);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .collect::<Vec<f64>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 8e-5);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 0.4 && (exact_solution.1 - y[1]).abs() <= 0.4
            })
    );
}

#[test]
fn test_solver_implicit_euler_case2_interval_is_not_a_multiplicity_of_step() {
    use super::ode_solving_data::implicit_euler_case2_data::IMPLICIT_EULER_CASE2_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&IMPLICIT_EULER_CASE2_DATA);

    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));
    let solver = Solver::ImplicitEuler {
        step: 0.1,
        optimizer,
    };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.45);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..65)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .chain([6.45].into_iter())
            .collect::<Vec<f64>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 8e-5);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 0.4 && (exact_solution.1 - y[1]).abs() <= 0.4
            })
    );
}

#[test]
fn test_solver_glrk4_case1() {
    use super::ode_solving_data::glrk4_case1_data::GLRK4_CASE1_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&GLRK4_CASE1_DATA);

    let optimizer = Arc::new(optimizers::CG::new(100, Some(1e-6), Some(1e-6)));
    let solver = Solver::GLRK4 {
        step: 0.1,
        optimizer,
    };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.5);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .collect::<Vec<f64>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 5e-5);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 4e-3 && (exact_solution.1 - y[1]).abs() <= 4e-3
            })
    );
}

#[test]
fn test_solver_rkf45_case1() {
    let solver = Solver::RKF45 {
        rtol: 1e-6,
        atol: 1e-8,
        min_step: 1e-10,
        safety_factor: 0.9,
    };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0, 6.5);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();

    // Check that the number of steps roughly matches
    let n = ys.size()[0];
    assert!(n >= 40 && n <= 50, "Expected 40-50 points but got {}", n);

    // Test accuracy of the solution
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 1.5e-3
                    && (exact_solution.1 - y[1]).abs() <= 1.5e-3
            })
    );
}

#[test]
fn test_solver_row1_case1() {
    use super::ode_solving_data::row1_case1_data::ROW1_CASE1_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&ROW1_CASE1_DATA);

    let solver = Solver::ROW1 { step: 0.1f64 };

    let y0 = Tensor::from_slice(&[-0.2f64, 0.1f64]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0f64), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0f64, 6.5f64);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f64) * 0.1)
            .collect::<Vec<f64>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-14);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 5e-5);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 0.35 && (exact_solution.1 - y[1]).abs() <= 0.35
            })
    );
}

#[test]
fn test_solver_row1_case2() {
    use super::ode_solving_data::row1_case2_data::ROW1_CASE2_DATA;
    let test_ys: tch::Tensor = tch::Tensor::from_slice2(&ROW1_CASE2_DATA);

    let solver = Solver::ROW1 { step: 0.1f64 };

    let y0 = Tensor::from_slice(&[-0.2f32, 0.1f32]);
    let mut closure = |inputs: &[Tensor]| {
        let _x = &inputs[0];
        let y = &inputs[1];
        let y0 = y.get(0);
        let y1 = y.get(1);

        let dy0 = y1;
        let dy1 = &y0 - &y0.pow_tensor_scalar(3.0);

        vec![Tensor::stack(&[dy0, dy1], 0)]
    };
    let model = CModule::create_by_tracing(
        "ode_fn",
        "forward",
        &[Tensor::from(0.0f32), y0.shallow_clone()],
        &mut closure,
    )
    .unwrap();

    let x_span = (0.0f64, 6.5f64);
    let (xs, ys) = solver.solve(model, x_span, y0).unwrap();
    let test_xs = tch::Tensor::from_slice(
        &(0..66)
            .into_iter()
            .map(|n| (n as f32) * 0.1)
            .collect::<Vec<f32>>(),
    );

    // Check with model run
    assert!((xs.copy() - test_xs).abs().max().double_value(&[]) <= 1e-6);
    assert!((ys.copy() - test_ys).abs().max().double_value(&[]) <= 5e-5);

    // Check accuracy
    assert!(
        Vec::<f64>::try_from(xs)
            .unwrap()
            .into_iter()
            .zip(Vec::<Vec::<f64>>::try_from(ys).unwrap().into_iter())
            .all(|(t, y)| {
                let exact_solution = ode_exact_solution(t);
                (exact_solution.0 - y[0]).abs() <= 0.35 && (exact_solution.1 - y[1]).abs() <= 0.35
            })
    );
}
