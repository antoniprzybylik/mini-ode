use std::sync::Arc;
use crate::Solver;
use super::mock_optimizer::MockOptimizer;
use crate::optimizers;

#[test]
fn test_solver_display_euler_case1() {
    let solver = Solver::Euler { step: 0.1 };
    assert_eq!(solver.to_string(), "Euler(step=0.1)");
}

#[test]
fn test_solver_display_rk4() {
    let solver = Solver::RK4 { step: 0.05 };
    assert_eq!(solver.to_string(), "RK4(step=0.05)");
}

#[test]
fn test_solver_display_implicit_euler_mock_optimizer() {
    let optimizer = Arc::new(MockOptimizer);
    let solver = Solver::ImplicitEuler {
        step: 0.2,
        optimizer,
    };
    assert_eq!(
        solver.to_string(),
        "ImplicitEuler(step=0.2, optimizer=MockOptimizer)"
    );
}

#[test]
fn test_solver_display_implicit_euler_cg_optimizer_case1() {
    let optimizer = Arc::new(optimizers::CG::new(
        100,
        None,
        None
    ));
    let solver = Solver::ImplicitEuler {
        step: 0.2,
        optimizer,
    };
    assert_eq!(
        solver.to_string(),
        "ImplicitEuler(step=0.2, optimizer=CG(max_steps=100))"
    );
}

#[test]
fn test_solver_display_implicit_euler_cg_optimizer_case2() {
    let optimizer = Arc::new(optimizers::CG::new(
        100,
        Some(0.001),
        None
    ));
    let solver = Solver::ImplicitEuler {
        step: 0.2,
        optimizer,
    };
    assert_eq!(
        solver.to_string(),
        "ImplicitEuler(step=0.2, optimizer=CG(max_steps=100, gtol=0.001))"
    );
}

#[test]
fn test_solver_display_implicit_euler_cg_optimizer_case3() {
    let optimizer = Arc::new(optimizers::CG::new(
        100,
        None,
        Some(7.1)
    ));
    let solver = Solver::ImplicitEuler {
        step: 0.2,
        optimizer,
    };
    assert_eq!(
        solver.to_string(),
        "ImplicitEuler(step=0.2, optimizer=CG(max_steps=100, ftol=7.1))"
    );
}

#[test]
fn test_solver_display_implicit_euler_cg_optimizer_case4() {
    let optimizer = Arc::new(optimizers::CG::new(
        100,
        Some(0.003),
        Some(0.0005)
    ));
    let solver = Solver::ImplicitEuler {
        step: 0.2,
        optimizer,
    };
    assert_eq!(
        solver.to_string(),
        "ImplicitEuler(step=0.2, optimizer=CG(max_steps=100, gtol=0.003, ftol=0.0005))"
    );
}

#[test]
fn test_solver_display_glrk4() {
    let optimizer = Arc::new(MockOptimizer);
    let solver = Solver::GLRK4 {
        step: 0.3,
        optimizer,
    };
    assert_eq!(
        solver.to_string(),
        "GLRK4(step=0.3, optimizer=MockOptimizer)"
    );
}

#[test]
fn test_solver_display_rkf45() {
    let solver = Solver::RKF45 {
        rtol: 1e-6,
        atol: 1e-8,
        min_step: 1e-10,
        safety_factor: 0.9,
    };
    assert_eq!(
        solver.to_string(),
        "RKF45(rtol=0.000001, atol=0.00000001, min_step=0.0000000001, safety_factor=0.9)"
    );
}

#[test]
fn test_solver_display_row1() {
    let solver = Solver::ROW1 { step: 0.4 };
    assert_eq!(solver.to_string(), "ROW1(step=0.4)");
}
