use crate::optimizers;

#[test]
fn test_optimizer_display_cg_case1() {
    let optimizer = optimizers::CG::new(10, None, None);
    assert_eq!(optimizer.to_string(), "CG(max_steps=10)");
}

#[test]
fn test_optimizer_display_cg_case2() {
    let optimizer = optimizers::CG::new(20, Some(2.1), None);
    assert_eq!(optimizer.to_string(), "CG(max_steps=20, gtol=2.1)");
}

#[test]
fn test_optimizer_display_cg_case3() {
    let optimizer = optimizers::CG::new(30, None, Some(1.1));
    assert_eq!(optimizer.to_string(), "CG(max_steps=30, ftol=1.1)");
}

#[test]
fn test_optimizer_display_cg_case4() {
    let optimizer = optimizers::CG::new(40, Some(4.5), Some(3.1));
    assert_eq!(optimizer.to_string(), "CG(max_steps=40, gtol=4.5, ftol=3.1)");
}

#[test]
fn test_optimizer_display_newton_case1() {
    let optimizer = optimizers::Newton::new(10, None, None);
    assert_eq!(optimizer.to_string(), "Newton(max_steps=10)");
}

#[test]
fn test_optimizer_display_newton_case2() {
    let optimizer = optimizers::Newton::new(20, Some(2.1), None);
    assert_eq!(optimizer.to_string(), "Newton(max_steps=20, gtol=2.1)");
}

#[test]
fn test_optimizer_display_newton_case3() {
    let optimizer = optimizers::Newton::new(30, None, Some(1.1));
    assert_eq!(optimizer.to_string(), "Newton(max_steps=30, ftol=1.1)");
}

#[test]
fn test_optimizer_display_newton_case4() {
    let optimizer = optimizers::Newton::new(40, Some(4.5), Some(3.1));
    assert_eq!(optimizer.to_string(), "Newton(max_steps=40, gtol=4.5, ftol=3.1)");
}

#[test]
fn test_optimizer_display_halley_case1() {
    let optimizer = optimizers::Halley::new(10, None, None);
    assert_eq!(optimizer.to_string(), "Halley(max_steps=10)");
}

#[test]
fn test_optimizer_display_halley_case2() {
    let optimizer = optimizers::Halley::new(20, Some(2.1), None);
    assert_eq!(optimizer.to_string(), "Halley(max_steps=20, gtol=2.1)");
}

#[test]
fn test_optimizer_display_halley_case3() {
    let optimizer = optimizers::Halley::new(30, None, Some(1.1));
    assert_eq!(optimizer.to_string(), "Halley(max_steps=30, ftol=1.1)");
}

#[test]
fn test_optimizer_display_halley_case4() {
    let optimizer = optimizers::Halley::new(40, Some(4.5), Some(3.1));
    assert_eq!(optimizer.to_string(), "Halley(max_steps=40, gtol=4.5, ftol=3.1)");
}

#[test]
fn test_optimizer_display_bfgs_case1() {
    let optimizer = optimizers::BFGS::new(10, None, None);
    assert_eq!(optimizer.to_string(), "BFGS(max_steps=10)");
}

#[test]
fn test_optimizer_display_bfgs_case2() {
    let optimizer = optimizers::BFGS::new(20, Some(2.1), None);
    assert_eq!(optimizer.to_string(), "BFGS(max_steps=20, gtol=2.1)");
}

#[test]
fn test_optimizer_display_bfgs_case3() {
    let optimizer = optimizers::BFGS::new(30, None, Some(1.1));
    assert_eq!(optimizer.to_string(), "BFGS(max_steps=30, ftol=1.1)");
}

#[test]
fn test_optimizer_display_bfgs_case4() {
    let optimizer = optimizers::BFGS::new(40, Some(4.5), Some(3.1));
    assert_eq!(optimizer.to_string(), "BFGS(max_steps=40, gtol=4.5, ftol=3.1)");
}
