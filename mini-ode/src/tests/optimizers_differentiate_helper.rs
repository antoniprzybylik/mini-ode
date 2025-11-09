use tch::{Tensor, IndexOp};
use crate::optimizers::differentiate;

#[test]
fn test_differentiate_case1() {
    let x = Tensor::from_slice(&[1.0f32, 2.0f32]);
    let grad = differentiate(&|y| y.i(0)+y.i(1), &x);
    let expected = Tensor::from_slice(&[1f32, 1f32]);
    assert_eq!(grad, expected);
}

#[test]
fn test_differentiate_case2() {
    let x = Tensor::from_slice(&[1.0f32, 2.0f32]);
    let grad = differentiate(&|y| y.i(0)*y.i(0) + y.i(1)*y.i(1)*y.i(1) + y.i(1), &x);
    let expected = Tensor::from_slice(&[2f32, 13f32]);
    assert_eq!(grad, expected);
}

#[test]
fn test_differentiate_case3() {
    let x = Tensor::from_slice(&[-1.0f32, 1.5f32]);
    let grad = differentiate(&|y| y.i(0).exp() + y.i(1), &x);
    let expected = Tensor::stack(&[Tensor::from(-1f32).exp(), Tensor::from(1f32)], 0);
    assert_eq!(grad, expected);
}
