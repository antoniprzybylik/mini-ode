use tch::{Tensor, IndexOp};
use crate::optimizers::gradient_and_hessian;

#[test]
fn test_gradient_and_hessian_case1() {
    let x = Tensor::from_slice(&[1.0f32, 2.0f32]);
    let (grad, hessian) = gradient_and_hessian(&|y| y.i(0)+y.i(1), &x);
    let expected_grad = Tensor::from_slice(&[1f32, 1f32]);
    let expected_hessian = Tensor::from_slice(&[0f32, 0f32, 0f32, 0f32]).reshape([2, 2]);

    assert_eq!(grad, expected_grad);
    assert_eq!(hessian, expected_hessian);
}

#[test]
fn test_gradient_and_hessian_case2() {
    let x = Tensor::from_slice(&[5.0f32, 1.0f32]);
    let (grad, hessian) = gradient_and_hessian(&|y| y.i(0)*y.i(0)+y.i(0)*y.i(1), &x);
    let expected_grad = Tensor::from_slice(&[11f32, 5f32]);
    let expected_hessian = Tensor::from_slice(&[2f32, 1f32, 1f32, 0f32]).reshape([2, 2]);

    assert_eq!(grad, expected_grad);
    assert_eq!(hessian, expected_hessian);
}
