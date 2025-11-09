use tch::{Tensor, IndexOp};
use crate::optimizers::derivative_tensors_123;

#[test]
fn test_derivative_tensors_123_case1() {
    let x = Tensor::from_slice(&[1.0f32, 2.0f32]);
    let (grad, hessian, d3_tensor) = derivative_tensors_123(&|y| y.i(0)+y.i(1), &x);
    let expected_grad = Tensor::from_slice(&[1f32, 1f32]);
    let expected_hessian = Tensor::from_slice(&[0f32, 0f32, 0f32, 0f32]).reshape([2, 2]);
    let expected_d3_tensor = Tensor::zeros([2, 2, 2], (x.kind(), x.device()));

    assert_eq!(grad, expected_grad);
    assert_eq!(hessian, expected_hessian);
    assert_eq!(d3_tensor, expected_d3_tensor);
}

#[test]
fn test_derivative_tensors_123_case2() {
    let x = Tensor::from_slice(&[5.0f32, 1.0f32]);
    let (grad, hessian, d3_tensor) = derivative_tensors_123(&|y| y.i(0)*y.i(0)+y.i(0)*y.i(1), &x);
    let expected_grad = Tensor::from_slice(&[11f32, 5f32]);
    let expected_hessian = Tensor::from_slice(&[2f32, 1f32, 1f32, 0f32]).reshape([2, 2]);
    let expected_d3_tensor = Tensor::zeros([2, 2, 2], (x.kind(), x.device()));

    assert_eq!(grad, expected_grad);
    assert_eq!(hessian, expected_hessian);
    assert_eq!(d3_tensor, expected_d3_tensor);
}

#[test]
fn test_derivative_tensors_123_case3() {
    let x = Tensor::from_slice(&[5.0f32, 1.0f32]);
    let (grad, hessian, d3_tensor) = derivative_tensors_123(&|y| y.i(0)*y.i(1)*y.i(1) + y.i(0)*y.i(0)+y.i(0)*y.i(1), &x);
    let expected_grad = Tensor::from_slice(&[12f32, 15f32]);
    let expected_hessian = Tensor::from_slice(&[2f32, 3f32, 3f32, 10f32]).reshape([2, 2]);
    let expected_d3_tensor = Tensor::from_slice(&[0f32, 0f32, 0f32, 2f32, 0f32, 2f32, 2f32, 0f32]).reshape([2, 2, 2]);

    assert_eq!(grad, expected_grad);
    assert_eq!(hessian, expected_hessian);
    assert_eq!(d3_tensor, expected_d3_tensor);
}
