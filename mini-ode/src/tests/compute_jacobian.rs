use tch::{Device, Kind, Tensor, IndexOp};
use crate::compute_jacobian;

#[test]
#[should_panic(expected = "x must be 1-dimensional")]
fn test_compute_jacobian_panic_non_1d_input() {
    let x = Tensor::zeros(&[2, 2], (Kind::Float, Device::Cpu));
    let _ = compute_jacobian(|_| Tensor::ones(&[1], (Kind::Float, Device::Cpu)), &x);
}

#[test]
#[should_panic(expected = "y must be 1-dimensional")]
fn test_compute_jacobian_panic_non_1d_output() {
    let x = Tensor::ones(&[1], (Kind::Float, Device::Cpu));
    let _ = compute_jacobian(|_| Tensor::ones(&[2, 2], (Kind::Float, Device::Cpu)), &x);
}

#[test]
fn test_compute_jacobian_linear() {
    let x = Tensor::from_slice(&[1.0f32, 2.0f32]);
    let jacobian = compute_jacobian(
        |y| {
            let a = Tensor::from_slice(&[2.0f32, 0.0f32, 0.0f32, 3.0f32]).reshape(&[2, 2]);
            a.matmul(y)
        },
        &x,
    );
    let expected = Tensor::from_slice(&[2.0f32, 0.0f32, 0.0f32, 3.0f32]).reshape(&[2, 2]);
    assert_eq!(jacobian, expected);
}

#[test]
fn test_compute_jacobian_nonlinear_scalar() {
    let x = Tensor::from_slice(&[2.0f32]);
    let jacobian = compute_jacobian(|y| y.pow_tensor_scalar(2), &x);
    let expected = Tensor::from_slice(&[4.0f32]).reshape(&[1, 1]);
    assert_eq!(jacobian, expected);
}

#[test]
fn test_compute_jacobian_multi_dim_nonlinear() {
    let x = Tensor::from_slice(&[1.0f32, 2.0f32]);
    let jacobian = compute_jacobian(
        |y| {
            let y0 = y.i(0);
            let y1 = y.i(1);
            Tensor::stack(&[&y0.pow_tensor_scalar(2) + &y1, (&y0 * &y1)], 0)
        },
        &x,
    );
    let expected = Tensor::from_slice(&[2.0f32, 1.0f32, 2.0f32, 1.0f32]).reshape(&[2, 2]);
    assert_eq!(jacobian, expected);
}
