use crate::utils::differentiation::compute_jacobian;
use tch::{Device, IndexOp, Kind, Tensor};

#[test]
fn test_compute_jacobian_error_non_1d_input() {
    let x = Tensor::zeros(&[2, 2], (Kind::Float, Device::Cpu));

    let result = compute_jacobian(|_| Tensor::ones(&[1], (Kind::Float, Device::Cpu)), &x);

    assert!(result.is_err());

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("one-dimensional"),
        "Unexpected error message: {}",
        err
    );
}

#[test]
fn test_compute_jacobian_error_non_1d_output() {
    let x = Tensor::ones(&[1], (Kind::Float, Device::Cpu));

    let result = compute_jacobian(|_| Tensor::ones(&[2, 2], (Kind::Float, Device::Cpu)), &x);

    assert!(result.is_err());

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("one-dimensional"),
        "Unexpected error message: {}",
        err
    );
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
    )
    .unwrap();

    let expected = Tensor::from_slice(&[2.0f32, 0.0f32, 0.0f32, 3.0f32]).reshape(&[2, 2]);

    assert_eq!(jacobian, expected);
}

#[test]
fn test_compute_jacobian_nonlinear_scalar() {
    let x = Tensor::from_slice(&[2.0f32]);

    let jacobian = compute_jacobian(|y| y.pow_tensor_scalar(2), &x).unwrap();

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

            Tensor::stack(&[&(&y0.pow_tensor_scalar(2) + &y1), &(&y0 * &y1)], 0)
        },
        &x,
    )
    .unwrap();

    let expected = Tensor::from_slice(&[2.0f32, 1.0f32, 2.0f32, 1.0f32]).reshape(&[2, 2]);

    assert_eq!(jacobian, expected);
}
