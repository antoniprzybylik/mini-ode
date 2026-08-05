use anyhow::anyhow;
use tch::IndexOp;
use tch::Tensor;

/// Computes the Jacobian matrix of a function f at point x
pub(crate) fn compute_jacobian<F>(f: F, x: &Tensor) -> anyhow::Result<Tensor>
where
    F: Fn(&Tensor) -> Tensor,
{
    if x.dim() != 1 {
        return Err(anyhow!(
            "Jacobian input tensor must be one-dimensional, got {} dimensions",
            x.dim()
        ));
    }

    let x_with_grad = x.detach().copy().set_requires_grad(true);

    let y = f(&x_with_grad);

    if y.dim() != 1 {
        return Err(anyhow!(
            "Jacobian output tensor must be one-dimensional, got {} dimensions",
            y.dim()
        ));
    }

    if y.isfinite().f_all()?.f_int64_value(&[])? == 0 {
        return Err(anyhow!(
            "Jacobian function returned tensor containing non-finite values"
        ));
    }

    let y_size = y.size()[0];
    let mut grads = Vec::with_capacity(y_size as usize);

    for i in 0..y_size {
        let yi = y.i(i);

        let grad = Tensor::f_run_backward(&[yi], &[&x_with_grad], true, false)?
            .first()
            .ok_or_else(|| anyhow!("Failed to compute Jacobian gradient"))?
            .copy();

        if grad.size() != x.size() {
            return Err(anyhow!(
                "Jacobian gradient has shape {:?}, expected shape {:?}",
                grad.size(),
                x.size()
            ));
        }

        if grad.isfinite().f_all()?.f_int64_value(&[])? == 0 {
            return Err(anyhow!("Jacobian computation produced non-finite values"));
        }

        grads.push(grad);
    }

    let jacobian = Tensor::f_stack(&grads, 0)?;

    if jacobian.size() != vec![y_size, x.size()[0]] {
        return Err(anyhow!(
            "Jacobian has shape {:?}, expected shape [{}, {}]",
            jacobian.size(),
            y_size,
            x.size()[0]
        ));
    }

    Ok(jacobian)
}

/// Computes the gradient of `function` at `x` using automatic differentiation.
///
/// # Arguments
/// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
/// * `x` - Evaluation point (1D tensor).
///
/// # Returns
/// Gradient tensor at `x`.
pub(crate) fn differentiate(
    function: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
) -> anyhow::Result<Tensor> {
    let x_with_grad = x.f_detach()?.copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    if y.size() != [] as [i64; 0] {
        return Err(anyhow!(
            "Bad shape of `y`. Expected [], but got {:?}",
            y.size()
        ));
    }

    if !y.requires_grad() {
        return Ok(tch::Tensor::f_zeros(x.size(), (x.kind(), x.device()))?);
    }

    let gradient = tch::Tensor::f_run_backward(&[y], &[x_with_grad], false, false)?[0].copy();

    // Note: We don't validate here because gradients can legitimately be non-finite
    // during exploration; caller will validate final result

    Ok(gradient)
}

/// Computes the gradient and Hessian of `function` at `x` using automatic differentiation.
///
/// # Arguments
/// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
/// * `x` - Evaluation point (1D tensor).
/// # Returns
/// Tuple `(grad, hessian)`, both detached tensors. `grad` is 1D, `hessian` is 2D.
pub(crate) fn gradient_and_hessian(
    function: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_with_grad = x.f_detach()?.copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    if y.size() != [] as [i64; 0] {
        return Err(anyhow!(
            "Bad shape of `y`. Expected [], but got {:?}",
            y.size()
        ));
    }

    if !y.requires_grad() {
        return Ok((
            tch::Tensor::f_zeros(x.size(), (x.kind(), x.device()))?,
            tch::Tensor::f_zeros([x.size()[0], x.size()[0]], (x.kind(), x.device()))?,
        ));
    }

    // keep_graph = true (this graph is needed for some functions during second differentiation)
    // create_graph = true (allow calculating second derivatives)
    let grad = Tensor::f_run_backward(&[y], &[&x_with_grad], true, true)?[0].copy();
    let grad_len = grad.size()[0];
    let grad_kind = grad.kind();
    let grad_device = grad.device();

    // If gradient is constant, immediately return gradient and zero hessian
    // It is not possible to differentiate constants in torch
    if !grad.requires_grad() {
        return Ok((
            grad,
            Tensor::f_zeros([grad_len, grad_len], (grad_kind, grad_device))?,
        ));
    }

    let mut vectors = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        // keep_graph = true (we need to run backward pass multiple times - in each iteration of the loop)
        // create_graph = false (we don't need to differentiate three times)
        vectors.append(&mut Tensor::f_run_backward(
            &[grad.i(i)],
            &[&x_with_grad],
            true,
            false,
        )?);
    }

    // Detach autograd computation graph
    let grad = grad.f_detach()?;
    // Stack slices of the Hessian matrix and detach autograd computation graph
    let hessian = Tensor::f_stack(&vectors, 0)?.f_detach()?;

    Ok((grad, hessian))
}

/// Computes the gradient, Hessian and third derivatives tensor of `function` at `x` using automatic differentiation.
///
/// # Arguments
/// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
/// * `x` - Evaluation point (1D tensor).
/// # Returns
/// Tuple `(grad, hessian, d3_tensor)`, both detached tensors. `grad` is 1D, `hessian` is 2D, `d3_tensor` is 3D.
pub(crate) fn derivative_tensors_123(
    function: &dyn Fn(&Tensor) -> Tensor,
    x: &Tensor,
) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    let x_with_grad = x.f_detach()?.copy().set_requires_grad(true);
    let y = function(&x_with_grad);

    if y.size() != [] as [i64; 0] {
        return Err(anyhow!(
            "Bad shape of `y`. Expected [], but got {:?}",
            y.size()
        ));
    }

    if !y.requires_grad() {
        return Ok((
            tch::Tensor::f_zeros(x.size(), (x.kind(), x.device()))?,
            tch::Tensor::f_zeros([x.size()[0], x.size()[0]], (x.kind(), x.device()))?,
            tch::Tensor::f_zeros(
                [x.size()[0], x.size()[0], x.size()[0]],
                (x.kind(), x.device()),
            )?,
        ));
    }

    // keep_graph = true (this graph is needed for some functions during second differentiation)
    // create_graph = true (allow calculating second derivatives)
    let grad = Tensor::f_run_backward(&[y], &[&x_with_grad], true, true)?[0].copy();
    let grad_len = grad.size()[0];
    let grad_kind = grad.kind();
    let grad_device = grad.device();

    // If gradient is constant, immediately return gradient zero hessian and
    // zero tensor of third order derivatives
    // It is not possible to differentiate constants in torch
    if !grad.requires_grad() {
        return Ok((
            grad,
            Tensor::f_zeros([grad_len, grad_len], (grad_kind, grad_device))?,
            Tensor::f_zeros([grad_len, grad_len, grad_len], (grad_kind, grad_device))?,
        ));
    }

    let mut vectors = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        // keep_graph = true (we need to run backward pass multiple times - in each iteration of the loop)
        // create_graph = true (we need to differentiate three times)
        vectors.append(&mut Tensor::f_run_backward(
            &[grad.i(i)],
            &[&x_with_grad],
            true,
            true,
        )?);
    }

    // Stack slices of the Hessian matrix
    let hessian = Tensor::f_stack(&vectors, 0)?;

    // If gradient is constant, immediately return gradient zero hessian and
    // zero tensor of third order derivatives
    // It is not possible to differentiate constants in torch
    if !hessian.requires_grad() {
        return Ok((
            grad,
            hessian,
            Tensor::f_zeros([grad_len, grad_len, grad_len], (grad_kind, grad_device))?,
        ));
    }

    let mut vectors2 = Vec::<Tensor>::with_capacity(grad_len as usize);
    for i in 0..grad_len {
        let mut vectors1 = Vec::<Tensor>::with_capacity(grad_len as usize);
        for j in 0..grad_len {
            vectors1.append(&mut Tensor::f_run_backward(
                &[hessian.i((i, j))],
                &[&x_with_grad],
                true,
                false,
            )?);
        }
        vectors2.push(Tensor::f_stack(&vectors1, 0)?);
    }

    // Detach autograd computation graph
    let grad = grad.f_detach()?;
    let hessian = hessian.f_detach()?;
    // Stack slices of the tensor of third derivatives and detach autograd computation graph
    let d3_tensor = Tensor::f_stack(&vectors2, 0)?.f_detach()?;

    Ok((grad, hessian, d3_tensor))
}
