use tch::Tensor;

/// Validates that a tensor contains only finite values.
/// Returns an error if any NaN or Inf values are detected.
pub(crate) fn validate_finite_tensor(tensor: &Tensor, context: &str) -> anyhow::Result<()> {
    if tensor.isfinite().f_all()?.f_int64_value(&[])? == 0 {
        anyhow::bail!(
            "Non-finite values (NaN/Inf) detected in {}: tensor shape {:?}",
            context,
            tensor.size()
        );
    }
    Ok(())
}

/// Validates that a scalar value is finite.
pub(crate) fn validate_finite_scalar(value: f64, context: &str) -> anyhow::Result<()> {
    if !value.is_finite() {
        anyhow::bail!("Non-finite value ({}) detected in {}", value, context);
    }
    Ok(())
}

/// Helper function to validate that optimizer output is finite
pub(crate) fn validate_optimizer_output(
    tensor: &Tensor,
    optimizer_name: &str,
) -> anyhow::Result<()> {
    if tensor.isfinite().f_all()?.f_int64_value(&[])? == 0 {
        anyhow::bail!(
            "Optimizer {} produced non-finite result (NaN/Inf)",
            optimizer_name
        );
    }
    Ok(())
}
