//! Nonlinear optimization algorithms which are required by some ODE solvers
//!
//! The user may create objects containing optimizer configuration and pass it to ODE solver.

use std::fmt;
use tch::Tensor;

/// Optimizer interface common for any optimizer in the library
pub trait Optimizer: Send + Sync + fmt::Display {
    /// Solves the problem of optimization of function `function` starting from point `x0`
    ///
    /// # Arguments
    /// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor.
    ///   representing the objective value to minimize.
    /// * `x0` - Initial guess, 1D tensor accepted by `function`.
    ///
    /// # Returns
    /// Optimal `x`, or error if optimization fails.
    ///
    /// # Panics
    /// May panic if libtorch tensor operations fail.
    fn optimize(&self, function: &dyn Fn(&Tensor) -> Tensor, x0: &Tensor)
    -> anyhow::Result<Tensor>;
}

mod newton;
pub use newton::Newton;
mod bfgs;
pub use bfgs::BFGS;
mod cg;
pub use cg::CG;
mod halley;
pub use halley::Halley;
