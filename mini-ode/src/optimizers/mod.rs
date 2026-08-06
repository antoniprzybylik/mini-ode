//! # Optimization Algorithms for Implicit ODE Solvers
//!
//! **This module provides nonlinear optimization algorithms** that are required by implicit ODE
//! solvers in the `mini-ode` library. Implicit methods (such as
//! [`Solver::ImplicitEuler`](crate::solvers::Solver::ImplicitEuler) and
//! [`Solver::GLRK4`](crate::solvers::Solver::GLRK4)) need to solve nonlinear equations at each
//! timestep, which is accomplished through numerical optimization.
//!
//! ## Design
//!
//! The optimization API follows a trait-based design that allows extensibility:
//! - All optimizers implement the [`Optimizer`] trait with a unified `optimize()` interface
//! - Implementations are stateless configuration objects (can be safely shared via
//! [`Arc`](std::sync::Arc))
//! - Each optimizer requires only the objective function and initial guess
//! - Results are returned as `anyhow::Result<Tensor>` for consistent error handling
//!
//! ## Available Optimizers
//!
//! The module exports four gradient-based optimization algorithms:
//!
//! | Optimizer | Order | Hessian | Best For |
//! |-----------|-------|---------|----------|
//! | [`Newton`] | 2nd | Yes | Fast convergence, well-conditioned problems |
//! | [`BFGS`] | Quasi-Newton | Approximated | General-purpose, memory-efficient |
//! | [`CG`] | 1st | No | Large-scale problems, limited memory |
//! | [`Halley`] | 3rd | Yes (Higher) | Very fast convergence |
//!
//! ## Usage Example
//!
//! Creating an optimizer and passing it to an implicit solver:
//!
//! ```rust,no_run
//! use mini_ode::optimizers;
//! use mini_ode::Solver;
//! use std::sync::Arc;
//!
//! // Configure Conjugate Gradient optimizer
//! let optimizer = optimizers::CG::new(50, Some(1e-6), Some(1e-8));
//!
//! // Use with implicit solver
//! let solver = Solver::ImplicitEuler {
//!     step: 0.01,
//!     optimizer: Arc::new(optimizer),
//! };
//! ```
//!
//! ## Implementation Details
//!
//! - **Automatic differentiation**: Optimizers use torch.autograd automatic differentiation to compute
//!   gradients and Hessians
//! - **Line search**: Step size is chosen using line search
//! - **Validation**: Output tensors are validated for finite values and proper ranks
//! - **Warnings**: Ill-conditioning and numerical issues produce runtime warnings
//!
//! ## Requirements
//!
//! These optimizers require the [`tch`] crate (libtorch Rust bindings) and depend on:
//! - PyTorch tensor operations on CPU or GPU
//! - Automatic differentiation support
//!
//! ## Error Handling
//!
//! Optimizers may return errors for:
//! - Input validation failures (non-scalar outputs, rank mismatches)
//! - Memory allocation failures (insufficient RAM for Hessian)
//! - Convergence failures (max steps reached without meeting tolerances)
//! - Numerical issues (NaN/Inf in intermediate calculations)

use std::fmt;
use tch::Tensor;

/// Core trait defining the interface for all optimization algorithms.
///
/// This trait abstracts over different optimization strategies (Newton, BFGS, CG, Halley),
/// allowing them to be used interchangeably with implicit ODE solvers. The trait is designed
/// for thread-safe usage with `Send + Sync` bounds.
///
/// ## Function Signature
///
/// The `optimize` method accepts:
/// - **`function`**: A closure taking a 1D tensor `x` and returning a scalar tensor representing
///   the objective value to minimize. The function should use torch operations compatible with
///   autodifferentiation.
/// - **`x0`**: Initial guess as a 1D tensor, which determines both the problem dimension and
///   computation device (CPU/GPU).
///
/// ## Return Value
///
/// Returns `Ok(optimal_x)` containing the optimized parameter vector, or `Err(e)` if optimization
/// fails due to convergence issues, numerical instability, or invalid inputs.
///
/// ## Implementation Notes
///
/// - The function is evaluated multiple times during optimization
/// - Gradients/Hessians are computed automatically via torch.autograd
/// - Tolerance parameters control convergence criteria
pub trait Optimizer: Send + Sync + fmt::Display {
    /// Minimizes an objective function starting from an initial guess.
    ///
    /// # Arguments
    /// * `function` - A closure that takes a 1D tensor `x` and returns a scalar tensor
    ///   representing the objective value to minimize. Must support automatic differentiation.
    /// * `x0` - Initial guess, 1D tensor accepted by `function`. Determines computation device
    ///   and floating-point precision.
    ///
    /// # Returns
    /// Optimal `x` that minimizes `function`, or error if optimization fails.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Input validation fails (non-scalar output, wrong rank, non-finite values)
    /// - Hessian allocation fails due to insufficient memory
    /// - Line search fails
    /// - Maximum iterations reached without convergence
    ///
    /// # Panics
    /// May panic if libtorch tensor operations fail unexpectedly (should be rare).
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
