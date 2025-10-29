# Changelog

This documents the main changes in the `mini-ode` project.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-10-26
### Added
- `std::fmt::Display` trait implementations for optimizers and solvers in `mini-ode` Rust crate.
- `__repr__` and `__str__` methods implementations for optimizers and solvers in `mini-ode` Python package
- `build.rs` files to enable GPU support in the built library.
- `.gitignore` file.

### Changed
- Migrated to Python 3.14, PyTorch 2.9.0, and tch 0.22.0.
- Changed the way parameters are provided (now integration interval is specified as a pair of floats, not tensor).

### Fixed
- Improved error reporting.

## [0.1.2] - 2025-10-16
### Added
- Additional checks for robustness.

### Changed
- Fixed Python version to 3.13 ABI (from 3.12 ABI) and migrated to PyTorch 2.8.0.

### Fixed
- Error in optimizers causing NaN values when local minimum is reached (gradient is zero).

## [0.1.1] - 2025-05-22
### Added
- License file.

### Changed
- Refactored code and updated interface.

## [0.0.3] - 2025-05-20
### Fixed
- Build issues on docs.rs.

## [0.0.2] - 2025-04-28
### Added
- GLRK4 solver implementation.

### Changed
- Changed Rust edition from "2024" to "2021" for `mini-ode-python`.

### Fixed
- Error with time interval handling.
- Mistake in `pyproject.toml`.
- Removed various warnings in `mini-ode` and `mini-ode-python`.

### Other
- Removed garbage files from the repository.

## [0.0.1] - 2025-04-17
### Added
- Initial implementation of `mini-ode`.
