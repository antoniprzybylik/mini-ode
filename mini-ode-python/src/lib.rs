use mini_ode::optimizers::Optimizer;
use mini_ode::Solver;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use std::io::Cursor;
use std::sync::Arc;

#[derive(Clone)]
#[pyclass(module = "rust.optimizers", name = "Optimizer")]
struct PyOptimizer(Arc<dyn Optimizer + Send + Sync>);

#[pyfunction(
    name = "CG",
    signature = (max_steps, gtol=None, ftol=None, linesearch_atol=None)
)]
fn create_cg(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
    linesearch_atol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Arc::new(mini_ode::optimizers::CG::new(
        max_steps,
        gtol,
        ftol,
        linesearch_atol,
    )))
}

#[pyfunction(
    name = "BFGS",
    signature = (max_steps, gtol=None, ftol=None, linesearch_atol=None)
)]
fn create_bfgs(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
    linesearch_atol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Arc::new(mini_ode::optimizers::BFGS::new(
        max_steps,
        gtol,
        ftol,
        linesearch_atol,
    )))
}

#[pyclass(module = "rust", name = "Solver")]
struct PySolver(Solver);

#[pymethods]
impl PySolver {
    fn solve(
        &self,
        py: Python,
        f: PyObject,
        x_span: PyTensor,
        y0: PyTensor
    ) -> PyResult<(PyTensor, PyTensor)> {
        let f_module = convert_function(py, f)?;
        let x_span_inner = x_span.0.copy();
        let y0_inner = y0.0.copy();
        py.allow_threads(|| {
            self.0
                .solve(f_module, x_span_inner, y0_inner)
                .map(|(x, y)| (PyTensor(x), PyTensor(y)))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

#[pyfunction(name = "EulerMethodSolver")]
fn create_euler_solver(step: f64) -> PySolver {
    PySolver(Solver::Euler { step })
}

#[pyfunction(name = "RK4MethodSolver")]
fn create_rk4_solver(step: f64) -> PySolver {
    PySolver(Solver::RK4 { step })
}

#[pyfunction(name = "ImplicitEulerMethodSolver")]
fn create_implicit_euler_solver(step: f64, optimizer: PyOptimizer) -> PySolver {
    PySolver(Solver::ImplicitEuler { step, optimizer: optimizer.0 })
}

#[pyfunction(name = "GLRK4MethodSolver")]
fn create_glrk4_solver(step: f64, optimizer: PyOptimizer) -> PySolver {
    PySolver(Solver::GLRK4 { step, optimizer: optimizer.0 })
}

#[pyfunction(name = "RKF45MethodSolver")]
fn create_rkf45_solver(rtol: f64, atol: f64, min_step: f64, safety_factor: f64) -> PySolver {
    PySolver(Solver::RKF45 { rtol, atol, min_step, safety_factor })
}

#[pyfunction(name = "ROW1MethodSolver")]
fn create_row1_solver(step: f64) -> PySolver {
    PySolver(Solver::ROW1 { step })
}

fn convert_function(py: Python, f: PyObject) -> PyResult<tch::CModule> {
    let torch = py.import("torch")?;
    let script_function_type = torch.getattr("jit")?.getattr("ScriptFunction")?;

    if !f.bind(py).is_instance(&script_function_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Function must be a torch.jit.ScriptFunction",
        ));
    }

    let buffer = f
        .call_method0(py, "save_to_buffer")?
        .extract::<Vec<u8>>(py)?;
    let mut cursor = Cursor::new(buffer);
    tch::CModule::load_data(&mut cursor).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load model: {}", e))
    })
}

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_euler_solver, m)?)?;
    m.add_function(wrap_pyfunction!(create_rk4_solver, m)?)?;
    m.add_function(wrap_pyfunction!(create_implicit_euler_solver, m)?)?;
    m.add_function(wrap_pyfunction!(create_glrk4_solver, m)?)?;
    m.add_function(wrap_pyfunction!(create_rkf45_solver, m)?)?;
    m.add_function(wrap_pyfunction!(create_row1_solver, m)?)?;
    m.add_function(wrap_pyfunction!(create_cg, m)?)?;
    m.add_function(wrap_pyfunction!(create_bfgs, m)?)?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyOptimizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
