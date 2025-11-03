use mini_ode::Solver;
use mini_ode::optimizers::Optimizer;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use std::io::Cursor;
use std::sync::Arc;

#[derive(Clone)]
#[pyclass(module = "rust.optimizers", name = "Optimizer")]
struct PyOptimizer(Arc<dyn Optimizer + Send + Sync>);

#[pymethods]
impl PyOptimizer {
    fn __repr__(&self) -> String {
        format!("{}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyfunction(
    name = "CG",
    signature = (max_steps, gtol=None, ftol=None)
)]
fn create_cg(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Arc::new(mini_ode::optimizers::CG::new(
        max_steps,
        gtol,
        ftol,
    )))
}

#[pyfunction(
    name = "BFGS",
    signature = (max_steps, gtol=None, ftol=None)
)]
fn create_bfgs(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Arc::new(mini_ode::optimizers::BFGS::new(
        max_steps,
        gtol,
        ftol,
    )))
}

#[pyfunction(
    name = "Newton",
    signature = (max_steps, gtol=None, ftol=None)
)]
fn create_newton(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Arc::new(mini_ode::optimizers::Newton::new(
        max_steps,
        gtol,
        ftol,
    )))
}

fn extract_pair<T>(object: &Bound<'_, PyAny>) -> Option<(T, T)>
where
    T: std::clone::Clone + for<'a> pyo3::FromPyObject<'a>,
{
    if let Ok(tuple) = object.downcast::<pyo3::types::PyTuple>() {
        if tuple.len() != 2 {
            return None;
        }

        let a = tuple.get_item(0).ok()?.extract::<T>().ok()?;
        let b = tuple.get_item(1).ok()?.extract::<T>().ok()?;

        return Some((a, b));
    }

    if let Ok(list) = object.downcast::<pyo3::types::PyList>() {
        if list.len() != 2 {
            return None;
        }

        let a = list.get_item(0).ok()?.extract::<T>().ok()?;
        let b = list.get_item(1).ok()?.extract::<T>().ok()?;

        return Some((a, b));
    }

    None
}

#[pyclass(module = "rust", name = "Solver")]
struct PySolver(Solver);

#[pymethods]
impl PySolver {
    fn solve<'py>(
        &self,
        py: Python<'py>,
        f: &Bound<'py, PyAny>,
        x_span: &Bound<'py, PyAny>,
        y0: PyTensor,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let f_module = convert_function(py, f)?;
        let x_span_extracted = match extract_pair::<f64>(x_span) {
            Some(pair) => pair,
            None => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "`x_span` must be a pair of floats",
                ));
            }
        };
        let y0_inner = y0.0.copy();
        py.allow_threads(|| {
            self.0
                .solve(f_module, x_span_extracted, y0_inner)
                .map(|(x, y)| (PyTensor(x), PyTensor(y)))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> String {
        let solver_string = format!("{}", self.0);
        let to_insert = "MethodSolver(";
        solver_string.replacen("(", to_insert, 1)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
        match name {
            "step" => {
                if let Some(step) = get_step(&self.0) {
                    Ok(step.into_pyobject(py)?.into())
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(format!("This solver has no attribute '{}'", name)))
                }
            }
            "optimizer" => {
                if let Some(optimizer_inner) = get_optimizer(&self.0) {
                    let py_optimizer = PyOptimizer(optimizer_inner);
                    Ok(py_optimizer.into_pyobject(py)?.into())
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(format!("This solver has no attribute '{}'", name)))
                }
            }
            "rtol" => {
                if let Some(rtol) = get_rtol(&self.0) {
                    Ok(rtol.into_pyobject(py)?.into())
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(format!("This solver has no attribute '{}'", name)))
                }
            }
            "atol" => {
                if let Some(atol) = get_atol(&self.0) {
                    Ok(atol.into_pyobject(py)?.into())
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(format!("This solver has no attribute '{}'", name)))
                }
            }
            "min_step" => {
                if let Some(min_step) = get_min_step(&self.0) {
                    Ok(min_step.into_pyobject(py)?.into())
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(format!("This solver has no attribute '{}'", name)))
                }
            }
            "safety_factor" => {
                if let Some(safety_factor) = get_safety_factor(&self.0) {
                    Ok(safety_factor.into_pyobject(py)?.into())
                } else {
                    Err(pyo3::exceptions::PyAttributeError::new_err(format!("This solver has no attribute '{}'", name)))
                }
            }
            _ => Err(pyo3::exceptions::PyAttributeError::new_err(format!("'Solver' object has no attribute '{}'", name))),
        }
    }

    fn __dir__(&self) -> Vec<String> {
        // Common attributes
        let mut attrs = vec!["solve".to_string(), "__repr__".to_string(), "__str__".to_string()];

        // Solver specific attributes
        match &self.0 {
            Solver::Euler { .. } | Solver::RK4 { .. } | Solver::ImplicitEuler { .. }
            | Solver::GLRK4 { .. } | Solver::ROW1 { .. } => attrs.push("step".to_string()),
            Solver::RKF45 { .. } => {
                attrs.push("rtol".to_string());
                attrs.push("atol".to_string());
                attrs.push("min_step".to_string());
                attrs.push("safety_factor".to_string());
            }
        }
        if has_optimizer(&self.0) {
            attrs.push("optimizer".to_string());
        }

        attrs
    }
}

fn get_step(solver: &Solver) -> Option<f64> {
    match solver {
        Solver::Euler { step } => Some(*step),
        Solver::RK4 { step } => Some(*step),
        Solver::ImplicitEuler { step, .. } => Some(*step),
        Solver::GLRK4 { step, .. } => Some(*step),
        Solver::ROW1 { step } => Some(*step),
        _ => None,
    }
}

fn get_optimizer(solver: &Solver) -> Option<Arc<dyn Optimizer + Send + Sync>> {
    match solver {
        Solver::ImplicitEuler { optimizer, .. } => Some(optimizer.clone()),
        Solver::GLRK4 { optimizer, .. } => Some(optimizer.clone()),
        _ => None,
    }
}

fn has_optimizer(solver: &Solver) -> bool {
    matches!(solver, Solver::ImplicitEuler { .. } | Solver::GLRK4 { .. })
}

fn get_rtol(solver: &Solver) -> Option<f64> {
    if let Solver::RKF45 { rtol, .. } = solver { Some(*rtol) } else { None }
}

fn get_atol(solver: &Solver) -> Option<f64> {
    if let Solver::RKF45 { atol, .. } = solver { Some(*atol) } else { None }
}

fn get_min_step(solver: &Solver) -> Option<f64> {
    if let Solver::RKF45 { min_step, .. } = solver { Some(*min_step) } else { None }
}

fn get_safety_factor(solver: &Solver) -> Option<f64> {
    if let Solver::RKF45 { safety_factor, .. } = solver { Some(*safety_factor) } else { None }
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
    PySolver(Solver::ImplicitEuler {
        step,
        optimizer: optimizer.0,
    })
}

#[pyfunction(name = "GLRK4MethodSolver")]
fn create_glrk4_solver(step: f64, optimizer: PyOptimizer) -> PySolver {
    PySolver(Solver::GLRK4 {
        step,
        optimizer: optimizer.0,
    })
}

#[pyfunction(name = "RKF45MethodSolver")]
fn create_rkf45_solver(rtol: f64, atol: f64, min_step: f64, safety_factor: f64) -> PySolver {
    PySolver(Solver::RKF45 {
        rtol,
        atol,
        min_step,
        safety_factor,
    })
}

#[pyfunction(name = "ROW1MethodSolver")]
fn create_row1_solver(step: f64) -> PySolver {
    PySolver(Solver::ROW1 { step })
}

fn convert_function<'py>(py: Python<'py>, f: &Bound<'py, PyAny>) -> PyResult<tch::CModule> {
    let torch = py.import("torch")?;
    let script_function_type = torch.getattr("jit")?.getattr("ScriptFunction")?;

    if !f.is_instance(&script_function_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Function must be a torch.jit.ScriptFunction",
        ));
    }

    let buffer = f.call_method0("save_to_buffer")?.extract::<Vec<u8>>()?;
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
    m.add_function(wrap_pyfunction!(create_newton, m)?)?;
    m.add_class::<PySolver>()?;
    m.add_class::<PyOptimizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
