use mini_ode::optimizers::Optimizer;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use std::io::Cursor;

#[pyclass(module = "rust.optimizers", name = "Optimizer")]
struct PyOptimizer(Box<dyn Optimizer + Send + Sync>);

#[pyfunction(name = "CG")]
fn create_cg(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
    linesearch_atol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Box::new(mini_ode::optimizers::CG::new(
        max_steps,
        gtol,
        ftol,
        linesearch_atol,
    )))
}

#[pyfunction(name = "BFGS")]
fn create_bfgs(
    max_steps: usize,
    gtol: Option<f64>,
    ftol: Option<f64>,
    linesearch_atol: Option<f64>,
) -> PyOptimizer {
    PyOptimizer(Box::new(mini_ode::optimizers::BFGS::new(
        max_steps,
        gtol,
        ftol,
        linesearch_atol,
    )))
}

fn convert_function(py: Python, f: PyObject) -> PyResult<tch::CModule> {
    let torch = py.import_bound("torch")?;
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

#[pyfunction]
fn solve_euler(
    py: Python,
    f: PyObject,
    x_span: PyTensor,
    y0: PyTensor,
    step: PyTensor,
) -> PyResult<(PyTensor, PyTensor)> {
    let f_module = convert_function(py, f)?;
    mini_ode::solve_euler(f_module, x_span.0, y0.0, step.0)
        .map(|(x, y)| (PyTensor(x), PyTensor(y)))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn solve_rk4(
    py: Python,
    f: PyObject,
    x_span: PyTensor,
    y0: PyTensor,
    step: PyTensor,
) -> PyResult<(PyTensor, PyTensor)> {
    let f_module = convert_function(py, f)?;
    mini_ode::solve_rk4(f_module, x_span.0, y0.0, step.0)
        .map(|(x, y)| (PyTensor(x), PyTensor(y)))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn solve_implicit_euler(
    py: Python,
    f: PyObject,
    x_span: PyTensor,
    y0: PyTensor,
    step: PyTensor,
    optimizer: &PyOptimizer,
) -> PyResult<(PyTensor, PyTensor)> {
    let f_module = convert_function(py, f)?;
    let x_span_inner = x_span.0.copy();
    let y0_inner = y0.0.copy();
    let step_inner = step.0.copy();

    py.allow_threads(|| {
        mini_ode::solve_implicit_euler(
            f_module,
            x_span_inner,
            y0_inner,
            step_inner,
            optimizer.0.as_ref(),
        )
        .map(|(x, y)| (PyTensor(x), PyTensor(y)))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
fn solve_glrk4(
    py: Python,
    f: PyObject,
    x_span: PyTensor,
    y0: PyTensor,
    step: PyTensor,
    optimizer: &PyOptimizer,
) -> PyResult<(PyTensor, PyTensor)> {
    let f_module = convert_function(py, f)?;
    let x_span_inner = x_span.0.copy();
    let y0_inner = y0.0.copy();
    let step_inner = step.0.copy();

    py.allow_threads(|| {
        mini_ode::solve_glrk4(
            f_module,
            x_span_inner,
            y0_inner,
            step_inner,
            optimizer.0.as_ref(),
        )
        .map(|(x, y)| (PyTensor(x), PyTensor(y)))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

#[pyfunction]
fn solve_rkf45(
    py: Python,
    f: PyObject,
    x_span: PyTensor,
    y0: PyTensor,
    rtol: PyTensor,
    atol: PyTensor,
    min_step: PyTensor,
    safety_factor: f64,
) -> PyResult<(PyTensor, PyTensor)> {
    let f_module = convert_function(py, f)?;
    mini_ode::solve_rkf45(
        f_module,
        x_span.0,
        y0.0,
        rtol.0,
        atol.0,
        min_step.0,
        safety_factor,
    )
    .map(|(x, y)| (PyTensor(x), PyTensor(y)))
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn solve_row1(
    py: Python,
    f: PyObject,
    x_span: PyTensor,
    y0: PyTensor,
    step: PyTensor,
) -> PyResult<(PyTensor, PyTensor)> {
    let f_module = convert_function(py, f)?;
    let x_span_inner = x_span.0.copy();
    let y0_inner = y0.0.copy();
    let step_inner = step.0.copy();

    py.allow_threads(|| {
        mini_ode::solve_row1(f_module, x_span_inner, y0_inner, step_inner)
            .map(|(x, y)| (PyTensor(x), PyTensor(y)))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_euler, m)?)?;
    m.add_function(wrap_pyfunction!(solve_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(solve_implicit_euler, m)?)?;
    m.add_function(wrap_pyfunction!(solve_glrk4, m)?)?;
    m.add_function(wrap_pyfunction!(solve_rkf45, m)?)?;
    m.add_function(wrap_pyfunction!(solve_row1, m)?)?;
    m.add_function(wrap_pyfunction!(create_cg, m)?)?;
    m.add_function(wrap_pyfunction!(create_bfgs, m)?)?;
    m.add_class::<PyOptimizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
