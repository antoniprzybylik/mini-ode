use std::fmt;
use tch::{Device, Kind, Tensor};

pub struct MockOptimizer;

impl crate::optimizers::Optimizer for MockOptimizer {
    fn optimize(&self, _f: &dyn Fn(&Tensor) -> Tensor, _x0: &Tensor) -> anyhow::Result<Tensor>
    {
        Ok(Tensor::zeros(&[1], (Kind::Float, Device::Cpu)))
    }
}

impl fmt::Display for MockOptimizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "MockOptimizer")
    }
}
