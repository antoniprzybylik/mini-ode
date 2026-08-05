#[cfg(feature = "warnings")]
pub(crate) use tracing::warn;

#[cfg(not(feature = "warnings"))]
macro_rules! warn_macro {
    ($($arg:tt)*) => {};
}

#[cfg(not(feature = "warnings"))]
pub(crate) use warn_macro as warn;
