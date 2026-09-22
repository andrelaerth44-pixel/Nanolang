use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendKind {
    Cpu,
    Gpu,
}

impl BackendKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BackendError(pub(crate) String);

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub(crate) trait TensorBackend {
    fn kind(&self) -> BackendKind;

    fn matmul(
        &self,
        left: &[f32],
        left_shape: &[usize],
        right: &[f32],
        right_shape: &[usize],
    ) -> Result<Vec<f32>, BackendError>;

    fn elementwise(
        &self,
        left: &[f32],
        right: &[f32],
        shape: &[usize],
        op: ElementwiseOp,
    ) -> Result<Vec<f32>, BackendError>;

    fn reduce(&self, data: &[f32], mean: bool) -> Result<f32, BackendError>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub(crate) struct CpuBackend;

impl TensorBackend for CpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
    }

    fn matmul(
        &self,
        left: &[f32],
        left_shape: &[usize],
        right: &[f32],
        right_shape: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        if left_shape.len() != 2 || right_shape.len() != 2 {
            return Err(BackendError("matmul CPU requer tensores 2D".into()));
        }

        let (m, k) = (left_shape[0], left_shape[1]);
        let (k2, n) = (right_shape[0], right_shape[1]);
        if k != k2 {
            return Err(BackendError(format!(
                "matmul CPU incompatível: {}x{} com {}x{}",
                m, k, k2, n
            )));
        }

        if left.len() != m * k || right.len() != k2 * n {
            return Err(BackendError("matmul CPU recebeu buffers incompatíveis com o shape".into()));
        }

        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0_f32;
                for x in 0..k {
                    sum += left[i * k + x] * right[x * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Ok(out)
    }

    fn elementwise(
        &self,
        left: &[f32],
        right: &[f32],
        shape: &[usize],
        op: ElementwiseOp,
    ) -> Result<Vec<f32>, BackendError> {
        let expected = shape.iter().copied().product::<usize>();
        if left.len() != expected || right.len() != expected || left.len() != right.len() {
            return Err(BackendError("elementwise CPU recebeu buffers incompatíveis com o shape".into()));
        }

        Ok(left.iter().zip(right).map(|(x, y)| match op {
            ElementwiseOp::Add => x + y,
            ElementwiseOp::Sub => x - y,
            ElementwiseOp::Mul => x * y,
            ElementwiseOp::Div => x / y,
        }).collect())
    }

    fn reduce(&self, data: &[f32], mean: bool) -> Result<f32, BackendError> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let sum = data.iter().copied().sum::<f32>();
        Ok(if mean { sum / data.len() as f32 } else { sum })
    }
}

pub(crate) fn cpu() -> CpuBackend {
    CpuBackend
}
