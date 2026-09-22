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

    pub(crate) fn parse(value: &str) -> Result<Self, BackendError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(BackendError(format!("backend desconhecido: '{other}'"))),
        }
    }
}

pub(crate) fn create(kind: BackendKind) -> Result<Box<dyn TensorBackend>, BackendError> {
    match kind {
        BackendKind::Cpu => Ok(Box::new(CpuBackend)),
        BackendKind::Gpu => Ok(Box::new(crate::gpu::GpuBackend::new()?)),
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

    fn matmul_resident(
        &self,
        _left_id: u64, left: &[f32], left_shape: &[usize],
        _right_id: u64, right: &[f32], right_shape: &[usize],
        _output_id: u64,
    ) -> Result<Vec<f32>, BackendError> {
        self.matmul(left, left_shape, right, right_shape)
    }

    fn elementwise(
        &self,
        left: &[f32],
        right: &[f32],
        shape: &[usize],
        op: ElementwiseOp,
    ) -> Result<Vec<f32>, BackendError>;

    fn elementwise_resident(
        &self,
        _left_id: u64, left: &[f32],
        _right_id: u64, right: &[f32],
        shape: &[usize], op: ElementwiseOp,
        _output_id: u64,
    ) -> Result<Vec<f32>, BackendError> {
        self.elementwise(left, right, shape, op)
    }

    fn fused_mul_add(
        &self,
        left: &[f32],
        right: &[f32],
        bias: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, BackendError>;

    fn fused_mul_add_resident(
        &self,
        _left_id: u64, left: &[f32],
        _right_id: u64, right: &[f32],
        _bias_id: u64, bias: &[f32],
        shape: &[usize], _output_id: u64,
    ) -> Result<Vec<f32>, BackendError> {
        self.fused_mul_add(left, right, bias, shape)
    }

    fn reduce(&self, data: &[f32], mean: bool) -> Result<f32, BackendError>;

    fn sync_tensor(&self, _id: u64, _data: &[f32]) -> Result<(), BackendError> { Ok(()) }
    fn release_tensor(&self, _id: u64) -> Result<(), BackendError> { Ok(()) }

    fn transfer(&self, data: &[f32]) -> Result<Vec<f32>, BackendError>;
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

    fn fused_mul_add(
        &self,
        left: &[f32],
        right: &[f32],
        bias: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, BackendError> {
        let expected = shape.iter().copied().product::<usize>();
        if left.len() != expected || right.len() != expected || bias.len() != expected {
            return Err(BackendError("fused_mul_add CPU recebeu buffers incompatíveis com o shape".into()));
        }
        Ok(left.iter()
            .zip(right)
            .zip(bias)
            .map(|((a, b), c)| a.mul_add(*b, *c))
            .collect())
    }

    fn reduce(&self, data: &[f32], mean: bool) -> Result<f32, BackendError> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let sum = data.iter().copied().sum::<f32>();
        Ok(if mean { sum / data.len() as f32 } else { sum })
    }

    fn transfer(&self, data: &[f32]) -> Result<Vec<f32>, BackendError> {
        Ok(data.to_vec())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_backend_names() {
        assert_eq!(BackendKind::parse("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::parse("GPU").unwrap(), BackendKind::Gpu);
        assert!(BackendKind::parse("npu").is_err());
    }

    #[test]
    fn cpu_matmul_is_correct() {
        let backend = CpuBackend;
        let out = backend
            .matmul(
                &[1.0, 2.0, 3.0, 4.0],
                &[2, 2],
                &[5.0, 6.0, 7.0, 8.0],
                &[2, 2],
            )
            .unwrap();
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn cpu_elementwise_and_reduce_are_correct() {
        let backend = CpuBackend;
        let add = backend
            .elementwise(
                &[1.0, 2.0, 3.0],
                &[4.0, 5.0, 6.0],
                &[3],
                ElementwiseOp::Add,
            )
            .unwrap();
        assert_eq!(add, vec![5.0, 7.0, 9.0]);

        assert_eq!(backend.reduce(&[2.0, 4.0, 6.0], false).unwrap(), 12.0);
        assert_eq!(backend.reduce(&[2.0, 4.0, 6.0], true).unwrap(), 4.0);
    }

    #[test]
    fn cpu_fused_mul_add_is_correct() {
        let backend = CpuBackend;
        let out = backend
            .fused_mul_add(
                &[1.0, 2.0, 3.0],
                &[4.0, 5.0, 6.0],
                &[7.0, 8.0, 9.0],
                &[3],
            )
            .unwrap();
        assert_eq!(out, vec![11.0, 18.0, 27.0]);
    }

    #[test]
    fn gpu_backend_is_registered() {
        assert_eq!(BackendKind::Gpu.name(), "gpu");
    }
}
