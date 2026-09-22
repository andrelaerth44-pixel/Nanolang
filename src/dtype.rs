#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        }
    }

    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::BF16),
            other => Err(format!("dtype desconhecido: '{other}'")),
        }
    }

    pub(crate) fn promote(a: Self, b: Self) -> Self {
        if a == Self::F32 || b == Self::F32 {
            Self::F32
        } else if a == Self::BF16 || b == Self::BF16 {
            Self::BF16
        } else {
            Self::F16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_parsing_and_size() {
        assert_eq!(DType::parse("f32").unwrap(), DType::F32);
        assert_eq!(DType::parse("F16").unwrap(), DType::F16);
        assert_eq!(DType::parse("bf16").unwrap(), DType::BF16);
        assert_eq!(DType::F32.bytes(), 4);
        assert_eq!(DType::F16.bytes(), 2);
        assert_eq!(DType::promote(DType::F16, DType::BF16), DType::BF16);
    }
}
