use crate::dtype::DType;

pub(crate) const DEFAULT_ALIGNMENT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryBlock {
    pub(crate) offset: usize,
    pub(crate) size: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryPlanner {
    alignment: usize,
    cursor: usize,
    free: Vec<MemoryBlock>,
}

impl MemoryPlanner {
    pub(crate) fn new() -> Self {
        Self { alignment: DEFAULT_ALIGNMENT, cursor: 0, free: Vec::new() }
    }

    pub(crate) fn align_up(&self, bytes: usize) -> usize {
        let mask = self.alignment - 1;
        (bytes + mask) & !mask
    }

    pub(crate) fn bytes_for(&self, elements: usize, dtype: DType) -> usize {
        self.align_up(elements.saturating_mul(dtype.bytes()))
    }

    pub(crate) fn allocate(&mut self, elements: usize, dtype: DType) -> MemoryBlock {
        let needed = self.bytes_for(elements, dtype);
        if let Some(index) = self.free.iter().position(|block| block.size >= needed) {
            return self.free.swap_remove(index);
        }
        let block = MemoryBlock { offset: self.cursor, size: needed };
        self.cursor = self.cursor.saturating_add(needed);
        block
    }

    pub(crate) fn release(&mut self, block: MemoryBlock) {
        self.free.push(block);
    }

    pub(crate) fn planned_bytes(&self) -> usize { self.cursor }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_reuses_aligned_blocks() {
        let mut planner = MemoryPlanner::new();
        let first = planner.allocate(10, DType::F16);
        assert_eq!(first.size, DEFAULT_ALIGNMENT);
        planner.release(first);
        let second = planner.allocate(10, DType::F16);
        assert_eq!(second.offset, first.offset);
    }
}
