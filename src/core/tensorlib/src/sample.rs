use crate::matrix::{Matrix, OwnedMatrix};

impl Matrix<'_> {
    pub fn sample(
        &self,
        shape: (usize, usize),
        stride: (usize, usize),
        offset: usize,
    ) -> OwnedMatrix {
        let data = (0..shape.0 * shape.1)
            .map(|idx| (idx / shape.1, idx % shape.1))
            .map(|(y, x)| self.data()[y * stride.0 + x * stride.1 + offset]);

        OwnedMatrix::from_vec(shape, data.collect())
    }
}
