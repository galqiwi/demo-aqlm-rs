use crate::matrix::{Matrix, OwnedMatrix};

impl Matrix<'_> {
    pub fn transpose(&self) -> OwnedMatrix {
        let (n_rows, n_cols) = self.shape();

        let transposed_data: Vec<f32> = (0..n_cols)
            .flat_map(|col| (0..n_rows).map(move |row| self.data()[row * n_cols + col]))
            .collect();

        Matrix::from_vec((n_cols, n_rows), transposed_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let matrix = Matrix::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = matrix.transpose();
        assert_eq!(
            transposed,
            Matrix::from_vec((3, 2), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        );
    }
}
