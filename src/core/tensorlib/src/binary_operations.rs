use crate::matrix::{Matrix, OwnedMatrix};

impl Matrix<'_> {
    pub fn binary_operation<F>(self, other: &Matrix, operation: F) -> OwnedMatrix
    where
        F: Fn(&mut f32, f32, (usize, usize)),
    {
        assert_eq!(self.shape(), other.shape());

        let (n_rows, n_cols) = self.shape();

        let mut output = self.into_owned();

        output
            .data_mut()
            .to_mut()
            .iter_mut()
            .zip(other.data().iter())
            .zip((0..n_rows * n_cols).map(|i| (i / n_cols, i % n_cols)))
            .for_each(|((a, b), (y, x))| operation(a, *b, (y, x)));

        output
    }

    pub fn binary_operation_row<F>(self, other: &[f32], operation: F) -> OwnedMatrix
    where
        F: Fn(&mut f32, &f32, (usize, usize)),
    {
        assert_eq!(self.n_cols(), other.len());

        let (n_rows, n_cols) = self.shape();

        let mut output = self.into_owned();

        output
            .data_mut()
            .to_mut()
            .iter_mut()
            .zip(std::iter::repeat(other).take(n_rows).flatten())
            .zip((0..n_rows * n_cols).map(|i| (i / n_cols, i % n_cols)))
            .for_each(|((a, b), (y, x))| operation(a, b, (y, x)));

        output
    }
}

impl Matrix<'_> {
    pub fn add_matrix(self, other: &Matrix) -> OwnedMatrix {
        self.binary_operation(other, |a, b, _| *a += b)
    }

    pub fn add_row(self, other: &[f32]) -> OwnedMatrix {
        self.binary_operation_row(other, |a, b, _| *a += *b)
    }

    pub fn multiply(self, other: &Matrix) -> OwnedMatrix {
        self.binary_operation(other, |a, b, _| *a *= b)
    }

    pub fn multiply_row(self, other: &[f32]) -> OwnedMatrix {
        self.binary_operation_row(other, |a, b, _| *a *= *b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let matrix1 = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::from_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]);
        let expected = Matrix::from_vec((2, 2), vec![2.0, 3.0, 4.0, 5.0]);

        let result = matrix1.add_matrix(&matrix2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_row() {
        let matrix = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let row = vec![1.0, 2.0];
        let expected = Matrix::from_vec((2, 2), vec![2.0, 4.0, 4.0, 6.0]);

        let result = matrix.add_row(&row);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiply() {
        let matrix1 = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let expected = Matrix::from_vec((2, 2), vec![1.0, 4.0, 9.0, 16.0]);

        let result = matrix1.multiply(&matrix2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiply_row() {
        let matrix = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let row = vec![1.0, 2.0];
        let expected = Matrix::from_vec((2, 2), vec![1.0, 4.0, 3.0, 8.0]);

        let result = matrix.multiply_row(&row);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_add_dimension_mismatch() {
        let matrix1 = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::from_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        matrix1.add_matrix(&matrix2);
    }

    #[test]
    #[should_panic(expected = "assertion")]
    fn test_add_row_dimension_mismatch() {
        let matrix = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        let row = vec![1.0, 1.0, 1.0];

        matrix.add_row(&row);
    }
}
