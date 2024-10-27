use crate::matrix::Matrix;

impl<'a> Matrix<'a> {
    pub fn scalar_operation<F>(mut self, operation: F) -> Self
    where
        F: Fn(&mut f32, (usize, usize)),
    {
        let (n_rows, n_cols) = self.shape();

        self.data_mut()
            .to_mut()
            .iter_mut()
            .zip((0..n_rows * n_cols).map(|i| (i / n_cols, i % n_cols)))
            .for_each(|(v, (y, x))| operation(v, (y, x)));

        self
    }

    pub fn scalar_operation_row<F>(mut self, operation: F) -> Self
    where
        F: Fn((&mut [f32], usize)),
    {
        let (n_rows, n_cols) = self.shape();

        self.data_mut()
            .to_mut()
            .chunks_mut(n_cols)
            .zip(0..n_rows)
            .for_each(operation);

        self
    }
}

impl Matrix<'_> {
    pub fn add_scalar(self, value: f32) -> Self {
        self.scalar_operation(|x, _| *x += value)
    }

    pub fn multiply_scalar(self, value: f32) -> Self {
        self.scalar_operation(|x, _| *x *= value)
    }

    pub fn squared(self) -> Self {
        self.scalar_operation(|x, _| *x *= *x)
    }

    pub fn rsqrt(self) -> Self {
        self.scalar_operation(|x, _| *x = 1.0f32 / x.sqrt())
    }
}

impl Matrix<'_> {
    pub fn mean_row_keepdim(self) -> Self {
        self.scalar_operation_row(|(row, _)| {
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            row.iter_mut().for_each(|v| *v = mean);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_scalar() {
        let mat = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);

        let result = mat.add_scalar(1.0f32);
        let expected = Matrix::from_vec((2, 2), vec![2.0, 3.0, 4.0, 5.0]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiply_scalar() {
        let mat = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);

        let result = mat.multiply_scalar(2.0f32);
        let expected = Matrix::from_vec((2, 2), vec![2.0, 4.0, 6.0, 8.0]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_squared() {
        let mat = Matrix::from_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]);

        let result = mat.squared();
        let expected = Matrix::from_vec((2, 2), vec![1.0, 4.0, 9.0, 16.0]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_rsqrt() {
        let mat = Matrix::from_vec((2, 1), vec![1.0, 0.25]);

        let result = mat.rsqrt();
        let expected = Matrix::from_vec((2, 1), vec![1.0, 2.0]);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_mean_row_keepdim() {
        let matrix = Matrix::from_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            matrix.mean_row_keepdim(),
            Matrix::from_vec((2, 3), vec![2.0, 2.0, 2.0, 5.0, 5.0, 5.0])
        );
    }
}
