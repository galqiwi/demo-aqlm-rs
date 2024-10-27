use crate::matrix::{Matrix, OwnedMatrix};

use ndarray::ArrayView2;

impl Matrix<'_> {
    pub fn matmul(&self, second: &Matrix) -> OwnedMatrix {
        let first_matrix = ArrayView2::from_shape(self.shape(), self.data()).unwrap();
        let second_matrix = ArrayView2::from_shape(second.shape(), second.data()).unwrap();

        let result_matrix = first_matrix.dot(&second_matrix.t());

        let (result_vec, _) = result_matrix.into_raw_vec_and_offset();
        Matrix::from_vec((self.shape().0, second.shape().0), result_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let first = Matrix::from_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]);
        let second = Matrix::from_vec((3, 2), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let matmul = first.matmul(&second);
        assert_eq!(
            matmul,
            Matrix::from_vec((2, 3), vec![5.0, 7.0, 9.0, 23.0, 33.0, 43.0])
        );
    }
}
