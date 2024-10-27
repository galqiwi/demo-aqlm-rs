use crate::matrix::{Matrix, OwnedMatrix};

impl Matrix<'_> {
    pub fn get_row(&self, row_id: usize) -> &[f32] {
        &self.data()[row_id * self.n_cols()..(row_id + 1) * self.n_cols()]
    }

    pub fn get_rows(&self, row_ids: &[usize]) -> OwnedMatrix {
        let new_data: Vec<f32> = row_ids
            .iter()
            .flat_map(|&row_id| self.get_row(row_id).iter().cloned())
            .collect();

        Matrix::from_vec((row_ids.len(), self.n_cols()), new_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_rows() {
        let first = Matrix::from_vec((3, 2), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            first.get_rows(&[2, 0]),
            Matrix::from_vec((2, 2), vec![4.0, 5.0, 0.0, 1.0])
        );
        assert_eq!(first.get_row(1), vec![2.0, 3.0])
    }
}
