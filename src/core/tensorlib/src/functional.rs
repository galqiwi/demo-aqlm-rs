use crate::matrix::{Matrix, OwnedMatrix};

pub fn linear(input: &Matrix, weight: &Matrix) -> OwnedMatrix {
    input.matmul(weight)
}

pub fn cat_row(matrices: &[Matrix]) -> OwnedMatrix {
    assert!(
        !matrices.is_empty(),
        "No matrices provided for concatenation"
    );

    let n_rows = matrices[0].n_rows();

    if !matrices.iter().all(|m| m.n_rows() == n_rows) {
        panic!("All matrices must have the same number of rows");
    }

    assert_eq!(n_rows, 1);

    if n_rows == 1 {
        let data: Vec<f32> = matrices
            .iter()
            .flat_map(|m| m.data().iter().copied())
            .collect();
        return Matrix::from_vec((1, data.len()), data);
    }

    let total_cols: usize = matrices.iter().map(|m| m.n_cols()).sum();

    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| matrices.iter().flat_map(move |m| m.get_row(row)).cloned())
        .collect();

    Matrix::from_vec((n_rows, total_cols), data)
}

pub fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn argmin(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

pub fn add_rows(mut first: Vec<f32>, second: &[f32]) -> Vec<f32> {
    first
        .iter_mut()
        .zip(second.iter())
        .for_each(|(f, s)| *f += s);
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cat_row_basic() {
        // Create some sample matrices
        let matrix1 = Matrix::from_slice((2, 2), &[1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::from_slice((2, 3), &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let matrix3 = Matrix::from_slice((2, 1), &[11.0, 12.0]);

        // Concatenate the matrices
        let result = cat_row(&[matrix1, matrix2, matrix3]);

        // Define the expected result
        let expected = OwnedMatrix::from_vec(
            (2, 6),
            vec![
                1.0, 2.0, 5.0, 6.0, 7.0, 11.0, // First row
                3.0, 4.0, 8.0, 9.0, 10.0, 12.0, // Second row
            ],
        );

        // Assert that the result matches the expectation
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "No matrices provided for concatenation")]
    fn test_cat_row_empty() {
        // Concatenate an empty array of matrices
        let _ = cat_row(&[]);
    }

    #[test]
    #[should_panic(expected = "All matrices must have the same number of rows")]
    fn test_cat_row_unequal_rows() {
        // Create matrices with different number of rows
        let matrix1 = Matrix::from_slice((2, 2), &[1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::from_slice((3, 2), &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // This should panic
        let _ = cat_row(&[matrix1, matrix2]);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 2.0, 1.0]), 2);
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 4.0, 1.0]), 3);
    }
}
