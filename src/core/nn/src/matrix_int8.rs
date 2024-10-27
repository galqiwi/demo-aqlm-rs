use std::borrow::Cow;
use tensorlib::matrix::OwnedMatrix;

pub struct MatrixInt8<'a> {
    max_values: Cow<'a, [f32]>,
    int8_values: Cow<'a, [i8]>,
}

impl<'a> MatrixInt8<'a> {
    pub fn new(max_values: Cow<'a, [f32]>, int8_values: Cow<'a, [i8]>) -> Self {
        assert_eq!(int8_values.len() % max_values.len(), 0);
        Self {
            max_values,
            int8_values,
        }
    }
}

impl MatrixInt8<'_> {
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
    }

    pub fn n_cols(&self) -> usize {
        self.max_values.len()
    }

    pub fn n_rows(&self) -> usize {
        self.int8_values.len() / self.max_values.len()
    }
}

impl<'a> MatrixInt8<'a> {
    pub fn get_row(&mut self, row_idx: usize) -> Vec<f32> {
        let n_cols = self.n_cols();

        let output = &self.int8_values[row_idx * n_cols..(row_idx + 1) * n_cols];

        output
            .iter()
            .zip(self.max_values.iter())
            .map(|(v, max_value)| (*v as f32) * max_value / 127f32)
            .collect()
    }

    pub fn get_rows(&mut self, rows_idx: &[usize]) -> OwnedMatrix {
        let data: Vec<f32> = rows_idx.iter().flat_map(|idx| self.get_row(*idx)).collect();

        OwnedMatrix::from_vec((rows_idx.len(), self.n_cols()), data)
    }
}

impl<'a> MatrixInt8<'a> {
    pub fn matmul(&mut self, x: &[f32]) -> OwnedMatrix {
        let output = self.matmul_row(x);

        OwnedMatrix::from_vec((1, output.len()), output)
    }

    pub fn matmul_row(&mut self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n_cols());
        let n_cols = self.n_cols();

        assert_eq!(self.max_values.len(), n_cols);

        let x: Vec<f32> = x
            .iter()
            .zip(self.max_values.iter())
            .map(|(x, max_value)| *x * *max_value / 127f32)
            .collect();

        self.int8_values
            .chunks_exact(n_cols)
            .map(|row| {
                row.iter()
                    .zip(x.iter())
                    .map(|(a, b)| (*a as f32) * *b)
                    .sum()
            })
            .collect()
    }
}
