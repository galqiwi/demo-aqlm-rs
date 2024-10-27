use crate::matrix_int8::MatrixInt8;

pub struct EmbeddingINT8<'a> {
    weight: MatrixInt8<'a>,
}

impl<'a> EmbeddingINT8<'a> {
    pub fn new(weight: MatrixInt8<'a>) -> Self {
        Self { weight }
    }

    pub fn forward(&mut self, x: usize) -> Vec<f32> {
        self.weight.get_row(x)
    }
}
