use crate::linear::Module;
use crate::matrix_int8::MatrixInt8;
use async_trait::async_trait;
use tensorlib::matrix::OwnedMatrix;

pub struct LinearINT8<'a> {
    weight: MatrixInt8<'a>,
}

#[async_trait(?Send)]
impl Module for LinearINT8<'_> {
    async fn forward(&mut self, x: &[f32]) -> OwnedMatrix {
        self.weight.matmul(x)
    }

    fn shape(&self) -> (usize, usize) {
        self.weight.shape()
    }
}

impl<'a> LinearINT8<'a> {
    pub fn new(weight: MatrixInt8<'a>) -> Self {
        Self { weight }
    }
}
