use serde::{Deserialize, Serialize};
use speedy::{Readable, Writable};
use std::borrow::Cow;
use tensorlib::matrix::Matrix;

#[derive(Serialize, Deserialize, Readable, Writable)]
pub struct SerdeMatrix<'a> {
    pub data: Cow<'a, [f32]>,
    pub shape: (usize, usize),
}

impl<'a> From<SerdeMatrix<'a>> for Matrix<'a> {
    fn from(value: SerdeMatrix<'a>) -> Self {
        Self::new(value.shape, value.data)
    }
}

impl From<Matrix<'_>> for SerdeMatrix<'static> {
    fn from(value: Matrix) -> Self {
        let shape = value.shape();
        let data = value.into_data();
        let data: Cow<'static, [f32]> = Cow::Owned(data.into_owned());

        SerdeMatrix { shape, data }
    }
}

impl From<&Matrix<'_>> for SerdeMatrix<'static> {
    fn from(value: &Matrix) -> Self {
        let shape = value.shape();
        let data = value.data().clone();
        let data: Cow<'static, [f32]> = Cow::Owned(data.into_owned());

        SerdeMatrix { shape, data }
    }
}
