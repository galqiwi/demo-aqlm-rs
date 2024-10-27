use bytemuck::cast_slice;
use tensorlib::matrix::OwnedMatrix;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum Dtype {
    F32,
    U8,
    I8,
}

pub struct OwnedTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

pub fn get_f32_data(tensor: OwnedTensor) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(tensor.dtype, Dtype::F32);
    // Suboptimal reallocation.
    // Can be fixed with unsafe, but I don't want to
    let raw_slice: &[f32] = cast_slice(&tensor.data);
    (raw_slice.to_vec(), tensor.shape)
}

pub fn get_u8_data(tensor: OwnedTensor) -> (Vec<u8>, Vec<usize>) {
    assert_eq!(tensor.dtype, Dtype::U8);
    // Suboptimal reallocation.
    // Can be fixed with unsafe, but I don't want to
    let raw_slice: &[u8] = cast_slice(&tensor.data);
    (raw_slice.to_vec(), tensor.shape)
}

pub fn get_i8_data(tensor: OwnedTensor) -> (Vec<i8>, Vec<usize>) {
    assert_eq!(tensor.dtype, Dtype::I8);
    // Suboptimal reallocation.
    // Can be fixed with unsafe, but I don't want to
    let raw_slice: &[i8] = cast_slice(&tensor.data);
    (raw_slice.to_vec(), tensor.shape)
}

pub fn get_matrix(tensor: OwnedTensor) -> OwnedMatrix {
    assert_eq!(tensor.shape.len(), 2);
    let shape = (tensor.shape[0], tensor.shape[1]);

    let data = get_f32_data(tensor).0;

    assert_eq!(data.len(), shape.0 * shape.1);

    OwnedMatrix::from_vec(shape, data)
}
