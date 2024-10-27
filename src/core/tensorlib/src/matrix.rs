use std::borrow::Cow;

pub type OwnedMatrix = Matrix<'static>;

#[derive(Debug, Clone)]
pub struct Matrix<'a> {
    shape: (usize, usize),
    data: Cow<'a, [f32]>,
}

impl<'a> Matrix<'a> {
    pub fn new(shape: (usize, usize), data: Cow<'a, [f32]>) -> Self {
        assert_eq!(shape.0 * shape.1, data.len());
        Matrix { shape, data }
    }

    pub fn from_slice(shape: (usize, usize), data: &'a [f32]) -> Self {
        Self::new(shape, Cow::Borrowed(data))
    }

    pub fn into_owned(self) -> OwnedMatrix {
        OwnedMatrix::new(self.shape, Cow::Owned(self.data.into_owned()))
    }
}

impl OwnedMatrix {
    pub fn from_vec(shape: (usize, usize), data: Vec<f32>) -> Self {
        Self::new(shape, Cow::Owned(data))
    }
}

impl<'a> Matrix<'a> {
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn n_rows(&self) -> usize {
        self.shape.0
    }

    pub fn n_cols(&self) -> usize {
        self.shape.1
    }

    pub fn data(&self) -> &Cow<'a, [f32]> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Cow<'a, [f32]> {
        &mut self.data
    }

    pub fn into_data(self) -> Cow<'a, [f32]> {
        self.data
    }
}

impl Matrix<'_> {
    pub fn reshape(self, shape: (usize, usize)) -> Self {
        assert_eq!(shape.0 * shape.1, self.n_rows() * self.n_cols());
        Matrix {
            data: self.data,
            shape,
        }
    }
}

impl PartialEq for Matrix<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        self.data() == other.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let m1 = Matrix::from_vec((3, 2), data.clone());
        assert_eq!(m1.shape(), (3, 2));
        assert_eq!(m1.n_rows(), 3);
        assert_eq!(m1.n_cols(), 2);

        let m2 = Matrix::from_slice((3, 2), &data);
        assert_eq!(m2.shape(), (3, 2));
        assert_eq!(m2.n_rows(), 3);
        assert_eq!(m2.n_cols(), 2);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_matrix_mut() {
        let mut m1 = Matrix::from_vec((3, 2), (0..6).map(|v| v as f32).collect());
        m1.data_mut()
            .to_mut()
            .iter_mut()
            .for_each(|v: &mut f32| *v += 1.0);

        assert_eq!(
            m1,
            Matrix::from_vec((3, 2), (0..6).map(|v| (v + 1) as f32).collect()),
        );
    }

    #[test]
    fn test_reshape() {
        let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let m = Matrix::from_vec((3, 2), data.clone());
        let m = m.reshape((2, 3));
        assert_eq!(m, Matrix::from_vec((2, 3), data));
    }
}
