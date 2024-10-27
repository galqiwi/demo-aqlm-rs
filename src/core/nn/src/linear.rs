use async_trait::async_trait;
use tensorlib::matrix::{Matrix, OwnedMatrix};

#[async_trait(?Send)]
pub trait Forward {
    async fn forward(&mut self, x: &Matrix) -> OwnedMatrix;
}

#[async_trait(?Send)]
pub trait Module {
    async fn forward(&mut self, x: &[f32]) -> OwnedMatrix;
    fn shape(&self) -> (usize, usize);
}
