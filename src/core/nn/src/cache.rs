// use crate::linear::Module;
// use tensorlib::matrix::Matrix;
//
// // Cache does not yet support invalidation
// #[derive(Default)]
// pub struct Cache {
//     data: Vec<f32>,
// }
//
// impl Cache {
//     pub async fn forward<'a, 'b, ModuleType: Module>(
//         &'a mut self,
//         module: &'b mut ModuleType,
//         x: &'b Matrix<'_>,
//     ) -> Matrix<'a> {
//         let x = x.get_rows(&[x.n_rows() - 1]);
//
//         let out_dim = module.shape().0;
//
//         let new_data = module.forward(&x).await;
//         self.data.extend(new_data.data().iter());
//
//         Matrix::from_slice((self.data.len() / out_dim, out_dim), &self.data)
//     }
// }
