use crate::handles::lock_handles;
use async_trait::async_trait;
use futures::future::join_all;
use nn::linear::Module;
use state_dict::from_state_dict::{load_f32_data, load_i8_data, FromStateDict};
use tensorlib::functional::cat_row;
use tensorlib::matrix::{Matrix, OwnedMatrix};

pub struct ParallelINT8Linear {
    name: String,
    out_dim: usize,
    in_dim: usize,
}

#[async_trait(?Send)]
impl FromStateDict for ParallelINT8Linear {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        let max_values = load_f32_data(&format!("{prefix}weight_max_values"))
            .await?
            .0;
        let int8_values = load_i8_data(&format!("{prefix}weight_int8")).await?.0;

        assert_eq!(int8_values.len() % max_values.len(), 0);

        let in_dim = max_values.len();
        let out_dim = int8_values.len() / max_values.len();

        let mut handles = lock_handles().await;

        let handlers = handles.borrow_mut();
        assert_ne!(handlers.len(), 0);

        let chunk_size = out_dim / handlers.len();
        let n_handlers = handlers.len();

        for (handler_idx, handler) in handlers.iter_mut().enumerate() {
            let begin = handler_idx * chunk_size;
            let end = match handler_idx == n_handlers - 1 {
                true => out_dim,
                false => (handler_idx + 1) * chunk_size,
            };

            let chunk_int8_values = &int8_values[begin * in_dim..end * in_dim];

            handler
                .add_int8(prefix.to_string(), &max_values, chunk_int8_values)
                .await;
        }

        Ok(ParallelINT8Linear {
            name: prefix.to_string(),
            out_dim,
            in_dim,
        })
    }
}

#[async_trait(?Send)]
impl Module for ParallelINT8Linear {
    async fn forward(&mut self, x: &[f32]) -> OwnedMatrix {
        let x = Matrix::from_slice((1, x.len()), x);

        let mut handlers = lock_handles().await;
        assert_ne!(handlers.borrow().len(), 0);

        let futures = handlers
            .borrow_mut()
            .iter_mut()
            .map(|handler| handler.int8_forward(&self.name, &x));

        let output = join_all(futures).await;

        cat_row(&output)
    }

    fn shape(&self) -> (usize, usize) {
        (self.out_dim, self.in_dim)
    }
}
