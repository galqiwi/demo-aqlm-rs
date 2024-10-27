use crate::calib::get_optimal_aqlm_n_workers;
use crate::handles::lock_handles;
use crate::registry_rpc_handle::RPCLinearRegistryHandle;
use async_trait::async_trait;
use futures::future::join_all;
use nn::linear::Module;
use state_dict::from_state_dict::{load_f32_data, load_u8_data, FromStateDict};
use std::mem;
use tensorlib::functional::cat_row;
use tensorlib::matrix::{Matrix, OwnedMatrix};

pub async fn set_handles(mut new_handles: Vec<RPCLinearRegistryHandle>) {
    new_handles.truncate(8);

    let mut handles = lock_handles().await;
    let _ = mem::replace(handles.borrow_mut(), new_handles);
}

pub struct ParallelAQLMLinear {
    name: String,
    out_dim: usize,
    in_group_dim: usize,
    n_workers: usize,
}

impl ParallelAQLMLinear {
    pub async fn new(
        name: String,
        codebooks: &[f32],
        scales: &[f32],
        codes: &[u8],
        out_dim: usize,
        in_group_dim: usize,
        n_workers: usize,
    ) -> Self {
        let mut handles = lock_handles().await;
        let handles = handles.borrow_mut();
        let handles: &mut [RPCLinearRegistryHandle] = &mut handles[0..n_workers];
        assert_ne!(handles.len(), 0);

        let chunk_size = out_dim / handles.len();
        let n_handlers = handles.len();

        for (handler_idx, handler) in handles.iter_mut().enumerate() {
            let begin = handler_idx * chunk_size;
            let end = match handler_idx == n_handlers - 1 {
                true => out_dim,
                false => (handler_idx + 1) * chunk_size,
            };

            // codes: [in_group_idx, codebook_idx, out_idx]
            let chunk_codes: Vec<u8> = (0..in_group_dim * 2)
                .flat_map(|idx| {
                    codes[idx * out_dim + begin..idx * out_dim + end]
                        .iter()
                        .copied()
                })
                .collect();

            handler
                .add_aqlm(
                    name.clone(),
                    codebooks,
                    &scales[begin..end],
                    &chunk_codes,
                    end - begin,
                    in_group_dim,
                )
                .await;
        }

        Self {
            name,
            out_dim,
            in_group_dim,
            n_workers,
        }
    }
}

#[async_trait(?Send)]
impl FromStateDict for ParallelAQLMLinear {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        let codebooks_file = format!("{prefix}codebooks");
        let scales_file = format!("{prefix}scales");
        let codes_file = format!("{prefix}codes_120");

        let codebooks_future = load_f32_data(&codebooks_file);
        let scales_future = load_f32_data(&scales_file);
        let codes_future = load_u8_data(&codes_file);

        let (codebooks, codebooks_shape) = codebooks_future.await?;
        assert_eq!(codebooks_shape, vec![2, 256, 1, 8]);

        let (scales, _) = scales_future.await?;
        let (codes, codes_shape) = codes_future.await?;

        let out_dim = codes_shape[2];
        let in_group_dim = codes_shape[0];

        let n_layer_workers =
            get_optimal_aqlm_n_workers(&codebooks, &scales, &codes, out_dim, in_group_dim).await?;

        Ok(Self::new(
            prefix.to_string(),
            &codebooks,
            &scales,
            &codes,
            out_dim,
            in_group_dim,
            n_layer_workers,
        )
        .await)
    }
}

#[async_trait(?Send)]
impl Module for ParallelAQLMLinear {
    async fn forward(&mut self, x: &[f32]) -> OwnedMatrix {
        let x = Matrix::from_slice((1, x.len()), x);

        let n_workers = self.n_workers;

        let mut handles = lock_handles().await;
        assert_ne!(handles.borrow().len(), 0);
        let handles: &mut [RPCLinearRegistryHandle] = handles.borrow_mut();

        let futures = handles
            .iter_mut()
            .take(n_workers)
            .map(|handler| handler.aqlm_forward(&self.name, &x));

        let output = join_all(futures).await;

        cat_row(&output)
    }

    fn shape(&self) -> (usize, usize) {
        (self.out_dim, self.in_group_dim * 8)
    }
}

impl ParallelAQLMLinear {
    pub async fn async_drop(&mut self) {
        let n_workers = self.n_workers;

        let mut handles = lock_handles().await;
        assert_ne!(handles.borrow().len(), 0);
        let handles: &mut [RPCLinearRegistryHandle] = handles.borrow_mut();

        let futures = handles
            .iter_mut()
            .take(n_workers)
            .map(|handler| handler.remove_aqlm(self.name.clone()));

        join_all(futures).await;
    }
}
