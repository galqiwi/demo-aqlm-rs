use crate::functional::softmax_row;
use crate::linear::Module;
use tensorlib::functional::{cat_row, linear};
use tensorlib::matrix::{Matrix, OwnedMatrix};

pub struct AttentionConfig {
    pub dim: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub rope_theta: f32,
}

impl AttentionConfig {
    pub fn get_emb_config(&self) -> RotaryEmbeddingConfig {
        RotaryEmbeddingConfig {
            head_dim: self.head_dim,
            rope_theta: self.rope_theta,
        }
    }
}

pub struct AttentionSubmodules<LinearType>
where
    LinearType: Module,
{
    pub v_proj: CachedAttentionLinear<LinearType>,
    pub q_proj: LinearType,
    pub k_proj: CachedAttentionLinear<LinearType>,
    pub o_proj: LinearType,
}

pub struct Attention<LinearType>
where
    LinearType: Module,
{
    submodules: AttentionSubmodules<LinearType>,
    config: AttentionConfig,
}

impl<LinearType> Attention<LinearType>
where
    LinearType: Module,
{
    pub fn new(submodules: AttentionSubmodules<LinearType>, config: AttentionConfig) -> Self {
        Attention { submodules, config }
    }
}

impl<LinearType> Attention<LinearType>
where
    LinearType: Module,
{
    pub async fn forward(&mut self, x: &[f32]) -> Matrix {
        let config = &self.config;
        let (head_dim, n_heads, n_kv_heads, rope_theta) = (
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            config.rope_theta,
        );

        let n_cached_tokens = self.submodules.k_proj.n_cached_tokens();
        assert_eq!(self.submodules.v_proj.n_cached_tokens(), n_cached_tokens,);
        let n_tokens = n_cached_tokens + 1;

        let tokens_q_proj = self.submodules.q_proj.forward(x).await;
        let tokens_k_proj = self.submodules.k_proj.forward(x).await;
        let tokens_v_proj = self.submodules.v_proj.forward(x).await;

        assert_eq!(tokens_q_proj.shape(), (1, head_dim * n_heads));
        assert_eq!(tokens_k_proj.shape(), (n_tokens, head_dim * n_kv_heads));
        assert_eq!(tokens_v_proj.shape(), (n_tokens, head_dim * n_kv_heads));

        let tokens_q_proj =
            rotate_tokens_proj(tokens_q_proj, head_dim, rope_theta, n_cached_tokens);
        assert_eq!(tokens_q_proj.shape(), (1, n_heads * head_dim));

        let qkv_heads: Vec<OwnedMatrix> = (0..n_heads)
            .map(|head_idx| {
                let q_proj_head = Self::get_proj_head(config, &tokens_q_proj, head_idx);
                let k_proj_head =
                    Self::get_proj_head(config, &tokens_k_proj, head_idx / (n_heads / n_kv_heads));
                let v_proj_head =
                    Self::get_proj_head(config, &tokens_v_proj, head_idx / (n_heads / n_kv_heads));

                assert_eq!(q_proj_head.shape(), (1, head_dim));
                assert_eq!(k_proj_head.shape(), (n_tokens, head_dim));
                assert_eq!(v_proj_head.shape(), (n_tokens, head_dim));

                let q_proj_head_last = Matrix::from_slice((1, head_dim), q_proj_head.data());

                let q_last_k = linear(&q_proj_head_last, &k_proj_head);
                assert_eq!(q_last_k.shape(), (1, n_tokens));

                let q_last_k = q_last_k.multiply_scalar(1.0f32 / (head_dim as f32).sqrt());
                assert_eq!(q_last_k.shape(), (1, n_tokens));

                let q_last_k = softmax_row(q_last_k);

                linear(&q_last_k, &v_proj_head.transpose())
            })
            .collect();

        // TODO: matrix!!!
        let q_last_kv = cat_row(&qkv_heads).into_data().into_owned();
        let output_last = self.submodules.o_proj.forward(&q_last_kv).await;
        assert_eq!(output_last.shape(), (1, x.len()));

        output_last
    }

    fn get_proj_head(config: &AttentionConfig, tokens_proj: &Matrix, head: usize) -> OwnedMatrix {
        let n_tokens = tokens_proj.n_rows();
        let head_dim = config.head_dim;
        let n_heads = tokens_proj.n_cols() / head_dim;

        tokens_proj.sample(
            (n_tokens, head_dim),
            (n_heads * head_dim, 1),
            head * head_dim,
        )
    }

    pub fn clear_cache(&mut self) {
        self.submodules.v_proj.cache.clear();
        self.submodules.k_proj.cache.clear();
    }
}

pub struct RotaryEmbeddingConfig {
    head_dim: usize,
    rope_theta: f32,
}

pub struct CachedAttentionLinear<LinearType: Module> {
    cache: Vec<f32>,
    inner: LinearType,
    emb_config: Option<RotaryEmbeddingConfig>,
}

impl<LinearType: Module> CachedAttentionLinear<LinearType> {
    pub fn n_cached_tokens(&self) -> usize {
        self.cache.len() / self.inner.shape().0
    }
}

impl<LinearType: Module> CachedAttentionLinear<LinearType> {
    pub fn new(inner: LinearType, emb_config: Option<RotaryEmbeddingConfig>) -> Self {
        Self {
            cache: Vec::new(),
            inner,
            emb_config,
        }
    }
}

impl<LinearType: Module> CachedAttentionLinear<LinearType> {
    async fn forward<'a, 'b>(&'a mut self, x: &'b [f32]) -> Matrix<'a> {
        let out_dim = self.inner.shape().0;
        let cached_tokens = self.cache.len() / out_dim;

        let new_data = self.inner.forward(x).await;

        let new_data = match &self.emb_config {
            None => new_data,
            Some(conf) => {
                rotate_tokens_proj(new_data, conf.head_dim, conf.rope_theta, cached_tokens)
            }
        };

        self.cache.extend(new_data.data().iter());

        Matrix::from_slice((self.cache.len() / out_dim, out_dim), &self.cache)
    }
}

fn rotate_tokens_proj(x: Matrix, head_dim: usize, rope_theta: f32, token_offset: usize) -> Matrix {
    let (n_rows, n_cols) = x.shape();

    let n_tokens = n_rows;
    let n_heads = n_cols / head_dim;

    assert_eq!(x.shape(), (n_tokens, n_heads * head_dim));

    let x = x.reshape((n_tokens * n_heads, head_dim));
    let x_rotated = rotate_half(&x);

    let get_angle = |y: usize, x: usize| -> f32 {
        let head_dim_idx = x % (head_dim / 2);
        let token_idx = y / n_heads + token_offset;

        (token_idx as f32) / rope_theta.powf((head_dim_idx as f32) / ((head_dim / 2) as f32))
    };

    let x = x.scalar_operation(|v, (y, x)| *v *= get_angle(y, x).cos());
    let x_rotated = x_rotated.scalar_operation(|v, (y, x)| *v *= get_angle(y, x).sin());

    x.add_matrix(&x_rotated)
        .reshape((n_tokens, n_heads * head_dim))
}

fn rotate_half(x: &Matrix) -> OwnedMatrix {
    let (n_rows, n_cols) = x.shape();

    let n_cols_half = x.n_cols() / 2;

    let rotated_data: Vec<f32> = x
        .data()
        .chunks_exact(n_cols)
        .flat_map(|row| {
            let (first_half, second_half) = row.split_at(n_cols_half);
            second_half
                .iter()
                .map(|x| -x)
                .chain(first_half.iter().copied())
        })
        .collect();

    Matrix::from_vec((n_rows, n_cols), rotated_data)
}
