use crate::attention::AttentionConfig;

pub static LLAMA_3_1_8B_CONFIG: LlamaConfig = LlamaConfig {
    dim: 4096,
    n_layers: 32,
    n_heads: 32,
    n_kv_heads: 8,
    norm_eps: 1e-5,
    rope_theta: 500000.0,
};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
}

impl LlamaConfig {
    pub fn to_attention_config(&self) -> AttentionConfig {
        AttentionConfig {
            dim: self.dim,
            head_dim: self.dim / self.n_heads,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            rope_theta: self.rope_theta,
        }
    }
}
