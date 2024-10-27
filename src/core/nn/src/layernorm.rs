use std::borrow::Cow;

pub struct LayerNorm<'a> {
    weight: Cow<'a, [f32]>,
    norm_eps: f32,
}

impl<'a> LayerNorm<'a> {
    pub fn new(weight: Cow<'a, [f32]>, norm_eps: f32) -> Self {
        LayerNorm { weight, norm_eps }
    }
}

impl LayerNorm<'_> {
    pub fn forward(&mut self, x: Vec<f32>) -> Vec<f32> {
        rms_norm_row(x, &self.weight, self.norm_eps)
    }
}

fn rms_norm_row(input: Vec<f32>, norm_weights: &[f32], norm_eps: f32) -> Vec<f32> {
    assert_eq!(input.len(), norm_weights.len());

    let squared_norm: f32 = input.iter().map(|x| x * x).sum();
    let input_inv_norm = 1f32 / ((squared_norm / input.len() as f32) + norm_eps).sqrt();

    input
        .into_iter()
        .zip(norm_weights)
        .map(|(x, norm)| x * *norm * input_inv_norm)
        .collect()
}
