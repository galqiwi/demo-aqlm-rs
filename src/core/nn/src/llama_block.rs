use crate::attention::Attention;
use crate::layernorm::LayerNorm;
use crate::linear::Module;
use crate::mlp::MLP;
use tensorlib::functional::add_rows;

pub struct LlamaBlockSubmodules<LinearType>
where
    LinearType: Module,
{
    pub input_layernorm: LayerNorm<'static>,
    pub attention: Attention<LinearType>,
    pub post_attention_layernorm: LayerNorm<'static>,
    pub mlp: MLP<LinearType>,
}

pub struct LlamaBlock<LinearType>
where
    LinearType: Module,
{
    submodules: LlamaBlockSubmodules<LinearType>,
}

impl<LinearType> LlamaBlock<LinearType>
where
    LinearType: Module,
{
    pub fn new(submodules: LlamaBlockSubmodules<LinearType>) -> Self {
        Self { submodules }
    }
}

impl<LinearType> LlamaBlock<LinearType>
where
    LinearType: Module,
{
    pub async fn forward(&mut self, x: Vec<f32>) -> Vec<f32> {
        let x = {
            let residual = x.clone();

            let x = self.submodules.input_layernorm.forward(x);

            let x = self.submodules.attention.forward(&x).await;

            add_rows(residual, x.data())
        };

        {
            let residual = x.clone();

            let x = self.submodules.post_attention_layernorm.forward(x);

            let x = self.submodules.mlp.forward(&x).await;

            add_rows(residual, &x)
        }
    }

    pub fn clear_cache(&mut self) {
        self.submodules.attention.clear_cache();
    }
}
