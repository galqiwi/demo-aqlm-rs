use crate::embedding::EmbeddingINT8;
use crate::layernorm::LayerNorm;
use crate::linear::Module;
use crate::llama_block::LlamaBlock;

pub struct LlamaSubmodules<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    pub embed_tokens: EmbeddingINT8<'static>,
    pub blocks: Vec<LlamaBlock<BlockLinearType>>,
    pub norm: LayerNorm<'static>,
    pub lm_head: HeadLinearType,
}

pub struct Llama<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    submodules: LlamaSubmodules<BlockLinearType, HeadLinearType>,
}

impl<BlockLinearType, HeadLinearType> Llama<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    pub fn new(submodules: LlamaSubmodules<BlockLinearType, HeadLinearType>) -> Self {
        Self { submodules }
    }
}

impl<BlockLinearType, HeadLinearType> Llama<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    pub async fn forward(&mut self, token: usize) -> Vec<f32> {
        let (embed_tokens, blocks, norm, lm_head) = (
            &mut self.submodules.embed_tokens,
            &mut self.submodules.blocks,
            &mut self.submodules.norm,
            &mut self.submodules.lm_head,
        );
        let mut x = embed_tokens.forward(token);

        for block in blocks {
            x = block.forward(x).await;
        }

        let x = norm.forward(x);
        // let x = Matrix::from_vec((1, x.len()), x);

        let x = lm_head.forward(&x).await;
        assert_eq!(x.n_rows(), 1);

        x.into_data().into_owned()
    }

    pub fn clear_cache(&mut self) {
        self.submodules
            .blocks
            .iter_mut()
            .for_each(|block| block.clear_cache());
    }
}
