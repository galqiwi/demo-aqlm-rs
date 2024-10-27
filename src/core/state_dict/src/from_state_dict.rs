use crate::owned_tensor::{get_f32_data, get_i8_data, get_u8_data, Dtype, OwnedTensor};
use anyhow::anyhow;
use async_trait::async_trait;
use nn::attention::{Attention, AttentionConfig, AttentionSubmodules, CachedAttentionLinear};
use nn::embedding::EmbeddingINT8;
use nn::layernorm::LayerNorm;
use nn::linear::Module;
use nn::linear_aqlm::LinearAQLM;
use nn::linear_int8::LinearINT8;
use nn::llama_block::{LlamaBlock, LlamaBlockSubmodules};
use nn::llama_config::LlamaConfig;
use nn::matrix_int8::MatrixInt8;
use nn::mlp::{MLPSubmodules, MLP};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::borrow::Cow;
use tokio::join;
use tokio::sync::mpsc;

#[async_trait(?Send)]
pub trait FromStateDictConf<ConfigType>: Sized {
    async fn from_state_dict(prefix: &str, config: ConfigType) -> anyhow::Result<Self>;
}

#[async_trait(?Send)]
pub trait FromStateDict: Sized {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self>;
}

#[async_trait(?Send)]
impl FromStateDictConf<f32> for LayerNorm<'static> {
    async fn from_state_dict(prefix: &str, norm_eps: f32) -> anyhow::Result<Self> {
        Ok(LayerNorm::new(
            Cow::Owned(load_f32_data(&format!("{prefix}weight")).await?.0),
            norm_eps,
        ))
    }
}

#[async_trait(?Send)]
impl FromStateDict for MatrixInt8<'static> {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        let max_values = load_f32_data(&format!("{prefix}weight_max_values"))
            .await?
            .0;
        let int8_values = load_i8_data(&format!("{prefix}weight_int8")).await?.0;

        let max_values = Cow::Owned(max_values);
        let int8_values = Cow::Owned(int8_values);

        Ok(MatrixInt8::new(max_values, int8_values))
    }
}

#[async_trait(?Send)]
impl FromStateDict for EmbeddingINT8<'static> {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        let matrix = MatrixInt8::from_state_dict(prefix).await?;
        Ok(EmbeddingINT8::new(matrix))
    }
}

#[async_trait(?Send)]
impl FromStateDict for LinearINT8<'static> {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        let matrix = MatrixInt8::from_state_dict(prefix).await?;
        Ok(LinearINT8::new(matrix))
    }
}

#[async_trait(?Send)]
impl FromStateDict for LinearAQLM<'static> {
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        let (codebooks, codebooks_shape) = load_f32_data(&format!("{prefix}codebooks")).await?;
        assert_eq!(codebooks_shape, vec![2, 256, 1, 8]);

        let (scales, _) = load_f32_data(&format!("{prefix}scales")).await?;
        let (codes, codes_shape) = load_u8_data(&format!("{prefix}codes_120")).await?;

        let out_dim = codes_shape[2];
        let in_group_dim = codes_shape[0];

        Ok(LinearAQLM::new(
            Cow::Owned(codebooks),
            Cow::Owned(scales),
            Cow::Owned(codes),
            out_dim,
            in_group_dim,
        ))
    }
}

#[async_trait(?Send)]
impl<LinearType> FromStateDict for MLP<LinearType>
where
    LinearType: Module + FromStateDict,
{
    async fn from_state_dict(prefix: &str) -> anyhow::Result<Self> {
        Ok(MLP::new(MLPSubmodules {
            up_proj: LinearType::from_state_dict(&format!("{prefix}up_proj.")).await?,
            gate_proj: LinearType::from_state_dict(&format!("{prefix}gate_proj.")).await?,
            down_proj: LinearType::from_state_dict(&format!("{prefix}down_proj.")).await?,
        }))
    }
}

#[async_trait(?Send)]
impl<LinearType> FromStateDictConf<AttentionConfig> for Attention<LinearType>
where
    LinearType: Module + FromStateDict,
{
    async fn from_state_dict(prefix: &str, config: AttentionConfig) -> anyhow::Result<Self> {
        let weights = AttentionSubmodules {
            v_proj: CachedAttentionLinear::new(
                LinearType::from_state_dict(&format!("{prefix}v_proj.")).await?,
                None,
            ),
            q_proj: LinearType::from_state_dict(&format!("{prefix}q_proj.")).await?,
            k_proj: CachedAttentionLinear::new(
                LinearType::from_state_dict(&format!("{prefix}k_proj.")).await?,
                Some(config.get_emb_config()),
            ),
            o_proj: LinearType::from_state_dict(&format!("{prefix}o_proj.")).await?,
        };

        Ok(Attention::new(weights, config))
    }
}

#[async_trait(?Send)]
impl<LinearType> FromStateDictConf<LlamaConfig> for LlamaBlock<LinearType>
where
    LinearType: Module + FromStateDict,
{
    async fn from_state_dict(prefix: &str, config: LlamaConfig) -> anyhow::Result<Self> {
        let input_layernorm_prefix = format!("{prefix}input_layernorm.");
        let input_layernorm = LayerNorm::from_state_dict(&input_layernorm_prefix, config.norm_eps);
        let attention_prefix = format!("{prefix}self_attn.");
        let attention = Attention::from_state_dict(&attention_prefix, config.to_attention_config());
        let post_attention_layernorm_prefix = format!("{prefix}post_attention_layernorm.");
        let post_attention_layernorm =
            LayerNorm::from_state_dict(&post_attention_layernorm_prefix, config.norm_eps);
        let mlp_prefix = format!("{prefix}mlp.");
        let mlp = MLP::from_state_dict(&mlp_prefix);

        let (input_layernorm, attention, post_attention_layernorm, mlp) =
            join!(input_layernorm, attention, post_attention_layernorm, mlp,);

        let (input_layernorm, attention, post_attention_layernorm, mlp) = (
            input_layernorm?,
            attention?,
            post_attention_layernorm?,
            mlp?,
        );

        let submodules = LlamaBlockSubmodules {
            input_layernorm,
            attention,
            post_attention_layernorm,
            mlp,
        };

        Ok(LlamaBlock::new(submodules))
    }
}

pub async fn fetch_url(url: &str) -> ehttp::Result<ehttp::Response> {
    let (tx, mut rx) = mpsc::unbounded_channel();

    let request = ehttp::Request::get(url);
    ehttp::fetch(request, move |result: ehttp::Result<ehttp::Response>| {
        tx.send(result).unwrap();
    });

    rx.recv().await.unwrap()
}

pub async fn get_data(url: &str) -> anyhow::Result<Vec<u8>> {
    let response = fetch_url(url).await.unwrap();

    if !response.ok {
        return Err(anyhow!("{} ({})", response.status_text, response.status));
    }

    Ok(response.bytes)
}

const LOCAL_PREFIX: &str = "http://localhost:8000/aqlm-f32/";
const LOCAL_SUFFIX: &str = "";
const PROD_PREFIX: &str =
    "https://huggingface.co/galqiwi/llama-3.1-aqlm-pv-2x8-f32-int8-emb/resolve/main/";
const PROD_SUFFIX: &str = "?download=true";

fn get_url_by_name(filename: &str) -> String {
    const LOCAL: bool = false;

    let (prefix, suffix) = match LOCAL {
        true => (LOCAL_PREFIX, LOCAL_SUFFIX),
        false => (PROD_PREFIX, PROD_SUFFIX),
    };

    format!("{prefix}{filename}{suffix}")
}

pub async fn get_file_by_name(filename: &str) -> anyhow::Result<Vec<u8>> {
    for _retry_idx in 0..2 {
        let output = get_data(&get_url_by_name(filename)).await;

        if output.is_ok() {
            return output;
        }
    }

    let output = get_data(&get_url_by_name(filename)).await;

    output.map_err(|err| anyhow!("huggingface error when loading: {}", err))
}

fn tensor_view_to_owned_tensor(value: TensorView) -> OwnedTensor {
    let data = value.data().to_vec();
    assert_eq!(data.capacity(), data.len());

    let shape = value.shape().to_vec();
    assert_eq!(shape.capacity(), shape.len());

    let dtype = value.dtype();

    let dtype = match dtype {
        safetensors::Dtype::F32 => Dtype::F32,
        safetensors::Dtype::U8 => Dtype::U8,
        safetensors::Dtype::I8 => Dtype::I8,
        _ => unimplemented!(),
    };

    OwnedTensor { data, shape, dtype }
}

pub fn read_owned_tensor(data: &[u8]) -> anyhow::Result<OwnedTensor> {
    let mut tensors = SafeTensors::deserialize(data)
        .map_err(|err| anyhow!("failed to parse tensor: {}", err))?
        .tensors();
    assert_eq!(tensors.len(), 1);

    let (_, tensor) = tensors.pop().ok_or_else(|| anyhow!("no tensor found"))?;

    Ok(tensor_view_to_owned_tensor(tensor))
}

pub async fn get_tensor(path: &str) -> anyhow::Result<OwnedTensor> {
    read_owned_tensor(&get_file_by_name(&format!("{path}.safetensors")).await?)
}

pub async fn load_f32_data(path: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    Ok(get_f32_data(get_tensor(path).await?))
}

pub async fn load_u8_data(path: &str) -> anyhow::Result<(Vec<u8>, Vec<usize>)> {
    Ok(get_u8_data(get_tensor(path).await?))
}

pub async fn load_i8_data(path: &str) -> anyhow::Result<(Vec<i8>, Vec<usize>)> {
    Ok(get_i8_data(get_tensor(path).await?))
}
