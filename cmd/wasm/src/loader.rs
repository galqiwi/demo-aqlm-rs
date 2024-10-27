use crate::api::LlamaAPI;
use crate::loader::StatusMessage::Cyanide;
use generator::Generator;
use nn::embedding::EmbeddingINT8;
use nn::layernorm::LayerNorm;
use nn::llama::{Llama, LlamaSubmodules};
use nn::llama_block::LlamaBlock;
use nn::llama_config::LLAMA_3_1_8B_CONFIG;
use state_dict::from_state_dict::{get_file_by_name, FromStateDict, FromStateDictConf};
use std::mem;
use std::option::Option;
use tokenizer::Llama3Tokenizer;
use tokio::sync::mpsc;
use wasm_bindgen::prelude::wasm_bindgen;
use web_sys::Worker;
use worker_engine::parallel_aqlm::{set_handles, ParallelAQLMLinear};
use worker_engine::parallel_int8::ParallelINT8Linear;
use worker_engine::registry_rpc_handle::RPCLinearRegistryHandle;

#[wasm_bindgen]
pub struct LlamaLoader {
    handles: Vec<RPCLinearRegistryHandle>,
    status_tx: mpsc::Sender<StatusMessage>,
    status_rx: Option<mpsc::Receiver<StatusMessage>>,
}

#[wasm_bindgen]
impl LlamaLoader {
    pub fn new(workers: Vec<Worker>) -> Self {
        let handles: Vec<_> = workers
            .into_iter()
            .map(RPCLinearRegistryHandle::new)
            .collect();

        let (status_tx, status_rx) = mpsc::channel(1);

        LlamaLoader {
            handles,
            status_tx,
            status_rx: Some(status_rx),
        }
    }
}

#[wasm_bindgen]
impl LlamaLoader {
    pub fn get_worker_response_senders(&self) -> Vec<WorkerResponseSender> {
        self.handles
            .iter()
            .map(|handle| WorkerResponseSender {
                response_tx: handle.get_sender(),
            })
            .collect()
    }

    pub fn get_download_status_sender(&mut self) -> DownloadStatusSender {
        let status_rx = self.status_rx.take().unwrap();

        DownloadStatusSender { status_rx }
    }

    pub async fn into_llama_api(mut self) -> LlamaAPI {
        let llama_api = self.do_into_llama_api().await;

        // let llama_api = llama_api.unwrap();

        match llama_api {
            Ok(api) => api,
            Err(err) => {
                self.send_str_status(err.to_string()).await;

                let (status_tx, mut status_rx) = mpsc::channel(1);

                // TODO: return something
                status_rx.recv().await.unwrap();
                status_tx.send(1).await.unwrap();

                panic!("{}", err);
            }
        }
    }

    async fn send_loading_status(&mut self, layer_idx: usize, n_layers: usize) {
        self.send_str_status(format!("Loading llama: {}/{}", layer_idx + 1, n_layers))
            .await;
    }

    async fn do_into_llama_api(&mut self) -> anyhow::Result<LlamaAPI> {
        set_handles(mem::take(&mut self.handles)).await;

        let generator = {
            let config = &LLAMA_3_1_8B_CONFIG;

            let mut blocks = Vec::new();

            let n_layers = config.n_layers;

            for layer_idx in 0..n_layers / 2 {
                self.send_loading_status(layer_idx, n_layers + 2).await;

                blocks.push(
                    LlamaBlock::<ParallelAQLMLinear>::from_state_dict(
                        &format!("model.layers.{i}.", i = layer_idx),
                        config.clone(),
                    )
                    .await?,
                );
            }

            let embed_tokens = EmbeddingINT8::from_state_dict("model.embed_tokens.").await?;
            self.send_loading_status(n_layers / 2, n_layers + 2).await;

            let norm = LayerNorm::from_state_dict("model.norm.", config.norm_eps).await?;

            let lm_head = ParallelINT8Linear::from_state_dict("lm_head.").await?;
            self.send_loading_status(n_layers / 2 + 1, n_layers + 2)
                .await;

            for layer_idx in n_layers / 2..n_layers {
                self.send_loading_status(layer_idx + 2, n_layers + 2).await;

                blocks.push(
                    LlamaBlock::<ParallelAQLMLinear>::from_state_dict(
                        &format!("model.layers.{i}.", i = layer_idx),
                        config.clone(),
                    )
                    .await?,
                );
            }

            let submodules = LlamaSubmodules {
                embed_tokens,
                blocks,
                norm,
                lm_head,
            };

            let llama = Llama::new(submodules);
            Generator::new(llama)
        };

        let tokenizer = {
            let tokenizer_data = get_file_by_name("tokenizer.model").await?;

            Llama3Tokenizer::from_data(tokenizer_data)?
        };

        self.send_status(Cyanide).await;

        Ok(LlamaAPI {
            generator,
            tokenizer,
        })
    }

    async fn send_str_status(&mut self, status: String) {
        self.send_status(StatusMessage::Message(status.to_string()))
            .await
    }

    async fn send_status(&mut self, status: StatusMessage) {
        self.status_tx.send(status).await.unwrap()
    }
}

#[wasm_bindgen]
pub struct DownloadStatusSender {
    status_rx: mpsc::Receiver<StatusMessage>,
}

#[wasm_bindgen]
impl DownloadStatusSender {
    pub async fn get_status(&mut self) -> String {
        let message = self.status_rx.recv().await.unwrap_or(Cyanide);

        match message {
            StatusMessage::Message(output) => output,
            Cyanide => "cyanide".to_string(),
        }
    }
}

#[wasm_bindgen]
pub struct WorkerResponseSender {
    response_tx: mpsc::Sender<Vec<u8>>,
}

enum StatusMessage {
    Message(String),
    Cyanide,
}

#[wasm_bindgen]
impl WorkerResponseSender {
    pub async fn register_response(&mut self, data: Vec<u8>) {
        self.response_tx.send(data).await.unwrap()
    }
}
