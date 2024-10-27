use generator::Generator;
use log::info;
use tokenizer::{Llama3Tokenizer, Message};
use wasm_bindgen::prelude::wasm_bindgen;
use web_time::Instant;
use worker_engine::parallel_aqlm::ParallelAQLMLinear;
use worker_engine::parallel_int8::ParallelINT8Linear;

#[wasm_bindgen]
pub struct LlamaAPI {
    pub(crate) generator: Generator<ParallelAQLMLinear, ParallelINT8Linear>,
    pub(crate) tokenizer: Llama3Tokenizer,
}

#[wasm_bindgen]
impl LlamaAPI {
    pub async fn set_prefix(&mut self, messages: Vec<String>) {
        let messages: Vec<Message> = messages
            .iter()
            .map(|message| serde_json::from_str(message).unwrap())
            .collect();

        self.generator
            .set_tokens(&self.tokenizer.encode_dialog_prompt(&messages))
            .await;
    }

    pub async fn next(&mut self) -> Vec<String> {
        let begin = Instant::now();
        self.generator.next_token().await;

        let output = self.tokenizer.decode_dialog(self.generator.tokens());

        info!("{}", self.tokenizer.decode(self.generator.tokens()));

        let output: Vec<String> = output
            .iter()
            .map(|message| serde_json::to_string(message).unwrap())
            .collect();

        info!("Seconds per token: {}", begin.elapsed().as_secs_f64());

        output
    }

    pub fn is_finished(&self) -> bool {
        let tokens = self.generator.tokens();
        if tokens.is_empty() {
            return false;
        }
        let last_token = tokens[tokens.len() - 1];

        self.tokenizer.is_eot(last_token)
    }

    pub fn clear(&mut self) {
        self.generator.clear();
    }
}
