use crate::Role::Assistant;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};
use std::str::from_utf8;

pub struct Llama3Tokenizer {
    inner: tiktoken_rs::CoreBPE,
    special_tokens_map: HashMap<String, usize>,
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct Message {
    pub content: String,
    pub role: Role,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Llama3Tokenizer {
    pub fn from_data(data: Vec<u8>) -> anyhow::Result<Self> {
        let mut special_tokens: Vec<String> = vec![
            "<|begin_of_text|>".to_string(),
            "<|end_of_text|>".to_string(),
            "<|reserved_special_token_0|>".to_string(),
            "<|reserved_special_token_1|>".to_string(),
            "<|reserved_special_token_2|>".to_string(),
            "<|reserved_special_token_3|>".to_string(),
            "<|start_header_id|>".to_string(),
            "<|end_header_id|>".to_string(),
            "<|reserved_special_token_4|>".to_string(),
            "<|eot_id|>".to_string(), // end of turn
        ];

        for i in 5..251 {
            special_tokens.push(format!("<|reserved_special_token_{}|>", i));
        }

        let data = from_utf8(&data)?;
        let mut mergeable_ranks: HashMap<Vec<u8>, usize> = HashMap::default();

        for line in data.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(parts.len(), 2);

            let base64_str = parts[0];
            let number: usize = match parts[1].parse() {
                Ok(num) => num,
                Err(_) => continue, // Skip lines where the number isn't valid
            };

            let decoded_bytes = match BASE64_STANDARD.decode(base64_str) {
                Ok(bytes) => bytes,
                Err(_) => continue, // Skip lines where the base64 decoding fails
            };

            mergeable_ranks.insert(decoded_bytes, number);
        }

        let special_tokens_map: HashMap<String, usize> = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, token)| (token, mergeable_ranks.len() + i))
            .collect();

        let inner = tiktoken_rs::CoreBPE::new(
            mergeable_ranks,
            special_tokens_map.clone(),
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )?;

        Ok(Self {
            inner,
            special_tokens_map,
        })
    }
}

impl Llama3Tokenizer {
    fn get_special_token_id(&self, token_name: &str) -> usize {
        *self.special_tokens_map.get(token_name).unwrap()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let output = self.inner._decode_native(tokens);
        String::from_utf8_lossy(&output).to_string()
    }

    fn decode_message(&self, tokens: &[usize]) -> Message {
        let role = (|| {
            let roles = vec![Role::User, Role::Assistant, Role::System];

            for role in roles {
                let role_header = self.encode_message_header(&Message {
                    role,
                    content: "".to_string(),
                });
                if tokens.starts_with(&role_header) {
                    return role;
                }
            }

            unreachable!("{:?} {:?}", tokens, self.decode(tokens));
        })();

        let header_end_idx = tokens
            .iter()
            .position(|&t| t == self.get_special_token_id("<|end_header_id|>"))
            .unwrap();

        assert_eq!(
            &tokens[header_end_idx..header_end_idx + 2],
            &vec![
                self.get_special_token_id("<|end_header_id|>"),
                self.encode_ordinary("\n\n")[0],
            ]
        );

        let mut tokens = &tokens[header_end_idx + 2..tokens.len()];
        if tokens.ends_with(&[self.get_special_token_id("<|eot_id|>")]) {
            tokens = &tokens[0..tokens.len() - 1];
        }

        let content = self.decode(tokens);

        Message { role, content }
    }

    pub fn decode_dialog(&self, tokens: &[usize]) -> Vec<Message> {
        assert_eq!(tokens[0], self.get_special_token_id("<|begin_of_text|>"));
        let tokens = &tokens[1..tokens.len()];

        let tokenized_messages: Vec<&[usize]> =
            split_delimited(tokens, &self.get_special_token_id("<|start_header_id|>"));

        tokenized_messages
            .iter()
            .map(|tokenized_message| self.decode_message(tokenized_message))
            .collect()
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        self.inner.encode_ordinary(text)
    }

    pub fn encode_dialog_prompt(&self, dialog: &[Message]) -> Vec<usize> {
        let mut output = Vec::new();

        output.push(self.get_special_token_id("<|begin_of_text|>"));

        for message in dialog {
            output.extend(self.encode_message(message));
        }

        output.extend(self.encode_message_header(&Message {
            role: Assistant,
            content: "".to_string(),
        }));

        output
    }

    pub fn is_eot(&self, token: usize) -> bool {
        token == self.get_special_token_id("<|eot_id|>")
    }

    pub fn encode_message(&self, msg: &Message) -> Vec<usize> {
        let mut output = self.encode_message_header(msg);

        output.extend(self.inner.encode_ordinary(msg.content.trim()));

        output.push(self.get_special_token_id("<|eot_id|>"));

        output
    }

    pub fn encode_message_header(&self, msg: &Message) -> Vec<usize> {
        let mut output = Vec::new();

        output.push(self.get_special_token_id("<|start_header_id|>"));

        output.extend(self.inner.encode_ordinary(match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }));

        output.push(self.get_special_token_id("<|end_header_id|>"));

        output.extend(self.inner.encode_ordinary("\n\n"));

        output
    }
}

pub fn split_delimited<'a, T>(input: &'a [T], delim: &T) -> Vec<&'a [T]>
where
    T: PartialEq<T>,
{
    let elems = input.iter().enumerate();
    let (k, mut r) = elems.fold((0, vec![]), |(i, mut r), (j, x)| {
        if x == delim && j > 0 {
            r.push(&input[i..j]);
            return (j, r);
        }
        (i, r)
    });
    if !input.is_empty() {
        r.push(&input[k..]);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_dialog() {
        let file =
            fs::read("/Users/galqiwi/vcs_projects/aqlm.rs/aqlm-f32/tokenizer.model").unwrap();

        let tokenizer = Llama3Tokenizer::from_data(file).unwrap();

        let mut dialog = vec![
            Message {
                content: "2+2=?".to_string(),
                role: Role::User,
            },
            Message {
                content: "2+2=4".to_string(),
                role: Role::Assistant,
            },
        ];

        let tokens = tokenizer.encode_dialog_prompt(&dialog);
        let dialog_decoded = tokenizer.decode_dialog(&tokens);

        dialog.push(Message {
            content: "".to_string(),
            role: Role::Assistant,
        });
        assert_eq!(dialog, dialog_decoded);
    }
}
