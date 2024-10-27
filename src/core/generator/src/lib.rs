// use log::info;
use nn::functional::softmax_one_row;
use nn::linear::Module;
use nn::llama::Llama;
use rand::random;
// use web_time::Instant;

pub struct Generator<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    model: Llama<BlockLinearType, HeadLinearType>,
    tokens: Vec<usize>,
}

impl<BlockLinearType, HeadLinearType> Generator<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    pub fn new(model: Llama<BlockLinearType, HeadLinearType>) -> Self {
        Self {
            model,
            tokens: Vec::new(),
        }
    }
}

impl<BlockLinearType, HeadLinearType> Generator<BlockLinearType, HeadLinearType>
where
    BlockLinearType: Module,
    HeadLinearType: Module,
{
    pub async fn next_token(&mut self) -> usize {
        // let begin = Instant::now();

        let logits = self.model.forward(*self.tokens.last().unwrap()).await;

        // let model_time = begin.elapsed().as_secs_f64();

        // let new_token = argmax(&logits);
        let new_token = Self::get_token(logits);
        self.tokens.push(new_token);

        // let sample_time = begin.elapsed().as_secs_f64() - model_time;

        // info!("Model time: {}, Sample time: {}", model_time, sample_time);

        new_token
    }

    fn top_p(mut logits: Vec<f32>) -> Vec<f32> {
        let probs = softmax_one_row(logits.clone());

        let mut indexed_probs: Vec<_> = probs.iter().enumerate().collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        let indexed_cumsums: Vec<_> = indexed_probs
            .iter()
            .scan(0f32, |cumsum, (index, prob)| {
                *cumsum += *prob;
                Some((*index, *cumsum))
            })
            .collect();

        for (index, cumsum) in indexed_cumsums[1..indexed_cumsums.len()].iter().rev() {
            if *cumsum > 0.9 {
                logits[*index] = f32::NEG_INFINITY;
            } else {
                break;
            }
        }

        logits
    }

    fn get_token(mut logits: Vec<f32>) -> usize {
        logits.iter_mut().for_each(|x| *x /= 0.6);

        let logits = Self::top_p(logits);

        let probs = softmax_one_row(logits);

        let x: f32 = random();
        let mut cumsum = 0f32;

        let n_tokens = probs.len();

        for (token_idx, prob) in probs.into_iter().enumerate() {
            cumsum += prob;
            if cumsum > x {
                return token_idx;
            }
        }

        n_tokens - 1
    }

    pub async fn add_tokens(&mut self, tokens: &[usize]) {
        for token in tokens {
            if let Some(token) = self.tokens.last() {
                self.model.forward(*token).await;
            }
            self.tokens.push(*token);
        }
    }

    pub async fn set_tokens(&mut self, tokens: &[usize]) {
        assert!(tokens.starts_with(&self.tokens));
        self.add_tokens(&tokens[self.tokens.len()..tokens.len()])
            .await
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.model.clear_cache();
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }
}
