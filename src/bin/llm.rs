//! A simple LLM that generates text token by token
//!
//! ## 1. Model Setup (once)
//! - Define vocabulary (word <-> ID mappings)
//! - Create embedding table (one vector per word, random init)
//! - Create attention (no learned weights in our model)
//! - Create output projection weights (random init)
//!
//! ## 2. Model Inference (loops for generation )
//!
//! Repeat these steps for each generated token:
//! - Tokenize (text -> IDs)
//! - Embed (IDs -> vector embeddings)
//! - Attention (vectors -> contextualized vectors)
//! - Last Hidden ( take the last vector only)
//! - Project (hidden -> logits)
//! - Sample (logits -> next token ID)
//! - Detokenize (IDs -> word)
//! - Append to text (for next loop)
//!
//!  
//! The model outputs gibberish as weights are random (no training).
//! But the structure is real - shows how a simple LLM works.
//!
//! ## 3. Glossary:
//! - embed_dim: Size of each vector (4 or 8). Every word becomes this many numbers.
//! - Embedding: Table lookup: token ID → vector
//! - Attention: Mix information from other tokens into each position
//! - Hidden state: The vector at a position after attention (contextualized)
//! - Last hidden: Hidden state at final position — has seen all input
//! - Project: Matrix multiply: convert 4-number hidden → 6-number logits
//! - Logits: Raw scores, one per vocab word. Higher = more likely.
//! - Sample: Pick next token from logits (greedy or probabilistic)
//!
//!
//!

mod tokenizer {
    use std::collections::HashMap;

    pub struct Tokenizer {
        pub word_to_id: HashMap<String, usize>,
        pub id_to_word: Vec<String>,
        pub embeddings: Vec<Vec<f32>>,
        pub embed_dim: usize,
    }

    impl Tokenizer {
        pub fn new(vocab: &[&str], embed_dim: usize) -> Self {
            // Build vocab mappings
            let id_to_word = vocab.iter().map(|&s| s.to_string()).collect();
            let word_to_id = vocab
                .iter()
                .enumerate()
                .map(|(i, &w)| (w.to_string(), i))
                .collect();

            // Build embedding table with random values
            let embeddings = (0..vocab.len())
                .map(|i| {
                    (0..embed_dim)
                        .map(|j| {
                            let hash = ((i * 1001 + j * 23) % 1000) as f32;
                            (hash / 500.0) - 1.0
                        })
                        .collect()
                })
                .collect();
            Self {
                word_to_id,
                id_to_word,
                embeddings,
                embed_dim,
            }
        }

        // Text to token IDs
        pub fn tokenize(&self, text: &str) -> Vec<usize> {
            text.split_whitespace()
                .map(|word| self.word_to_id.get(word).cloned().unwrap_or(1))
                .collect()
        }

        // Token IDs to text
        pub fn _detokenize(&self, ids: &[usize]) -> String {
            ids.iter()
                .map(|&id| {
                    self.id_to_word
                        .get(id)
                        .map(|s| s.as_str())
                        .unwrap_or("<UNK>")
                })
                .collect::<Vec<&str>>()
                .join(" ")
        }

        // Token Ids to Vectors
        pub fn embed(&self, ids: &[usize]) -> Vec<Vec<f32>> {
            ids.iter()
                .map(|&id| {
                    self.embeddings
                        .get(id)
                        .cloned()
                        .unwrap_or(vec![0.0; self.embed_dim])
                })
                .collect()
        }

        pub fn vocab_size(&self) -> usize {
            self.id_to_word.len()
        }
    }
}

mod attention {
    pub fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
        vec1.iter().zip(vec2.iter()).map(|(v1, v2)| v1 * v2).sum()
    }

    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_score = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = logits.iter().map(|&x| (x - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter().map(|&x| x / sum).collect::<Vec<f32>>()
    }

    pub struct Attention {
        pub embed_dim: usize,
    }

    impl Attention {
        pub fn new(embed_dim: usize) -> Self {
            Self { embed_dim }
        }

        /// Apply self-attention with causal masking
        /// Each token can only attend to previous tokens
        pub fn forward(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
            let seq_len = embeddings.len();
            let scale = (self.embed_dim as f32).sqrt();

            let mut outputs = Vec::with_capacity(seq_len);

            for i in 0..seq_len {
                let query = &embeddings[i];

                // Compute attention scores (with causal mask)
                let mut scores = Vec::with_capacity(seq_len);
                for (j, _) in embeddings.iter().enumerate() {
                    if j <= i {
                        // Can attend to current and past positions
                        let key = &embeddings[j];
                        let score = dot_product(query, key) / scale;
                        scores.push(score);
                    } else {
                        // Mask future positions
                        scores.push(f32::NEG_INFINITY);
                    }
                }

                let attn_weights = softmax(&scores);

                // Weighted sum of values
                let mut output = vec![0.0; self.embed_dim];
                for (j, weight) in attn_weights.iter().enumerate() {
                    if *weight > 0.0 {
                        let value = &embeddings[j];
                        for (k, v) in value.iter().enumerate() {
                            output[k] += weight * v;
                        }
                    }
                }
                outputs.push(output);
            }
            outputs
        }
    }
}

mod output {
    use crate::attention::dot_product;

    pub struct OutputProjection {
        pub weights: Vec<Vec<f32>>, // Shape: [vocab_size, embed_dim]
    }

    impl OutputProjection {
        pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
            // Pseudo-random weights
            let weights = (0..vocab_size)
                .map(|i| {
                    (0..embed_dim)
                        .map(|j| {
                            let hash = ((i * 7919 + j * 6271) % 1000) as f32;
                            (hash / 500.0) - 1.0
                        })
                        .collect()
                })
                .collect();
            Self { weights }
        }

        /// Convert hidden state to logits over vocabulary
        pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
            self.weights
                .iter()
                .map(|w| dot_product(hidden, w))
                .collect()
        }
    }
}

mod mini_lm {
    use crate::attention::{Attention, softmax};
    use crate::output::OutputProjection;
    use crate::tokenizer::Tokenizer;

    pub struct MiniLM {
        pub tokenizer: Tokenizer,
        pub attention: Attention,
        pub output: OutputProjection,
    }

    impl MiniLM {
        pub fn new(vocab: &[&str], embed_dim: usize) -> Self {
            let tokenizer = Tokenizer::new(vocab, embed_dim);
            let vocab_size = tokenizer.vocab_size();
            let attention = Attention::new(embed_dim);
            let output = OutputProjection::new(vocab_size, embed_dim);

            Self {
                tokenizer,
                attention,
                output,
            }
        }

        /// Forward pass: text → logits for next token
        pub fn forward(&self, text: &str) -> Vec<f32> {
            // Step 1: Tokenize
            let token_ids = self.tokenizer.tokenize(text);

            // Step 2: Embed
            let embeddings = self.tokenizer.embed(&token_ids);

            // Step 3: Attention
            let hidden_states = self.attention.forward(&embeddings);

            // Step 4: Get last hidden state and project to vocabulary
            let last_hidden = hidden_states.last().unwrap();
            self.output.forward(last_hidden)
        }

        /// Sample next token from logits
        pub fn sample(&self, logits: &[f32], temperature: f32) -> usize {
            // Apply temperature
            let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
            let probs = softmax(&scaled);

            // Pick highest probability (greedy)
            probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }

        /// Generate text
        pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> String {
            let mut text = prompt.to_string();

            for _ in 0..max_tokens {
                let logits = self.forward(&text);
                let next_token_id = self.sample(&logits, temperature);
                let next_token = &self.tokenizer.id_to_word[next_token_id];

                // Stop at end token
                if next_token == "<EOS>" {
                    break;
                }

                text.push(' ');
                text.push_str(next_token);
            }

            text
        }
    }
}

fn main() {
    println!("Mini LLM Model");

    use mini_lm::MiniLM;

    // Model Setup
    let vocab = ["<PAD>", "<UNK>", "<EOS>", "the", "cat", "sat", "on", "mat"];
    let model = MiniLM::new(&vocab, 8);

    println!("Model created with:");
    println!(
        "vocab_size={} and embed_dim={}",
        vocab.len(),
        model.tokenizer.embed_dim
    );
    println!("  - Embedding table: {} words × {} dims", vocab.len(), 8);
    println!("  - Attention: embed_dim = 8 (no learned weights)");
    println!(
        "  - Output projection: {} words × {} dims\n",
        vocab.len(),
        8
    );

    // Model inference

    let prompt = "the cat";

    // Step 1: Tokenize
    let token_ids = model.tokenizer.tokenize(prompt);
    println!("Input prompt: `{}`. Token IDs: {:?}", prompt, token_ids);

    // Step 2: Embed
    let embeddings = model.tokenizer.embed(&token_ids);

    // Step 3: Attention
    let hidden_states = model.attention.forward(&embeddings);

    // Step 4: Last Hidden
    let last_hidden = hidden_states.last().unwrap();

    // Step 5: Output projection
    let logits = model.output.forward(last_hidden);

    // Step 6: Sample
    let next_token_id = model.sample(&logits, 1.0);
    let next_token = &model.tokenizer.id_to_word[next_token_id];

    // Step 7: Output
    println!("Step 7 - APPEND:");
    println!(
        "  \"{}\" + \"{}\" = \"{} {}\"\n",
        prompt, next_token, prompt, next_token
    );
    // Generate multiple tokens
    let prompts = ["the cat", "cat sat", "the mat"];
    for p in prompts {
        let generated = model.generate(p, 5, 1.0);
        println!("Prompt: `{}` => Generated: `{}`", p, generated);
    }
}

//----------------
// Tests
//----------------

#[cfg(test)]
mod tests {

    use crate::attention::*;
    use crate::mini_lm::*;
    use crate::output::*;
    use crate::tokenizer::*;

    //Tokenizer tests
    #[test]
    fn test_tokenizer_new() {
        let vocab = ["<PAD>", "<UNK>", "hello", "world"];
        let tokenizer = Tokenizer::new(&vocab, 4);

        // Test vocab mappings
        assert_eq!(tokenizer.id_to_word[0], "<PAD>");
        assert_eq!(tokenizer.id_to_word[2], "hello");
        assert_eq!(tokenizer.word_to_id["world"], 3);
        assert_eq!(tokenizer.vocab_size(), 4);
        assert_eq!(tokenizer.embed_dim, 4);
    }

    #[test]
    fn test_tokenize() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let tokenizer = Tokenizer::new(&vocab, 5);
        let ids = tokenizer.tokenize("the cat sat");
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn test_tokenize_unknown() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let tokenizer = Tokenizer::new(&vocab, 5);
        let ids = tokenizer.tokenize("cat dog");
        assert_eq!(ids, vec![3, 1]);
    }

    #[test]
    fn test_detokenize() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let tokenizer = Tokenizer::new(&vocab, 5);
        let text = tokenizer._detokenize(&[2, 3, 4]);
        assert_eq!(text, "the cat sat");
    }

    #[test]
    fn test_detokenize_unknown() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let tokenizer = Tokenizer::new(&vocab, 5);
        let text = tokenizer._detokenize(&[3, 10]);
        assert_eq!(text, "cat <UNK>");
    }

    // Embed tests

    #[test]
    fn test_embed() {
        let vocab = ["<PAD>", "<UNK>", "hello", "world"];
        let tokenizer = Tokenizer::new(&vocab, 4);
        let embedding = tokenizer.embed(&[2, 0]);

        assert_eq!(embedding.len(), 2);
        assert_eq!(embedding[0].len(), 4);
        assert_eq!(embedding[1].len(), 4);
    }

    // Attention tests
    #[test]
    fn test_dot_product() {
        let vec1 = [1.0, 2.0, 3.0];
        let vec2 = [4.0, 5.0, 6.0];
        let result = dot_product(&vec1, &vec2);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_product_zeros() {
        let vec1 = [0.0, 0.0, 0.0];
        let vec2 = [4.0, 5.0, 6.0];
        let result = dot_product(&vec1, &vec2);
        assert_eq!(result, 0.0);
    }
    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Probabilities should sum to 1
        let sum = probs.iter().sum::<f32>();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit = Higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
    #[test]
    fn test_softmax_equal_logits() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = softmax(&logits);
        // A;; probabilities should be equal
        assert!((probs[0] - probs[1]).abs() < 1e-6);
        assert!((probs[1] - probs[2]).abs() < 1e-6);
    }

    #[test]
    fn test_attention_output_shape() {
        let attention = Attention::new(4);
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let output = attention.forward(&embeddings);
        assert_eq!(output.len(), embeddings.len());
        assert_eq!(output[0].len(), 4);
        for i in 0..embeddings.len() {
            assert_eq!(output[i].len(), embeddings[i].len());
        }
    }
    #[test]
    fn test_output_causal_mask() {
        let attention = Attention::new(4);
        let emb1 = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0], // Different third token
        ];
        let emb2 = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0], // Different third token
        ];

        let out1 = attention.forward(&emb1);
        let out2 = attention.forward(&emb2);

        // First position should be identitical - can't see the future
        assert_eq!(out1[0], out2[0]);
        // Second position should be identical - can't see the future
        assert_eq!(out1[1], out2[1]);
        // Third position should differ
        assert_ne!(out1[2], out2[2]);
    }
    // Output projection tests
    #[test]
    fn test_output_projection_shape() {
        let output = OutputProjection::new(10, 4);
        let hidden = vec![1.0, 0.0, 0.0, 0.0];
        let logits = output.forward(&hidden);

        // One logit per vocab word
        assert_eq!(logits.len(), 10);
    }

    #[test]
    fn test_output_projection_deterministic() {
        let output = OutputProjection::new(10, 4);
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let logits1 = output.forward(&hidden);
        let logits2 = output.forward(&hidden);

        assert_eq!(logits1, logits2);
    }

    // MiniLM tests
    #[test]
    fn test_minilm_forward() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let model = MiniLM::new(&vocab, 4);

        let logits = model.forward("the cat");

        // Logits length = vocab size
        assert_eq!(logits.len(), vocab.len());
    }

    #[test]
    fn test_mini_lm_sample() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let model = MiniLM::new(&vocab, 4);

        let generated = model.generate("the", 3, 1.0);

        // should start with prompt
        assert!(generated.starts_with("the"));

        // Should have generated some tokens
        assert!(generated.len() > 3);
    }

    #[test]
    fn test_mini_lm_generate_stops_at_eos() {
        let vocab = ["<PAD>", "<UNK>", "<EOS>", "the", "cat"];
        let model = MiniLM::new(&vocab, 4);

        // Generate with high max tokens
        let generated = model.generate("the", 100, 1.0);

        // Should not contain <EOS> in output
        // And should be reasonable length
        assert!(generated.split_whitespace().count() <= 101);
    }
}
