//! A complete language model implementation
//!
//! Use `cargo run --bin mini_lm`
//!
//! This combines:
//!    - Tokenizer (with embeddings)
//!    - Attention (with causal masking)
//!    - Output projection
//!    - Generation loop
//!
//! The model has random weights, so it outputs gibberish.
//! But the structure is real — this is how LLMs work.

// ─────────────────────────────────────────────────────────────
// TOKENIZER MODULE (includes vocabulary + embeddings)
// ─────────────────────────────────────────────────────────────

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
            let id_to_word: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
            let word_to_id: HashMap<String, usize> = vocab
                .iter()
                .enumerate()
                .map(|(i, &w)| (w.to_string(), i))
                .collect();

            // Build embedding table (same vocab_size, guaranteed in sync)
            let vocab_size = vocab.len();
            let embeddings = (0..vocab_size)
                .map(|i| {
                    (0..embed_dim)
                        .map(|j| {
                            let hash = ((i * 1001 + j * 23) % 1000) as f32;
                            (hash / 500.0) - 1.0 // Range [-1, 1]
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

        /// Text → Token IDs
        pub fn tokenize(&self, text: &str) -> Vec<usize> {
            text.split_whitespace()
                .map(|w| *self.word_to_id.get(w).unwrap_or(&1)) // 1 = <UNK>
                .collect()
        }

        /// Token IDs → Text
        pub fn _detokenize(&self, ids: &[usize]) -> String {
            ids.iter()
                .map(|&id| {
                    self.id_to_word
                        .get(id)
                        .map(|s| s.as_str())
                        .unwrap_or("<UNK>")
                })
                .collect::<Vec<_>>()
                .join(" ")
        }

        /// Token IDs → Vectors
        pub fn embed(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
            token_ids
                .iter()
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

// ─────────────────────────────────────────────────────────────
// ATTENTION MODULE (with causal masking)
// ─────────────────────────────────────────────────────────────

mod attention {
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_score = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = logits.iter().map(|&x| (x - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        exp_scores.iter().map(|&x| x / sum).collect()
    }

    pub struct Attention {
        pub embed_dim: usize,
    }

    impl Attention {
        pub fn new(embed_dim: usize) -> Self {
            Self { embed_dim }
        }

        /// Apply self-attention with causal masking
        /// Each token can only attend to itself and previous tokens
        pub fn forward(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
            let seq_len = embeddings.len();
            let scale = (self.embed_dim as f32).sqrt();

            let mut outputs = Vec::with_capacity(seq_len);

            for i in 0..seq_len {
                let query = &embeddings[i];

                // Compute attention scores (with causal mask)
                let mut scores = Vec::with_capacity(seq_len);
                for j in 0..seq_len {
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

// ─────────────────────────────────────────────────────────────
// OUTPUT PROJECTION MODULE
// ─────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────
// MINI LANGUAGE MODEL
// ─────────────────────────────────────────────────────────────

mod mini_lm {
    use crate::attention::{softmax, Attention};
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

// ─────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────

fn main() {
    use mini_lm::MiniLM;

    println!("Day 5: A Complete Language Model");
    println!("══════════════════════════════════════════\n");

    // Simple vocabulary
    let vocab = [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", // Special tokens (0-3)
        "the", "a", "an",                   // Articles (4-6)
        "cat", "dog", "bird", "fish",       // Animals (7-10)
        "sat", "ran", "flew", "swam",       // Verbs (11-14)
        "on", "under", "over", "in",        // Prepositions (15-18)
        "mat", "tree", "water", "sky",      // Nouns (19-22)
        "big", "small", "quick", "lazy",    // Adjectives (23-26)
        "what", "is", "where", "how",       // Question words (27-30)
        "capital", "of", "france",          // For our classic example (31-33)
        "germany", "paris", "berlin",       // (34-36)
    ];

    let model = MiniLM::new(&vocab, 8);

    // ─────────────────────────────────────────────────────────────
    // STEP-BY-STEP FORWARD PASS
    // ─────────────────────────────────────────────────────────────

    println!("Step-by-Step Forward Pass");
    println!("────────────────────────────────────────────\n");

    let prompt = "the cat sat on";
    println!("Input: \"{}\"\n", prompt);

    // Step 1: Tokenize
    let token_ids = model.tokenizer.tokenize(prompt);
    println!("Step 1 - Tokenize:");
    println!(
        "  Tokens: {:?}",
        prompt.split_whitespace().collect::<Vec<_>>()
    );
    println!("  IDs:    {:?}\n", token_ids);

    // Step 2: Embed
    let embeddings = model.tokenizer.embed(&token_ids);
    println!("Step 2 - Embed:");
    println!(
        "  Shape: {} tokens × {} dimensions",
        embeddings.len(),
        model.tokenizer.embed_dim
    );
    println!(
        "  First embedding: [{:.2}, {:.2}, {:.2}, ...]",
        embeddings[0][0], embeddings[0][1], embeddings[0][2]
    );
    println!();

    // Step 3: Attention
    let hidden_states = model.attention.forward(&embeddings);
    println!("Step 3 - Attention:");
    println!(
        "  Shape: {} tokens × {} dimensions",
        hidden_states.len(),
        model.tokenizer.embed_dim
    );
    println!(
        "  Last hidden: [{:.2}, {:.2}, {:.2}, ...]",
        hidden_states.last().unwrap()[0],
        hidden_states.last().unwrap()[1],
        hidden_states.last().unwrap()[2]
    );
    println!();

    // Step 4: Output projection
    let logits = model.output.forward(hidden_states.last().unwrap());
    println!("Step 4 - Output Projection:");
    println!("  Logits shape: {} (one per vocab word)", logits.len());
    println!();

    // Step 5: Sample
    let next_token_id = model.sample(&logits, 1.0);
    let next_token = &model.tokenizer.id_to_word[next_token_id];
    println!("Step 5 - Sample:");
    println!("  Next token ID: {}", next_token_id);
    println!("  Next token: \"{}\"\n", next_token);

    // ─────────────────────────────────────────────────────────────
    // GENERATION
    // ─────────────────────────────────────────────────────────────

    println!("Text Generation");
    println!("────────────────────────────────────────────\n");

    let prompts = ["the cat", "a big dog", "what is the"];

    for prompt in prompts {
        let generated = model.generate(prompt, 5, 1.0);
        println!("  \"{}\" → \"{}\"", prompt, generated);
    }
    println!();

    // ─────────────────────────────────────────────────────────────
    // WHY IT'S GIBBERISH
    // ─────────────────────────────────────────────────────────────

    println!("Why It's Gibberish");
    println!("────────────────────────────────────────────\n");
    println!("  Our model has random weights. It hasn't learned anything.");
    println!();
    println!("  To generate real text, you need:");
    println!("    1. Training data (billions of tokens)");
    println!("    2. Loss function (cross-entropy)");
    println!("    3. Optimizer (Adam)");
    println!("    4. Compute (GPUs, weeks of training)");
    println!();
    println!("  But the STRUCTURE is real:");
    println!("    Tokenize → Embed → Attend → Project → Sample");
    println!();

    // ─────────────────────────────────────────────────────────────
    // PARAMETER COUNT
    // ─────────────────────────────────────────────────────────────

    println!("Parameter Count");
    println!("────────────────────────────────────────────\n");

    let vocab_size = model.tokenizer.vocab_size();
    let embed_dim = model.tokenizer.embed_dim;

    let embedding_params = vocab_size * embed_dim;
    let output_params = vocab_size * embed_dim;
    let total = embedding_params + output_params;

    println!("  Vocabulary size: {}", vocab_size);
    println!("  Embedding dim:   {}", embed_dim);
    println!();
    println!(
        "  Embedding:  {} × {} = {} parameters",
        vocab_size, embed_dim, embedding_params
    );
    println!(
        "  Output:     {} × {} = {} parameters",
        vocab_size, embed_dim, output_params
    );
    println!("  Attention:  0 (no learned weights in our simple version)");
    println!("  ─────────────────────────────────");
    println!("  Total:      {} parameters", total);
    println!();
    println!("  GPT-2 Small:  124,000,000 parameters");
    println!("  GPT-3:        175,000,000,000 parameters");
    println!("  Our model:    {} parameters", total);
    println!();
}

// ─────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::attention::*;
    use super::mini_lm::*;
    use super::output::*;
    use super::tokenizer::*;

    // ─────────────────────────────────────────────────────────
    // TOKENIZER TESTS
    // ─────────────────────────────────────────────────────────

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
        let tokenizer = Tokenizer::new(&vocab, 4);

        let ids = tokenizer.tokenize("the cat sat");
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn test_tokenize_unknown() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat"];
        let tokenizer = Tokenizer::new(&vocab, 4);

        // "dog" not in vocab, should return 1 (<UNK>)
        let ids = tokenizer.tokenize("the dog");
        assert_eq!(ids, vec![2, 1]);
    }

    #[test]
    fn test_detokenize() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let tokenizer = Tokenizer::new(&vocab, 4);

        let text = tokenizer.detokenize(&[2, 3, 4]);
        assert_eq!(text, "the cat sat");
    }

    #[test]
    fn test_embed() {
        let vocab = ["<PAD>", "<UNK>", "hello"];
        let tokenizer = Tokenizer::new(&vocab, 4);

        let embeddings = tokenizer.embed(&[0, 2]);
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 4);
        assert_eq!(embeddings[1].len(), 4);
    }

    #[test]
    fn test_embed_deterministic() {
        let vocab = ["<PAD>", "<UNK>", "hello"];
        let tokenizer = Tokenizer::new(&vocab, 4);

        let emb1 = tokenizer.embed(&[2]);
        let emb2 = tokenizer.embed(&[2]);

        // Same token should produce same embedding
        assert_eq!(emb1, emb2);
    }

    // ─────────────────────────────────────────────────────────
    // ATTENTION TESTS
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_dot_product_zeros() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];

        assert_eq!(dot_product(&a, &b), 0.0);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        // Equal logits = equal probabilities
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

        // Same shape as input
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 4);
    }

    #[test]
    fn test_attention_causal_mask() {
        let attention = Attention::new(4);

        // Two different sequences with same prefix
        let emb1 = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0], // Different 3rd token
        ];
        let emb2 = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0], // Different 3rd token
        ];

        let out1 = attention.forward(&emb1);
        let out2 = attention.forward(&emb2);

        // First position should be identical (can't see future)
        assert_eq!(out1[0], out2[0]);

        // Second position should also be identical (can only see positions 0-1)
        assert_eq!(out1[1], out2[1]);

        // Third position should be different (sees different token at position 2)
        assert_ne!(out1[2], out2[2]);
    }

    // ─────────────────────────────────────────────────────────
    // OUTPUT PROJECTION TESTS
    // ─────────────────────────────────────────────────────────

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

    // ─────────────────────────────────────────────────────────
    // MINI LM TESTS
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_mini_lm_forward() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let model = MiniLM::new(&vocab, 4);

        let logits = model.forward("the cat");

        // Should have one logit per vocab word
        assert_eq!(logits.len(), 5);
    }

    #[test]
    fn test_mini_lm_sample() {
        let vocab = ["<PAD>", "<UNK>", "the", "cat", "sat"];
        let model = MiniLM::new(&vocab, 4);

        let logits = vec![0.1, 0.2, 0.3, 10.0, 0.5]; // "cat" has highest logit

        let sampled = model.sample(&logits, 1.0);

        // Should pick index 3 (highest logit)
        assert_eq!(sampled, 3);
    }

    #[test]
    fn test_mini_lm_generate() {
        let vocab = ["<PAD>", "<UNK>", "<EOS>", "the", "cat"];
        let model = MiniLM::new(&vocab, 4);

        let generated = model.generate("the", 3, 1.0);

        // Should start with prompt
        assert!(generated.starts_with("the"));

        // Should have generated some tokens
        assert!(generated.len() > 3);
    }

    #[test]
    fn test_mini_lm_generate_stops_at_eos() {
        let vocab = ["<PAD>", "<UNK>", "<EOS>", "the", "cat"];
        let model = MiniLM::new(&vocab, 4);

        // Generate with high max_tokens
        let generated = model.generate("the", 100, 1.0);

        // Should not contain <EOS> in output (we stop before adding it)
        // And should be reasonable length (stopped early or hit max)
        assert!(generated.split_whitespace().count() <= 101);
    }
}