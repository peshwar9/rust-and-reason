//!
//! Attention - how tokens look at each other
//! Use cargo run --bin attention
//!
//! This shows:
//!     - Why attention is useful
//!     - How queries, keys and values work
//!     - Scaled dot product attention from scratch

use std::collections::HashMap;

// Compute dot product of two vectors
fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x * y).sum()
}

// Softmax function to convert a vector of numbers into a probability distribution
fn softmax(logits: &[f32]) -> Vec<f32> {
    //subtract max for numerical stability
    let max_score = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    // Use exponential function to make all values positive and amplify differences (big numbers become bigger)
    let exp_scores: Vec<f32> = logits.iter().map(|&x| (x - max_score).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();
    exp_scores.iter().map(|&x| x / sum).collect()
}

/// Simple attention mechanism
/// For each token, comput ehow much to 'attend' to other tokens
/// then output a weighted combination of all token values
pub struct Attention {
    embed_dim: usize,
}

impl Attention {
    pub fn new(embed_dim: usize) -> Self {
        Self { embed_dim }
    }

    /// Compute attention for a series of embeddings
    /// Input: Vec of token embeddings
    /// Output: Vec of contextualised embeddings
    pub fn forward(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = embeddings.len();
        let scale = (self.embed_dim as f32).sqrt();

        let mut outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let query = &embeddings[i];
            // Compute attention scores for this query against other tokens
            let mut scores = Vec::with_capacity(seq_len);
            for key in embeddings.iter() {
                let score = dot_product(query, key) / scale;
                scores.push(score);
            }
            // Convert scores to probabilities
            let attn_weights = softmax(&scores);
            //Weighted sum of values (which are same as embeddings here)
            let mut output = vec![0.0; self.embed_dim];
            for (j, weight) in attn_weights.iter().enumerate() {
                let value = &embeddings[j];
                for (k, v) in value.iter().enumerate() {
                    output[k] += weight * v;
                }
            }
            outputs.push(output);
        }
        outputs
    }

    fn get_embeddings(tokens: &[&str], vocab: &HashMap<&str, Vec<f32>>) -> Vec<Vec<f32>> {
        tokens
            .iter()
            .map(|&t| vocab.get(t).cloned().unwrap_or(vec![0.0; 4]))
            .collect()
    }
}

fn main() {
    // Setup simple vocabulary with meaningful embeddings
    let vocab: HashMap<&str, Vec<f32>> = [
        // Questions have similar embeddings
        ("what", vec![1.0, 0.0, 0.0, 0.0]),
        ("is", vec![0.5, 0.5, 0.0, 0.0]),
        ("the", vec![0.3, 0.3, 0.3, 0.0]),
        ("capital", vec![0.0, 1.0, 0.0, 0.0]),
        ("of", vec![0.2, 0.2, 0.2, 0.2]),
        // Countries have distinct embeddings
        ("france", vec![0.0, 0.0, 1.0, 0.0]),
        ("germany", vec![0.0, 0.0, 0.0, 1.0]),
        ("japan", vec![0.0, 0.0, 0.8, 0.8]),
        // Answers
        ("paris", vec![0.1, 0.0, 0.9, 0.0]),  // Similar to france
        ("berlin", vec![0.1, 0.0, 0.0, 0.9]), // Similar to germany
    ]
    .into_iter()
    .collect();

    let attention = Attention::new(4);
    println!("Question 1: what is the capital of france");
    println!("Question 2: what is the capital of germany");

    let tokens1 = vec!["what", "is", "the", "capital", "of", "france"];
    let tokens2 = vec!["what", "is", "the", "capital", "of", "germany"];

    let embeddings1 = Attention::get_embeddings(&tokens1, &vocab);
    let embeddings2 = Attention::get_embeddings(&tokens2, &vocab);

    let output1 = attention.forward(&embeddings1);
    let output2 = attention.forward(&embeddings2);

    println!("Question 1: {:?}", tokens1);
    println!("Question 2: {:?}\n", tokens2);

    for i in 0..tokens1.len() {
        println!(
            "Position {} | {:>8} → {:?}",
            i,
            tokens1[i],
            output1[i]
                .iter()
                .map(|x| format!("{:.3}", x))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "           | {:>8} → {:?}",
            tokens2[i],
            output2[i]
                .iter()
                .map(|x| format!("{:.3}", x))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}
