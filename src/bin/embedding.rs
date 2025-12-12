use std::collections::HashMap;

#[derive(Default, Debug)]
pub struct Embedding {
    table: Vec<Vec<f32>>,
    embed_dim: usize,
}

impl Embedding {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn from_embeddings(vocab_size: usize, embed_dim: usize) -> Self {
        let table = Self::load_embeddings(vocab_size, embed_dim);
        Self { table, embed_dim }
    }

    pub fn load_embeddings(vocab_size: usize, embed_dim: usize) -> Vec<Vec<f32>> {
        (0..vocab_size)
            .map(|i| {
                (0..embed_dim)
                    .map(|j| ((i * 1001 + j * 23) % 1000) as f32)
                    .map(|hash| (hash / 500.0) - 1.0)
                    .collect()
            })
            .collect()
    }
    pub fn lookup(&self, index: usize) -> Option<&Vec<f32>> {
        self.table.get(index)
    }

    pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(x, y)| x * y).sum();
        let mag1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (mag1 * mag2)
    }
}

fn main() {
    // Load random embeddings for testing

    let vocab: HashMap<&str, usize> = [
        ("<PAD>", 0),
        ("<UNK>", 1),
        ("king", 2),
        ("queen", 3),
        ("man", 4),
        ("woman", 5),
        ("cat", 6),
        ("dog", 7),
        ("paris", 8),
        ("france", 9),
    ]
    .into_iter()
    .collect();

    // Load embeddings
    let embed_dim = 5;
    let embedding = Embedding::from_embeddings(vocab.len(), embed_dim);

    // Get stored embeddings
    let _ = embedding
        .table
        .iter()
        .enumerate()
        .map(|(i, _)| {
            println!(
                "Embedding for index{:?} with dimension of {:?} is {:?}",
                i,
                embedding.embed_dim,
                embedding.lookup(i).unwrap_or(&vec![])
            )
        })
        .collect::<Vec<_>>();

    // Compute cosine similarity between a word pair
    let similarity_index = embedding
        .lookup(2)
        .zip(embedding.lookup(3))
        .map(|(king, queen)| Embedding::cosine_similarity(king, queen))
        .unwrap_or(0.0);
    println!(
        "Cosine similarity between 'king' and 'queen': {:?}",
        similarity_index
    );
}
