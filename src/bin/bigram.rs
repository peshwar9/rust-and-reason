use std::collections::HashMap;
use std::env::args;
use std::io::BufRead;

#[derive(Default)]
pub struct BigramModel {
    counts: HashMap<String, HashMap<String, usize>>,
}

impl BigramModel {
    pub fn new() -> Self {
        Self::default()
    }

    // Train from a reader

    pub fn train(&mut self, corpus: &str) {
        corpus
            .split(|c| matches!(c, '.' | '!' | '?')) // split into sentences
            .filter_map(|s| {
                let words: Vec<String> = s
                    .split_whitespace()
                    .map(|w| {
                        w.trim_matches(|c: char| !c.is_alphanumeric())
                            .to_lowercase()
                    })
                    .filter(|w| !w.is_empty())
                    .collect();
                if words.len() < 2 { None } else { Some(words) } // skip sentences with <2 words
            })
            .for_each(|words| {
                words.windows(2).for_each(|w| {
                    let (a, b) = (&w[0], &w[1]);
                    *self
                        .counts
                        .entry(a.clone())
                        .or_insert_with(HashMap::new)
                        .entry(b.clone())
                        .or_insert(0) += 1;
                });
            });
    }

    pub fn train_from_reader<R: BufRead>(&mut self, reader: R) {
        reader
            .lines()
            .filter_map(|line| line.ok())
            .for_each(|line| self.train(&line));
    }

    // Predict the next word based on max count
    pub fn predict(&self, word: &str) -> Option<&str> {
        let word = word.to_lowercase();
        self.counts.get(&word).and_then(|next_words| {
            next_words
                .iter()
                .max_by_key(|&(_, count)| *count)
                .map(|(word, _)| word.as_str())
        })
    }

    // Predict the next word from a reader
    pub fn generate(&self, start_word: &str, max_len: usize) -> Vec<String> {
        std::iter::successors(Some(start_word.to_string()), |word| {
            self.predict(word).map(|s| s.to_string())
        })
        .take(max_len + 1)
        .collect()
    }

    // Print the trained model
    pub fn print_model(&self) {
        self.counts.iter().for_each(|(word, next_words)| {
            next_words
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(next_word, count)| {
                    println!("{} -> {} ({})", word, next_word, count);
                });
        })
    }
}

fn main() {
    println!("Bigram Model");
    let mut model = BigramModel::new();
    let corpus = "the quick brown fox jumps over the lazy dog. \
              the dog was sleeping under the tree. \
              the fox was very clever and quick. \
              a quick brown fox is always alert. \
              the lazy dog did not move. \
              the brown fox ran through the forest. \
              the forest was dark and deep. \
              the dog barked at the fox. \
              the quick fox escaped into the forest. \
              the lazy dog slept all day.";
    model.train(corpus);
    // model.print_model();

    let start_word = args().skip(1).next().unwrap_or_else(|| "the".to_string());

    let generated = model.generate(&start_word, 10);
    println!("Generated sequence: {:?}", generated);
}
