//!
//! LLM warmup: This is a simple tokenizer that maps words to unique IDs and vice versa.
//! It reads a vocabulary file "vocab.txt" where each line contains a single word.
//! The tokenizer provides functionality to load the vocabulary and retrieve IDs for words.
//!
//! Use `cargo run --bin tokenizer -- What is the capital of France`
//!
//! The vocab.txt file should be in the root directory and contain one workd per line
//!
//!
use std::collections::HashMap;
use std::env::args;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use thiserror::Error;

pub struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
        }
    }
    // Load from a reader (easy to test)
    pub fn load_vocab(&mut self) -> Result<(), TokenizerError> {
        //Read file and populate word_to_id and id_to_word
        let f = File::open("vocab.txt")?;
        let reader = std::io::BufReader::new(f);
        // for line_result in reader.lines() {
        //     let line = line_result?.trim().to_string();
        //     if line.is_empty() {
        //         continue;
        //     }
        //     let id = self.id_to_word.len();
        //     self.word_to_id.insert(line.clone(), id);
        //     self.id_to_word.push(line);
        // }
        let lines: Vec<String> = reader
            .lines()
            .map(|r| r.map(|s| s.trim().to_string()))
            .collect::<Result<_, _>>()?;

        self.word_to_id = lines
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect();
        self.id_to_word = lines;
        Ok(())
    }
}

fn main() {
    // Instantiate new Tokenizer and load vocab
    let mut tokenizer = Tokenizer {
        word_to_id: HashMap::new(),
        id_to_word: Vec::new(),
    };
    tokenizer.load_vocab().unwrap_or_else(|e| {
        eprintln!("Error loading vocab.txt file");
        std::process::exit(1);
    });
    println!("words length: {:?}", tokenizer.word_to_id.len());

    // Get the input text from command line arguments
    let input_text = args().skip(1).collect::<Vec<_>>();
    println!("Input text: {:?}", input_text);

    // Lookup the token for each word in the input text
    let token_vec = input_text
        .into_iter()
        .map(|word| tokenizer.word_to_id.get(&word).copied().unwrap_or(0))
        .collect::<Vec<usize>>();
    println!("TokenIds: {:?}", token_vec);
}
