use std::collections::HashMap;
use std::env::args;
use std::fs::File;
use std::io::{BufRead, BufReader};
use thiserror::Error;

#[derive(Default)]
pub struct Tokenizer {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self::default()
    }
    // Load from a reader (easy to test)
    pub fn load_vocab<R: BufRead>(&mut self, reader: R) -> Result<(), TokenizerError> {
        self.id_to_word = reader
            .lines()
            .map(|line| line.map(|l| l.trim().to_string()))
            .collect::<Result<Vec<String>, std::io::Error>>()?;

        self.word_to_id = self
            .id_to_word
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect::<HashMap<String, usize>>();

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Word not found in Vocab: {0}")]
    Vocab(String),
}

pub fn main() -> Result<(), TokenizerError> {
    // Load Vocab from file and instantiate Tokenizer
    let file = File::open("vocab.txt")?;
    let reader = BufReader::new(file);
    let mut tokenizer = Tokenizer::new();
    tokenizer.load_vocab(reader).unwrap_or_else(|e| {
        eprintln!("Error loading vocab: {}", e);
        std::process::exit(1);
    });
    println!("Words loaded length: {}", tokenizer.word_to_id.len());

    // Get input from user in command line
    let input_text = args().skip(1).collect::<Vec<String>>();
    println!("Input text is : {:?}", input_text);

    let tokens = input_text
        .into_iter()
        .map(|word| tokenizer.word_to_id.get(&word).copied().unwrap_or(1))
        .collect::<Vec<usize>>();
    println!("Token ids: {:?}", tokens);
    Ok(())
}
