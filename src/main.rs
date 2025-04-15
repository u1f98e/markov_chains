use std::{
    fmt::Display,
    fs::File,
    io::{self, BufRead, BufReader, Write},
    ops::Deref,
    path::PathBuf,
};

use clap::Parser;
use rand::distr::{weighted::WeightedIndex, Distribution};
use serde::{Deserialize, Serialize};
use utf8_chars::BufReadCharsExt;

#[derive(Parser)]
struct Args {
    /// A text file containing text to learn off of, or a
    /// .bin file containing a previously saved transition matrix.
    input_file: Option<PathBuf>,
    initial_phrase: Option<String>,

    /// Number of tokens to generate
    #[arg(short('s'), long, default_value_t = 200)]
    output_size: u32,

    /// The number of tokens to use per state in the transition matrix. Default
    /// is 2, performance decreases exponentially when increased.
    ///
    /// When loading an existing matrix this option is ignored.
    #[arg(short('t'), long, default_value_t = 2)]
    state_size: u32,

    /// Save transition matrix to a binary file instead of generating tokens
    #[arg(long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "markov.bin")]
    save: Option<PathBuf>,
}

/// Magic numbers prefixed to exported transition matrix files, so we can detect
/// them more easily.
static MAGIC_FILE_BYTES: [u8; 3] = [0x3, 0x4, 0x5];

#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct State(Vec<String>);

impl State {
    pub fn new(tokens: Vec<String>) -> Self {
        Self(tokens)
    }

    pub fn from_slice(tokens: &[String], state_size: usize) -> Self {
        // Clone the last `size` tokens into the front of `last_tokens`
        let index = tokens.len().saturating_sub(state_size);
        let slice = &tokens[index..];
        Self::new(slice.to_vec())
    }
}

impl Deref for State {
    type Target = [String];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for token in &self.0 {
            write!(f, "{} ", token)?;
        }
        Ok(())
    }
}

// Making a trait for this to benchmark performance for
// a few implementations.
trait StateIndex {
    fn get_state(&self, index: usize) -> Option<&State>;
    fn get_index(&self, state: &State) -> Option<usize>;
    fn insert(&mut self, index: usize, state: State);
    fn len(&self) -> usize;
}

impl StateIndex for Vec<State> {
    fn get_state(&self, index: usize) -> Option<&State> {
        self.get(index)
    }

    fn get_index(&self, state: &State) -> Option<usize> {
        self.iter().position(|s| s == state)
    }

    fn insert(&mut self, index: usize, state: State) {
        self.insert(index, state)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

type MarkovGenerator = MarkovGeneratorBase<Vec<State>>;

// TODO: Consider a custom ser/de impelmentation to avoid writing the size for every state
#[derive(Serialize, Deserialize)]
struct MarkovGeneratorBase<S>
where
    S: StateIndex + Default,
{
    mat: sprs::CsMat<u16>,
    states: S,
    state_size: u32,
}

impl<S> MarkovGeneratorBase<S>
where
    S: StateIndex + Default,
{
    pub fn from_tokens(tokens: &Vec<String>, state_size: u32) -> Self {
        let max_possible_states = tokens.len() - (state_size as usize - 1);
        let mut state_indicies: S = Default::default();
        let mut mat = sprs::CsMat::zero((max_possible_states, max_possible_states));
        let mut current_rows = 0;

        let mut i = 0;
        let mut last_state_index = None;
        while (i + state_size as usize) <= tokens.len() {
            let state = State::from_slice(&tokens[i..i + state_size as usize], state_size as usize);

            let row = match state_indicies.get_index(&state) {
                Some(r) => r,
                None => {
                    let row = current_rows;
                    state_indicies.insert(row, state.clone());
                    current_rows += 1;
                    row
                }
            };

            if let Some(col) = last_state_index {
                match mat.get_mut(row, col) {
                    Some(count) => *count += 1,
                    None => mat.insert(row, col, 1),
                }
            }

            i += 1;
            last_state_index = Some(row);
        }

        Self {
            mat,
            states: state_indicies,
            state_size,
        }
    }

    fn random_state_index(&self) -> usize {
        rand::random_range(0..self.states.len())
    }

    pub fn random_state(&self) -> &State {
        self.states.get_state(self.random_state_index()).unwrap()
    }

    pub fn predict(&self, current_state: &State) -> State {
        let row_index = self
            .states
            .get_index(current_state)
            .unwrap_or_else(|| self.random_state_index());
        let row_view: sprs::CsVecView<_> = self.mat.outer_view(row_index).unwrap();

        // If no next tokens are available, pick one at random
        if row_view.nnz() == 0 {
            return self.random_state().clone();
        }

        let (columns, weights): (Vec<usize>, Vec<&u16>) = row_view.iter().unzip();
        let dist = WeightedIndex::new(weights).unwrap();
        let selected_col = columns[dist.sample(&mut rand::rng())];
        self.states.get_state(selected_col).unwrap().clone()
    }
}

fn tokenize_input<R: BufRead>(reader: &mut R) -> io::Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    fn finish_token(current_token: &mut String, tokens: &mut Vec<String>) {
        if !current_token.is_empty() {
            tokens.push(current_token.clone());
            current_token.clear();
        }
    }

    for ch in reader.chars() {
        let ch = ch?;
        if ch.is_whitespace() {
            finish_token(&mut current_token, &mut tokens);
        } else if ch.is_ascii_punctuation() {
            if !current_token.ends_with(|c: char| c.is_ascii_punctuation()) {
                finish_token(&mut current_token, &mut tokens);
            }

            current_token.push(ch.to_ascii_lowercase());
        } else {
            current_token.push(ch.to_ascii_lowercase());
        }
    }

    finish_token(&mut current_token, &mut tokens);
    Ok(tokens)
}

fn format_output(tokens: &Vec<String>) -> String {
    fn capitalize(word: &str) -> String {
        let mut c = word.chars();
        match c.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
        }
    }

    let mut output = String::new();
    let mut capitalize_next = true;
    for token in tokens {
        // Add a space before this token, unless it's punctuation or the beginning of the output.
        let first_char = token.chars().next();
        if !(first_char.is_none_or(|c| c.is_ascii_punctuation()) || output.is_empty()) {
            output.push(' ');
        }

        if capitalize_next {
            capitalize_next = false;
            output.push_str(&capitalize(&token));
        } else {
            output.push_str(&token);
        }

        if first_char.is_some_and(|c| c == '.' || c == ';' || c == '!' || c == '?') {
            capitalize_next = true;
        }
    }

    output
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let mut reader: Box<dyn BufRead> = if let Some(path) = args.input_file {
        Box::new(BufReader::new(File::open(path).unwrap()))
    } else {
        Box::new(BufReader::new(io::stdin()))
    };

    let file_preview = reader
        .fill_buf()
        .expect("Failed to read initial bytes from file");

    let markov = if file_preview[..3] == MAGIC_FILE_BYTES {
        reader.consume(MAGIC_FILE_BYTES.len()); // Skip magic bytes

        // This is a transition matrix file, load it instead of training
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        postcard::from_bytes(&buf).expect("Expected valid binary matrix file")
    } else {
        let input_tokens = tokenize_input(&mut reader)?;
        MarkovGenerator::from_tokens(&input_tokens, args.state_size)
    };

    if let Some(output_path) = args.save {
        let bytes = postcard::to_allocvec(&markov).unwrap();
        let mut output_file = File::create(output_path).expect("Failed to open output file");
        output_file
            .write_all(&MAGIC_FILE_BYTES)
            .expect("Failed to write prefix to file");
        output_file
            .write_all(&bytes)
            .expect("Failed to write transition matrix to file");
        return Ok(());
    }

    let mut output = Vec::new();
    let mut prev_state = if let Some(s) = args.initial_phrase {
        let initial_tokens = tokenize_input(&mut s.as_bytes())?;
        let s = State::from_slice(&initial_tokens, markov.state_size as usize);
        output.extend(initial_tokens);
        s
    } else {
        let s = markov.random_state();
        output.extend_from_slice(&s);
        s.clone()
    };

    let total_tokens = args.output_size / markov.state_size;
    for _ in 0..total_tokens {
        prev_state = markov.predict(&prev_state);
        output.extend_from_slice(&prev_state);
    }

    println!("{}", format_output(&output));
    Ok(())
}
