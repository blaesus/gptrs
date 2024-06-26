mod tokenize;
mod matrix;
mod nn;
mod rand;

use std;
use crate::tokenize::{bbpe_compress, bbpe_decompress};

fn read_text_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}


fn main() {
    let path = "./data/shakespear.txt";
    match read_text_file(path) {
        Ok(content) => {
            let result = bbpe_compress(content.as_bytes(), 1000);
            println!("\n###Compressed ({} tokens): {:?}\n", result.compressed.len(), result.compressed.iter().take(100).collect::<Vec<&u16>>());
            let decompressed = bbpe_decompress(&result);
            println!("\n###Decompressed:\n {}", decompressed);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
