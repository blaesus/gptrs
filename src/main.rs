mod tokenize;

use std;
use crate::tokenize::{bbpe_compress, bbpe_decompress};

fn read_text_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}

fn text_to_bytes(text: &str) -> Vec<u8> {
    text.bytes().collect()
}


fn main() {
    let path = "./data/shakespear.txt";
    match read_text_file(path) {
        Ok(content) => {
            // shorten the first 100 characters
            let content = &content[..20000];
            let bytes = text_to_bytes(content);
            let result = bbpe_compress(&bytes);
            println!("\n###Compressed ({} tokens): {:?}\n", result.compressed.len(), result.compressed.iter().take(100).collect::<Vec<&u16>>());
            let decompressed = bbpe_decompress(&result);
            println!("\n###Decompressed:\n {}", decompressed);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
