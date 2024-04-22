use std;
use std::collections::HashMap;

fn read_text_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}

fn text_to_bytes(text: &str) -> Vec<u8> {
    text.bytes().collect()
}

struct CompressResult {
    compressed: Vec<u16>,
    dict: Vec<(u16, u16)>,
}

fn byte_pair_key(byte1: u16, byte2: u16) -> (u16, u16) {
    (byte1, byte2)
}

fn bbpe_compress(input: &[u8]) -> CompressResult {
    let mut dict: Vec<(u16, u16)> = Vec::new();
    for i in 0..=255 {
        dict.push((i as u16, 0));
    }

    let mut compressed: Vec<u16> = input.iter().map(|u| *u as u16).collect(); // convert u8 to u16
    while dict.len() < 2000 {
        // Find max frequency pair
        let (max_freq, max_freq_pair) = {
            let mut frequency_dict = HashMap::new();
            for i in 0..compressed.len() - 1 {
                let key = byte_pair_key(compressed[i], compressed[i + 1]);
                let counter = frequency_dict.entry(key).or_insert(0);
                *counter += 1;
            }
            let (max_freq_pair, max_freq) = frequency_dict.iter().max_by_key(|&(_, count)| count).unwrap();
            (*max_freq, *max_freq_pair)
        };
        println!("Max frequency pair: {:?} with frequency {}", max_freq_pair, max_freq);
        dict.push(max_freq_pair);

        let mut i = 0;
        while i < compressed.len() - 1 {
            let key = byte_pair_key(compressed[i], compressed[i + 1]);
            if key == max_freq_pair {
                compressed[i] = dict.len() as u16 - 1;
                compressed.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
    CompressResult { compressed, dict }
}

fn main() {
    let path = "./data/shakespear.txt";
    match read_text_file(path) {
        Ok(content) => {
            // shorten the first 100 characters
            let content = &content[..50000];
            println!("{}", content);
            let bytes = text_to_bytes(content);
            let result = bbpe_compress(&bytes);
            println!("Compressed ({} tokens)", result.compressed.len());
            println!("Dictionary: {:?}", result.dict);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
