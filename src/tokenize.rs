use std::collections::HashMap;

pub struct CompressResult {
    pub compressed: Vec<u16>,
    pub dict: Vec<(u16, u16)>,
}
fn byte_pair_key(byte1: u16, byte2: u16) -> (u16, u16) {
    (byte1, byte2)
}

fn find_most_frequent_pair(compressed: &[u16]) -> (u32, (u16, u16)) {
    let mut frequency_dict = HashMap::new();
    for i in 0..compressed.len() - 1 {
        let key = byte_pair_key(compressed[i], compressed[i + 1]);
        let counter = frequency_dict.entry(key).or_insert(0);
        *counter += 1;
    }
    let (max_freq_pair, max_freq) = frequency_dict.iter().max_by_key(|&(_, count)| count).unwrap();
    (*max_freq, *max_freq_pair)
}

const PLACEHOLDER: u16 = 0;

pub fn bbpe_compress(input: &[u8], dict_size: usize) -> CompressResult {
    assert!(dict_size >= 256);
    assert!(dict_size < 65536);
    let mut dict: Vec<(u16, u16)> = Vec::new();
    for i in 0..=255 {
        dict.push((i as u16, PLACEHOLDER)); // second int is meaningless
    }

    let mut compressed: Vec<u16> = input.iter().map(|u| *u as u16).collect(); // convert u8 to u16
    while dict.len() < dict_size {
        // Find max frequency pair
        let (max_freq, max_freq_pair) = find_most_frequent_pair(&compressed);
        if max_freq <= 1 {
            break;
        }
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

fn fully_decompressed(input: &[u16]) -> bool {
    input.iter().all(|&x| x <= 255)
}

pub fn bbpe_decompress(result: &CompressResult) -> String {
    let mut decompressed: Vec<u16> = Vec::new();
    for i in 0..result.compressed.len() {
        let token = result.compressed[i];
        let (byte1, byte2) = result.dict[token as usize];
        decompressed.push(byte1);
        if byte2 != PLACEHOLDER {
            decompressed.push(byte2);
        }
    }
    if fully_decompressed(&result.compressed) {
        return String::from_utf8(result.compressed.iter().map(|&u| u as u8).collect()).unwrap();
    } else {
        bbpe_decompress(&CompressResult { compressed: decompressed, dict: result.dict.clone() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_pair_key() {
        assert_eq!(byte_pair_key(1, 2), (1, 2));
    }

    #[test]
    fn test_find_most_frequent_pair() {
        let compressed = vec![1, 2, 3, 0, 1, 2, 0, 4, 1, 2, 4, 4, 1, 2, 4, 4];
        assert_eq!(find_most_frequent_pair(&compressed), (4, (1, 2)));
    }

    #[test]
    fn test_decompress_the_compressed() {
        let input = "Hello world! Hello you! お願いします~ありがとうございます~".repeat(10);
        let result = bbpe_compress(input.as_bytes(), 256+10);
        assert!(result.compressed.len() < input.len());
        assert_eq!(bbpe_decompress(&result).as_str(), input);
    }

    #[test]
    fn test_fully_decompressed() {
        assert_eq!(fully_decompressed(&[1, 2, 3, 4]), true);
        assert_eq!(fully_decompressed(&[1, 2, 3, 256, 257]), false);
    }
}
