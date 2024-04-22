use std;
fn read_text_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}
fn main() {
    let path = "./data/shakespear.txt";
    match read_text_file(path) {
        Ok(content) => {
            // shorten the first 100 characters
            let content = &content[..100];
            println!("{}", content);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
