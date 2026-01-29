use std::path::{PathBuf};
use RustML::csv_reader::open_data;

fn main() {
    
    let digits_path = PathBuf::from("./Digit/train/train.csv");
    let data = open_data(&digits_path);

    println!("Successfully loaded {} images.", data.len());
}