use std::path::{PathBuf};

use RustML::csv_reader::open_data;
use RustML::nn::data::DataSet;
use RustML::nn::{Matrix, Layer, Sequential}; 

fn main() {

    // load data from train.csv
    
    let digits_path = PathBuf::from("./Digit/train/train.csv");
    let data = open_data(&digits_path);

    // each image has 784 pixel, where the first column is the label (true answer)

    println!("Successfully loaded {} images.", data.len());
    println!("Each image has {} pixels", data[0].len() - 1);

    let temp = DataSet::new(128, data);
    let (mut train, mut test) = temp.split(0.2);

    // assume all hidden layers are relu and the output is softmax

    let mut seq = Sequential::new(
        0.0001,
        128,
        vec![
            Layer::new(784, 32, "relu"),
            Layer::new(32, 32, "relu"),
            Layer::new(32, 16, "relu"),
            Layer::new(16, 10, "softmax"),
        ]
    );

    seq.train(10, train, test, true);

}