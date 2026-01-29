use std::path::{PathBuf};

use RustML::csv_reader::open_data;
use RustML::nn::{Matrix, Layer, Sequential}; 

fn main() {

    // load data from train.csv
    
    let digits_path = PathBuf::from("./Digit/train/train.csv");
    let data = open_data(&digits_path);

    // each image has 784 pixel, where the first column is the label (true answer)

    println!("Successfully loaded {} images.", data.len());
    println!("Each image has {} pixels", data[0].len() - 1);

    // assume all hidden layers are relu and the output is softmax

    let mut seq = Sequential::new(
        0.005,
        vec![
            Layer::new(784, 64, "relu"),
            Layer::new(64, 32, "relu"),
            Layer::new(32, 16, "relu"),
            Layer::new(16, 10, "softmax"),
        ]
    );

    for i in 0..10 {
        for (j, row) in data.iter().take(20000).enumerate() {
            let test = seq.forward(Matrix::to_matrix(row));
            seq.backward(test, data[0][0] as usize);

            if j % 100 == 0 {
                println!("{i} epoch {j} / 20000")
            }
        }

        let mut correct = 0;
        for k in 41000..42000 {
            let test = seq.forward(Matrix::to_matrix(&data[k]));
            let (_, result) = test.get_max();
            if result == data[k][0] as usize {
                correct += 1;
            }
            println!("accuracy {}%", correct as f32/ 10.0);
        }
        println!("{i} epoch completed");
    }

}