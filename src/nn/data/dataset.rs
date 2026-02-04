use super::{test_set::TestSet, train_set::TrainSet};
use super::shuffle::shuffle;

pub struct DataSet {
    batch_size: usize,
    data: Vec<Vec<u8>>,
}

impl DataSet {
    pub fn new(batch_size: usize, data: Vec<Vec<u8>>) -> Self {
        DataSet {
            batch_size: batch_size,
            data: data,
        }
    }

    pub fn split(mut self, test_split: f32) -> (TrainSet, TestSet) {
    if test_split >= 1.0 || test_split <= 0.0 {
        panic!("Split should be between 0 and 1");
    }

    shuffle(&mut self.data);

    let test_size = (self.data.len() as f32 * test_split) as usize;
    let train_size = self.data.len() - test_size;

    let test_data = self.data.split_off(train_size);
    let train_data = self.data;

    (
        TrainSet::new(self.batch_size, train_data), 
        TestSet::new(test_data)
    )
}

}





