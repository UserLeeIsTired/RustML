use super::shuffle::shuffle;

pub struct TrainSet {
    batch_size: usize,
    data: Vec<Vec<u8>>,
    pub pointer: usize,
    pub n: usize,
}

impl TrainSet {
    pub fn new(batch_size: usize, data: Vec<Vec<u8>>) -> TrainSet {
        let n = (data.len() + batch_size - 1) / batch_size;
        TrainSet { 
            batch_size: batch_size,
            data: data,
            pointer: 0,
            n: n
        }
    }

    fn shuffle(&mut self) {
        shuffle(&mut self.data);
    }

    pub fn get(&mut self) -> &[Vec<u8>] {
        if self.pointer * self.batch_size >= self.data.len() {
            self.pointer = 0;
        }

        let start = self.pointer * self.batch_size;
        let end = (start + self.batch_size).min(self.data.len());
        let slice = &self.data[start..end];
        self.pointer += 1;

        slice
    }

    pub fn signal(&mut self) -> bool {
        if !(self.pointer * self.batch_size >= self.data.len()) {
            return true;
        }

        self.shuffle();
        self.pointer = 0;
        false
    }
}