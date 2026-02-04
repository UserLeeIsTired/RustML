pub struct TestSet {
    data: Vec<Vec<u8>>,
    pub n: usize,
}

impl TestSet {
    pub fn new(data: Vec<Vec<u8>>) -> Self {
        let n = data.len();
        TestSet {
            data: data,
            n: n,
        }
    }

    pub fn get(&self) -> &[Vec<u8>] {
        &self.data[0..self.data.len()]
    }
}