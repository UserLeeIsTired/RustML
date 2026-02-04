use rand::Rng;  

pub fn shuffle<T>(vector: &mut Vec<T>) {
    let mut rng = rand::rng();

    for i in (1..vector.len()).rev() {
        let j = rng.random_range(0..=i);
        vector.swap(i, j);
    }
}