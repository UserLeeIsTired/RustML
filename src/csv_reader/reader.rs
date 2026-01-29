use std::{fs::File, path::PathBuf};
use std::io::{BufRead, BufReader};

pub fn open_data(filepath: &PathBuf) -> Vec<Vec<u8>> {
    let file = File::open(filepath).expect("Cannot open file");
    let reader = BufReader::new(file);

    reader.lines()
        .filter_map(|line| {
            let l = line.ok()?; 
            
            let row: Vec<u8> = l.split(',')
                .filter_map(|s| s.trim().parse::<u8>().ok())
                .collect();
            
            if row.is_empty() { None } else { Some(row) }
        })
        .collect()
}