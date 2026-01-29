use std::fs;
use std::path::{PathBuf};

fn get_image_paths(directory_path: &PathBuf) -> Vec<PathBuf> {
    fs::read_dir(directory_path)
        .map(|read_dir| {
            read_dir
                .filter_map(|entry| {
                    let path = entry.ok()?.path();
                    let ext = path.extension()?.to_str()?;
                    
                    if path.is_file() && (ext == "jpg" || ext == "jpeg" || ext == "png") {
                        Some(path)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_else(|_| vec![]) 
}

pub fn get_images_data(directory_path: &PathBuf) -> Vec<Vec<f32>> {
    get_image_paths(directory_path)
        .into_iter()
        .filter_map(|path| {
            let img = image::open(&path).ok()?;
            Some(img.to_luma32f().into_raw())
        })
        .collect()
}