use crate::math::matrix::Matrix;
use std::{fs::File, io::BufReader, io::Read};

pub fn parse(file_path: &str) -> Vec<Matrix> {
    match File::open(file_path) {
        Ok(file) => {
            let mut br = BufReader::new(file);
            let mut magic_num = [0_u8; 4];
            let _ = br.read(&mut magic_num[..]).unwrap();
            println!("Magic number {:?}", magic_num);
            // We are reading MNIST idx file which has ubyte data, so no need to check for data type.
            // Only checking for dimensions as images have 3 dimensions and labels has 1
            let dim_count = magic_num[3] as usize;
            println!("Dimensions count: {}", dim_count);

            let mut y: Vec<Matrix> = vec![];
            if dim_count == 1 {
                let mut first_dim = [0_u8; 4];
                let _ = br.read(&mut first_dim).unwrap();
                for _ in 0..u32::from_be_bytes(first_dim) {
                    let mut d = [0_u8; 1];
                    let _ = br.read_exact(&mut d[..]).unwrap();
                    y.push(Matrix::new(1, 1, vec![d[0] as f64]));
                }
            } else {
                let mut first_dim = [0_u8; 4];
                let _ = br.read(&mut first_dim).unwrap();
                // Flatten vector of the 28x28 image
                for _ in 0..u32::from_be_bytes(first_dim) {
                    let mut buf_d = [0_u8; 784];
                    let _ = br.read_exact(&mut buf_d[..]).unwrap();
                    y.push(Matrix::new(
                        784,
                        1,
                        Vec::from(buf_d).into_iter().map(|x| x as f64).collect(),
                    ));
                }
            }

            y
        }
        Err(e) => {
            println!("{:?}", e);
            panic!("Failed to open file");
        }
    }
}
