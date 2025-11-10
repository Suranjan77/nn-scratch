use rand::Rng;

#[inline]
fn function_to_approximate(x: f64) -> [f64; 2] {
    match x.floor() % 4.0 {
        x if x == 0.0 => [0.0, 0.0],
        x if x == 1.0 => [0.0, 1.0],
        x if x == 2.0 => [1.0, 0.0],
        x if x == 3.0 => [1.0, 1.0],
        _ => unreachable!(),
    }
}


pub fn generate_train(sample_size: usize) -> Vec<(f64, [f64; 2])> {
    let mut data: Vec<(f64, [f64; 2])> = Vec::with_capacity(sample_size);

    let mut rng = rand::rng();
    for _ in 0..sample_size {
        let x: f64 = rng.random_range(0.0..60.0);
        data.push((x, function_to_approximate(x)))
    }

    data
}


pub fn generate_test(sample_size: usize) -> Vec<(f64, [f64; 2])> {
    let mut data: Vec<(f64, [f64; 2])> = Vec::with_capacity(sample_size);

    let mut rng = rand::rng();
    for _ in 0..sample_size {
        let x: f64 = rng.random_range(60.0..90.0);
        data.push((x, function_to_approximate(x)))
    }

    data
}


pub fn generate_validation(sample_size: usize) -> Vec<(f64, [f64; 2])> {
    let mut data: Vec<(f64, [f64; 2])> = Vec::with_capacity(sample_size);

    let mut rng = rand::rng();
    for _ in 0..sample_size {
        let x: f64 = rng.random_range(90.0..100.0);
        data.push((x, function_to_approximate(x)))
    }

    data
}
