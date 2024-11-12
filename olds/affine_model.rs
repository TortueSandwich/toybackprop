use rand::prelude::*;

use crate::trainning_sets::*;

pub static TRAIN_PTR: &TrainDataSet<1> = &LINEAR_TRAIN;
pub const EPS: f64 = 0.001;
pub const RATE: f64 = 0.1;
pub const EPOCH_COUNT: usize = 150;

fn rand_float() -> f64 {
    let mut rng: ThreadRng = rand::thread_rng();
    rng.gen_range(0.0..1.0)
}

fn cost(w: f64, b: f64) -> f64 {
    let mut result = 0.;
    for datal in TRAIN_PTR.iter() {
        let DataLine {
            entry: [x],
            expected,
        } = datal;

        let y = x * w + b;
        let d = y - expected;
        result += d * d;
    }
    result / (TRAIN_PTR.data.len() as f64)
}

#[allow(unused)]
pub fn run() {
    let mut w = rand_float() * 10.0;
    let mut b = rand_float() * 5.0;
    
    for _ in 0..EPOCH_COUNT {
        let c = cost(w, b);
        let dw = (cost(w + EPS, b) - c) / EPS;
        let db = (cost(w, b + EPS) - c) / EPS;
        w -= dw * RATE;
        b -= db * RATE;
    }

    println!("w * x + b");
    println!("w = {}", w);
    println!("b = {}", b);
}
