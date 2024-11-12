use rand::Rng;

use crate::trainning_sets::*;
use crate::tools::*;

pub static TRAIN_PTR: &TrainDataSet<2> = &XOR_TRAIN;
pub static EPOCH_COUNT: usize = 50000;
pub const EPS: f64 = 0.0001;
pub const RATE: f64 = 0.2;

#[derive(Debug)]
struct NeuronGate {
    w1: f64,
    w2: f64,
    b: f64,
}

fn cost(w1: f64, w2: f64, b: f64) -> f64 {
    let mut result = 0.;
    for datal in TRAIN_PTR.iter() {
        let DataLine {
            entry: [x1, x2],
            expected,
        } = datal;
        let y = sigmoid(x1 * w1 + x2 * w2+ b);
        let d = y - expected;
        result += d * d;
    }
    result / (TRAIN_PTR.data.len() as f64)
}

impl NeuronGate {
    pub fn rand() -> Self {
        let mut rng = rand::thread_rng();
        let w1 = rng.gen_range(0.0..1.0);
        let w2 = rng.gen_range(0.0..1.0);
        let b = rng.gen_range(0.0..1.0);
        Self { w1, w2, b }
    }

    fn train(&mut self) {
        for _ in 0..EPOCH_COUNT {
            let c = cost(self.w1, self.w2, self.b);
            let dw1 = (cost(self.w1 + EPS, self.w2, self.b) - c) / EPS;
            let dw2 = (cost(self.w1,  self.w2 + EPS, self.b) - c) / EPS;
            let db = (cost(self.w1,  self.w2, self.b + EPS) - c) / EPS;
            self.w1 -= dw1 * RATE;
            self.w2 -= dw2 * RATE;
            self.b -= db * RATE;
        }
    }

    fn test_on(&self, data : &TrainDataSet<2>) {
        let red = "\x1b[31m";
        let green = "\x1b[32m";
        let orange = "\x1b[33m";
        for DataLine {
            entry: [x1, x2],
            expected,
        } in data.iter() {
            let y = sigmoid(x1 * self.w1 + x2 * self.w2 + self.b);
            let res = (y - expected).abs();
            let color = match res {
                res if res < 0.1 => green,
                res if res < 0.3 => orange,
                _ => red,
            };
            println!("{x1} | {x2} = {color}{y}\x1b[0m  {res}");
        }
    }

}


#[allow(unused)]
pub fn run() {
    let mut gate = NeuronGate::rand();
    println!("{gate:?}");
    println!("{}", cost(gate.w1, gate.w2, gate.b));
    gate.train();
    println!("{gate:?}");
    println!("{}", cost(gate.w1, gate.w2, gate.b));
    println!();

    gate.test_on(TRAIN_PTR);
}
