use std::ops::SubAssign;

use rand::prelude::*;

use crate::tools::*;
use crate::trainning_sets::*;

pub static TRAIN_PTR: &TrainDataSet<2> = &XOR_TRAIN;
pub static EPOCH_COUNT: usize = 50000;
pub const EPS: f64 = 0.001;
pub const RATE: f64 = 0.2;

#[derive(Debug)]
struct NeuronGate {
    w1: f64,
    w2: f64,
    b: f64,
}

impl NeuronGate {
    pub fn rand() -> Self {
        let mut rng = rand::thread_rng();
        let w1 = rng.gen_range(0.0..1.0);
        let w2 = rng.gen_range(0.0..1.0);
        let b = rng.gen_range(0.0..1.0);
        Self { w1, w2, b }
    }
}

#[derive(Debug)]
struct ThreeNeuronGate {
    or: NeuronGate,
    nand: NeuronGate,
    and: NeuronGate,
}

impl ThreeNeuronGate {
    ///feeds
    fn foward(&self, x: f64, y: f64) -> f64 {
        let NeuronGate { w1, w2, b: bias } = self.or;
        let a = sigmoid(x * w1 + w2 * y + bias);
        let NeuronGate { w1, w2, b: bias } = self.nand;
        let b = sigmoid(x * w1 + w2 * y + bias);
        let NeuronGate { w1, w2, b: bias } = self.and;
        // inputs are from the previous layer
        sigmoid(a * w1 + w2 * b + bias)
    }

    fn cost(&self) -> f64 {
        let result = TRAIN_PTR.iter().fold(
            0.,
            |acc,
             DataLine {
                 entry: [x1, x2],
                 expected,
             }| {
                let y = self.foward(*x1, *x2);
                let d = y - expected;
                acc + d * d
            },
        );
        result / (TRAIN_PTR.data.len() as f64)
    }

    fn rand() -> ThreeNeuronGate {
        ThreeNeuronGate {
            or: NeuronGate::rand(),
            nand: NeuronGate::rand(),
            and: NeuronGate::rand(),
        }
    }

    fn train(&mut self) {
        for _ in 0..EPOCH_COUNT {
            let g = self.finite_diff();
            *self -= g;
        }
    }

    fn finite_diff(&mut self) -> ThreeNeuronGate {
        let c = self.cost();
        let mut g = ThreeNeuronGate::rand();

        let saved = self.or.w1;
        self.or.w1 += EPS;
        g.or.w1 = (self.cost() - c) / EPS;
        self.or.w1 = saved;

        let saved = self.or.w2;
        self.or.w2 += EPS;
        g.or.w2 = (self.cost() - c) / EPS;
        self.or.w2 = saved;

        let saved = self.or.b;
        self.or.b += EPS;
        g.or.b = (self.cost() - c) / EPS;
        self.or.b = saved;

        let saved = self.nand.w1;
        self.nand.w1 += EPS;
        g.nand.w1 = (self.cost() - c) / EPS;
        self.nand.w1 = saved;

        let saved = self.nand.w2;
        self.nand.w2 += EPS;
        g.nand.w2 = (self.cost() - c) / EPS;
        self.nand.w2 = saved;

        let saved = self.nand.b;
        self.nand.b += EPS;
        g.nand.b = (self.cost() - c) / EPS;
        self.nand.b = saved;

        let saved = self.and.w1;
        self.and.w1 += EPS;
        g.and.w1 = (self.cost() - c) / EPS;
        self.and.w1 = saved;

        let saved = self.and.w2;
        self.and.w2 += EPS;
        g.and.w2 = (self.cost() - c) / EPS;
        self.and.w2 = saved;

        let saved = self.and.b;
        self.and.b += EPS;
        g.and.b = (self.cost() - c) / EPS;
        self.and.b = saved;

        g
    }

    
    fn test_on(&self, data : &TrainDataSet<2>) {
            let red = "\x1b[31m";
            let green = "\x1b[32m";
            let orange = "\x1b[33m";
            for DataLine {
                entry: [x1, x2],
                expected,
            } in data.iter() {
                let y = self.foward(*x1,*x2);
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

impl SubAssign for ThreeNeuronGate {
    fn sub_assign(&mut self, rhs: Self) {
        self.or.w1 -= RATE * rhs.or.w1;
        self.or.w2 -= RATE * rhs.or.w2;
        self.or.b -= RATE * rhs.or.b;
        self.nand.w1 -= RATE * rhs.nand.w1;
        self.nand.w2 -= RATE * rhs.nand.w2;
        self.nand.b -= RATE * rhs.nand.b;
        self.and.w1 -= RATE * rhs.and.w1;
        self.and.w2 -= RATE * rhs.and.w2;
        self.and.b -= RATE * rhs.and.b;
    }
}

#[allow(unused)]
pub fn run() {
    let mut gate = ThreeNeuronGate::rand();
    // println!("{gate:#?}");
    println!("{}", gate.cost());
    gate.test_on(TRAIN_PTR);
    gate.train();
    println!("{}", gate.cost());

    gate.test_on(TRAIN_PTR);

}
