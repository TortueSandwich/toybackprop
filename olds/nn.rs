use crate::trainning_sets::*;
use nalgebra::{DMatrix, MatrixCross};
use rand::{thread_rng, Rng};
use std::ops::Mul;
use crate::tools::*;

pub static TRAIN_PTR: &TrainDataSet<2> = &XOR_TRAIN;
pub static EPOCH_COUNT: usize = 50000;
pub const EPS: f64 = 0.001;
pub const RATE: f64 = 0.2;

// Col nb de noeux dans une couche
// #[derive(Clone)]
struct ModelXor {
    a0: DMatrix<f64>,
    w1: DMatrix<f64>, // derniere colone = 0,0,1
    a1: DMatrix<f64>,
    w2: DMatrix<f64>, // derniere couche donc pas de 0,0,1
    a2: DMatrix<f64>,
}

impl ModelXor {
    fn rand() -> Self {
        let mut rng = thread_rng();
        Self {
            a0: DMatrix::zeros(1, 3), // Matrice d'activations a0
            w1: DMatrix::from_fn(3, 3, |_, _| rng.gen_range(-1.0..1.0)), // Poids aléatoires w1
            a1: DMatrix::zeros(1, 3), // Matrice d'activations a1
            w2: DMatrix::from_fn(3, 1, |_, _| rng.gen_range(-1.0..1.0)), // Poids aléatoires w2
            a2: DMatrix::zeros(1, 1), // Matrice de sortie a2
        }
    }

    fn cost(&mut self) -> f64 {
        let result = TRAIN_PTR.iter().fold(
            0.,
            |acc,
             DataLine {
                 entry: [x1, x2],
                 expected,
             }| {
                let y = self.forward(*x1, *x2);
                let d = y - expected;
                acc + d * d
            },
        );
        result / (TRAIN_PTR.data.len() as f64)
    }

    fn sigmoid(m: &mut DMatrix<f64>) {
        m.apply(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    }
    

    fn forward(&mut self, x1: f64, x2: f64) -> f64 {
        self.a0[(0, 0)] = x1;
        self.a0[(0, 1)] = x2;

        // Calcul des activations pour la première couche
        self.a1 = &self.a0 * &self.w1;
        Self::sigmoid(&mut self.a1);

        // Calcul des activations pour la deuxième couche
        self.a2 = &self.a1 * &self.w2;
        Self::sigmoid(&mut self.a2);

        self.a2[(0, 0)]
    }

    fn test_on(&mut self, data: &TrainDataSet<2>) {
        let red = "\x1b[31m";
        let green = "\x1b[32m";
        let orange = "\x1b[33m";
        for DataLine {
            entry: [x1, x2],
            expected,
        } in data.iter()
        {
            let y = self.forward(*x1, *x2);
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


pub fn run() {
    let mut x = ModelXor::rand();
    x.test_on(&XOR_TRAIN);
    println!("{}", x.cost());
}
