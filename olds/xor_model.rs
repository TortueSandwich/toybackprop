#![allow(dead_code)]

use crate::{constant::*, tools::*};
use std::time::{SystemTime, UNIX_EPOCH};
use rand::{Rng, rngs::StdRng, SeedableRng};
#[derive(Debug, Clone)]
pub struct Xor {
    or_w1 : T, or_w2 : T, or_b : T,
    nand_w1 : T, nand_w2 : T, nand_b : T,
    
    and_w1 : T, and_w2 : T, and_b : T,
}

impl Model for Xor {
    fn rand() -> Self {
        #[allow(unused)]
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Le temps actuel est antérieur à l'époque Unix.")
            .as_secs();
        //let seed = 4021;
        let mut rng = StdRng::seed_from_u64(seed);
        Xor {
            or_w1 : rng.gen(), or_w2 : rng.gen(), or_b : rng.gen(),
            nand_w1 : rng.gen(), nand_w2 : rng.gen(), nand_b : rng.gen(),
            
            and_w1 : rng.gen(), and_w2 : rng.gen(), and_b : rng.gen(),
        }
    }
    fn forward(&self, x1: T,x2: T) -> T {
        let a = sigmoid(self.or_w1*x1 + self.or_w2*x2 + self.or_b);
        let b = sigmoid(self.nand_w1*x1 + self.nand_w2*x2 + self.nand_b);
        return sigmoid(a*self.and_w1 + b*self.and_w2 + self.and_b);
    }

    fn finite_diff(&mut self) -> Self {
        let mut g = Xor::rand();
        let c = self.cost();

        macro_rules! finite_diff_helper {
            ($field:ident) => {{
                let saved = self.$field;
                self.$field += EPS;
                g.$field = (self.cost() - c) / EPS;
                self.$field = saved;
            }};
        }

        finite_diff_helper!(or_w1);
        finite_diff_helper!(or_w2);
        finite_diff_helper!(or_b);
        finite_diff_helper!(nand_w1);
        finite_diff_helper!(nand_w2);
        finite_diff_helper!(nand_b);
        finite_diff_helper!(and_w1);
        finite_diff_helper!(and_w2);
        finite_diff_helper!(and_b);

        g
    }
    fn learn_with(&mut self, g : Self) {
        macro_rules! learn_with_helper {
            ($field:ident) => {{
                self.$field -= RATE * g.$field;
            }};
        }
        learn_with_helper!(or_w1);
        learn_with_helper!(or_w2);
        learn_with_helper!(or_b);
        learn_with_helper!(nand_w1);
        learn_with_helper!(nand_w2);
        learn_with_helper!(nand_b);
        learn_with_helper!(and_w1);
        learn_with_helper!(and_w2);
        learn_with_helper!(and_b);
    }
}

#[allow(unused)]
pub fn run() {
    let mut x = Xor::rand();
    
    for _ in 0..100_000 {
       let g = x.finite_diff();
       x.learn_with(g);
       println!("cost {}", x.cost());
    }
    println!("-----------------------------");
    println!("Complete brain");
    x.print_table(*TRAIN_PTR);
    println!("-----------------------------");
    println!("OR neuron");
    for arr in OR_TRAIN {
        let (i,j) = (arr[0], arr[1]);
        println!("{}|{} = {:?}", i,j, sigmoid(x.or_w1 * i + x.or_w2 * j + x.or_b));
    } 
    println!("-----------------------------");
    println!("NAND neuron");
    for arr in NAND_TRAIN {
        let (i,j) = (arr[0], arr[1]);
        println!("{}|{} = {:?}", i,j, sigmoid(x.nand_w1 * i + x.nand_w2 * j + x.nand_b));
    } 
    println!("-----------------------------");
    println!("AND neuron");
    for arr in AND_TRAIN {
        let (i,j) = (arr[0], arr[1]);
        println!("{}|{} = {:?}", i,j, sigmoid(x.and_w1 * i + x.and_w2 * j + x.and_b));
    } 
}

