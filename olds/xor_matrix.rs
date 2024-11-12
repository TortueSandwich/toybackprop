#![allow(dead_code)]

use std::cell::Cell;
use crate::{aiframework::matrice::Matrix, tools::Model, constant::*};

#[derive(Default)]
struct Xor {
    a0 :Cell<Matrix<1,2,T>>,
    w1 :Matrix<2,2,T>,
    b1 :Matrix<1,2,T>,
    a1 :Cell<Matrix<1,2,T>>,
    w2 :Matrix<2,1,T>,
    b2 :Matrix<1,1,T>,
    a2 :Cell<Matrix<1,1,T>>,
}

impl Model for Xor {
    fn rand() -> Self {
        Xor {
            a0: Matrix::<1,2,T>::default().into(),

            w1: Matrix::<2,2,T>::rand(),
            b1: Matrix::<1,2,T>::rand(),
        
            w2: Matrix::<2,1,T>::rand(),
            b2: Matrix::<1,1,T>::rand(),

            a1 : Matrix::<1,2,T>::default().into(),
            a2 : Matrix::<1,1,T>::default().into(),
        }
    }
    fn forward(&self, x1: T,x2: T) -> T {
        self.a0.set(Matrix::new([[x1,x2]]));
        self.a1.set(((self.a0.take() * self.w1.clone()) + self.b1.clone()).sigmoid());
        self.a2.set(((self.a1.take() * self.w2.clone()) + self.b2.clone()).sigmoid());
        self.a2.take().into()
    }

    fn finite_diff(&mut self) -> Self {
        let mut g = Self::rand();
        let c = self.cost();
        macro_rules! finite_diff_helper {
            ($field:ident) => {{
                let (m,n) = self.$field.get_nb_col_row();
                for y in 0..m {
                    for x in 0..n {
                        let t = self.$field.get((y,x)).clone();
                        self.$field.set((y,x), t+EPS);
                        g.$field.set((y,x), ((self.cost() - c) / EPS));
                        self.$field.set((y,x), t);
                    }
                }
            }};
        }
        finite_diff_helper!(w1);
        finite_diff_helper!(b1);
        finite_diff_helper!(w2);
        finite_diff_helper!(b2);
        g

    }
    fn learn_with(&mut self, g : Self) {
        macro_rules! learn_with_helper {
            ($field:ident) => {{
                self.$field -= g.$field * RATE;
            }};
        }
        learn_with_helper!(w1);
        learn_with_helper!(b1);
        learn_with_helper!(w2);
        learn_with_helper!(b2);
    }
}



fn run() {
    let mut x = Xor::rand();
    for _ in 0..100_000 {
        let g = x.finite_diff();
        x.learn_with(g);
        println!("cost {}", x.cost());
    }
    x.print_table(*TRAIN_PTR);

}


