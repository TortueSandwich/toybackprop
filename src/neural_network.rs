use std::marker::PhantomData;

use crate::{
    activation::{ActivationFunction, FonctionActivation},
    loss::FonctionLoss,
    matrix::Matrix,
    veclayer::LayerLink,
};

#[derive(Clone)]
pub struct NeuralNetwork<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS, OUTPUTS>> {
    pub network: H,
    pub loss_function: FonctionLoss<OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS, OUTPUTS>>
    NeuralNetwork<INPUTS, OUTPUTS, H>
{
    pub fn new(network: H, loss_function: FonctionLoss<OUTPUTS>) -> Self {
        NeuralNetwork {
            network,
            loss_function,
        }
    }

    // forward pass
    pub fn predict(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.network.forward(input)
    }

    #[allow(unused)]
    pub fn get_gradiant(&mut self, input: Matrix<INPUTS, 1>, target: Matrix<OUTPUTS, 1>) {
        let predicted = self.network.forward(input.clone());
        self.loss_function.compute(&predicted, &target);
    }

    fn apply_train_gradiants(&mut self, learning_rate: f64) {
        self.network.apply_train(learning_rate);
    }

    #[allow(non_snake_case, dead_code)]
    pub fn entraine<const N: usize>(
        &mut self,
        X: &[Matrix<1, INPUTS>; N],
        Y: &[Matrix<OUTPUTS, 1>; N],
        learning_rate: f64,
    ) {
        for k in 0..N {
            let input = X[k].transpose();
            let y = Y[k].clone();
            self.network.backward(input, self.loss_function.clone(), y);
        }
        self.apply_train_gradiants(learning_rate);
    }

    /// N = Size of data
    #[allow(non_snake_case)]
    pub fn entraine_full_batch<const N: usize>(
        &mut self,
        X: &[Matrix<1, INPUTS>; N],
        Y: &[Matrix<OUTPUTS, 1>; N],
        learning_rate: f64,
    ) {
        let X = X.clone().map(|x| x.transpose());
        self.network.backward_batch(X, self.loss_function.clone(), Y.clone());
        self.apply_train_gradiants(learning_rate);
    }

    #[allow(non_snake_case, dead_code)]
    pub fn entraine_mini_batch<const N: usize, const BATCH_SIZE: usize>(
        &mut self,
        X: &[Matrix<1, INPUTS>; N],
        Y: &[Matrix<OUTPUTS, 1>; N],
        learning_rate: f64,
    ) {
        let mut start = 0;
        while start < N {
            let end = usize::min(start + BATCH_SIZE, N);

            let mut X_batch: [Matrix<INPUTS, 1>; BATCH_SIZE] = std::array::from_fn(|_| Matrix::zeros());
            let mut Y_batch: [Matrix<OUTPUTS, 1>; BATCH_SIZE] = std::array::from_fn(|_| Matrix::<OUTPUTS, 1>::zeros());
            
            for (i, (x, y)) in X[start..end].iter().zip(Y[start..end].iter()).enumerate() {
                if i < BATCH_SIZE {
                    X_batch[i] = x.clone().transpose();
                    Y_batch[i] = y.clone();
                }
            }

            self.network.backward_batch(X_batch, self.loss_function.clone(), Y_batch);

            self.apply_train_gradiants(learning_rate);
            start += BATCH_SIZE;
        }
    }

    pub fn printequations(&self) {
        self.network.printeq(0);
    }

    #[allow(unused)]
    pub fn count_layer(&self) -> usize {
        self.network.len()
    }

    pub fn init_he(&mut self) {
        self.network.initialiser_avec_poids_he();
    }
}
