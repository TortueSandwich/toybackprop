#![allow(unused, non_snake_case)]

use std::collections::LinkedList;

use crate::{
    activation::ActivationFunction, layer::DenseLayer, loss::FonctionLoss, matrix::Matrix, matrixchain::{HCons, HLeaf, HList, Nil, Push}
};

pub trait LayerLink<const INPUTS: usize, const LAST_OUTPUTS: usize> {
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<LAST_OUTPUTS, 1>;
    fn backward_batch<const BATCH_SIZE: usize>(
        &mut self,
        previous_activation: [Matrix<INPUTS, 1>; BATCH_SIZE], // aL
        loss_function: FonctionLoss<LAST_OUTPUTS>, // loss 
        y: [Matrix<LAST_OUTPUTS, 1>; BATCH_SIZE],  // target_output
    ) -> [Matrix<INPUTS, 1>;BATCH_SIZE];

    fn backward(
        &mut self,
        previous_activation: Matrix<INPUTS, 1>, // aL
        loss_function: FonctionLoss<LAST_OUTPUTS>, // loss 
        y: Matrix<LAST_OUTPUTS, 1>,  // target_output
    ) -> Matrix<INPUTS, 1>;

    fn apply_train(&mut self, learning_rate : f64);

    fn printeq(&self, i: u32);
    fn len(&self) -> usize;
    fn initialiser_avec_poids_he(&mut self);
}

// LayerChain -> LayerChain<.. N> -> LayerChain<N, ..> -> ... -> OutputLayer

/// La derniere couche de neurone
/// OUTPUTS = LAST_OUTPUTS
#[derive(Clone)]
pub struct OutputLayer<const INPUTS: usize, const OUTPUTS: usize> {
    pub layer: DenseLayer<INPUTS, OUTPUTS>,
}

// toute dernière couche
impl<const INPUTS: usize, const OUTPUTS: usize> LayerLink<INPUTS, OUTPUTS>
    for OutputLayer<INPUTS, OUTPUTS>
{
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.layer.forward_pass(input)
    }

    fn backward_batch<const BATCH_SIZE: usize>(
        &mut self,
        previous_activation: [Matrix<INPUTS, 1>; BATCH_SIZE], // aL
        loss_function: FonctionLoss<OUTPUTS>, // loss 
        y: [Matrix<OUTPUTS, 1>; BATCH_SIZE],  // target_output
    ) -> [Matrix<INPUTS, 1>;BATCH_SIZE] {
        let activations = previous_activation.clone().map( |x|self.layer.forward_pass(x.clone()));
        // let mut accloss = 0.0;
        let mut accgrad = std::array::from_fn(|_| Matrix::zeros());
        for i in 0..BATCH_SIZE {
            // accloss += loss_function.compute(&activations[i], &y[i]);
            accgrad[i] = loss_function.compute_grad(&activations[i], &y[i]);
        }
        
        return self.layer.backpropagate_batch(previous_activation, accgrad);
    }

    fn backward(
        &mut self,
        previous_activation: Matrix<INPUTS, 1>, // aL
        loss_function: FonctionLoss<OUTPUTS>, // loss 
        y: Matrix<OUTPUTS, 1>,  // target_output
    ) -> Matrix<INPUTS, 1> {
        let activation = self.layer.forward_pass(previous_activation.clone());
        let gradient_wrt_output =  loss_function.compute_grad(&activation, &y);
        return self.layer.backpropagate(previous_activation, gradient_wrt_output);

    }
    

    fn apply_train(&mut self, learning_rate : f64) {
        self.layer.apply_gradients(learning_rate);
    }

    fn printeq(&self, i: u32) {
        println!("layer {i}");
        self.layer.printequation();
    }

    fn len(&self) -> usize {
        0
    }

    fn initialiser_avec_poids_he(&mut self) {
        self.layer.initialiser_poids_he(INPUTS);
    }
}

pub struct LayerChain<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const LAST_OUTPUTS: usize,
    Next: LayerLink<OUTPUTS, LAST_OUTPUTS>,
> {
    pub layer: DenseLayer<INPUTS, OUTPUTS>,
    pub next: Next,
}

impl<
        const INPUTS: usize,
        const OUTPUTS: usize,
        const LAST_OUTPUTS: usize,
        Next: LayerLink<OUTPUTS, LAST_OUTPUTS>,
    > LayerLink<INPUTS, LAST_OUTPUTS> for LayerChain<INPUTS, OUTPUTS, LAST_OUTPUTS, Next>
{
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<LAST_OUTPUTS, 1> {
        let prev_layer_output = self.layer.forward_pass(input);
        self.next.forward(prev_layer_output)
    }

    // couche strictement interne
    fn backward(
        &mut self,
        aLprev: Matrix<INPUTS, 1>, // aL
        loss: FonctionLoss<LAST_OUTPUTS>,
        y: Matrix<LAST_OUTPUTS, 1>,
    ) -> Matrix<INPUTS, 1> {
        // Propagation avant pour la couche courante
        let aL = self.layer.forward_pass(aLprev.clone());

        // Récupération du gradient depuis la couche suivante
        let (dCaL) = self.next.backward(aL, loss, y);
        return self.layer.backpropagate(aLprev, dCaL);
    }

    fn backward_batch<const BATCH_SIZE: usize>(
            &mut self,
            previous_activation: [Matrix<INPUTS, 1>; BATCH_SIZE], // aL
            loss_function: FonctionLoss<LAST_OUTPUTS>, // loss 
            y: [Matrix<LAST_OUTPUTS, 1>; BATCH_SIZE],  // target_output
        ) -> [Matrix<INPUTS, 1>;BATCH_SIZE] {
        // Propagation avant pour la couche courante
        let aL = previous_activation.clone().map(|x| self.layer.forward_pass(x));

        // Récupération du gradient depuis la couche suivante
        let (dCaL) = self.next.backward_batch(aL, loss_function, y);
        return self.layer.backpropagate_batch(previous_activation, dCaL);
    }

    fn apply_train(&mut self, learning_rate : f64) {
        self.layer.apply_gradients(learning_rate);
        self.next.apply_train(learning_rate);
    }

    fn printeq(&self, i: u32) {
        println!("layer {i}");
        self.layer.printequation();
        self.next.printeq(i + 1);
    }

    fn len(&self) -> usize {
        1 + self.next.len()
    }

    fn initialiser_avec_poids_he(&mut self) {
        self.layer.initialiser_poids_he(INPUTS);
        self.next.initialiser_avec_poids_he();
    }
}
