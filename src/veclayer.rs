#![allow(unused, non_snake_case)]

use std::collections::LinkedList;

use crate::{
    layer::Layer,
    loss::{LossFunction, MeanSquaredError},
    matrix::Matrix, matrixchain::{HCons, HLeaf, HList, Nil, Push},
};

pub trait LayerLink<const INPUTS: usize, const LAST_OUTPUTS: usize> {
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<LAST_OUTPUTS, 1>;
    // fn backward(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<LAST_OUTPUTS, 1>>)
    fn backward(
        &mut self,
        aLprev: Matrix<INPUTS, 1>,
        loss: MeanSquaredError<LAST_OUTPUTS>,
        y: Matrix<LAST_OUTPUTS, 1>,
    ) -> 
    // (HCons<INPUTS, LAST_OUTPUTS, LAST_OUTPUTS, impl HList<LAST_OUTPUTS>>, 
        Matrix<INPUTS, 1>;
    // ); // (aL, wl, bl)

    fn apply_train(&mut self);

    fn printeq(&self, i: u32);
    fn len(&self) -> usize;
}

// LayerChain -> LayerChain<.. N> -> LayerChain<N, ..> -> ... -> OutputLayer

/// La derniere couche de neurone
/// OUTPUTS = LAST_OUTPUTS
#[derive(Clone)]
pub struct OutputLayer<const INPUTS: usize, const OUTPUTS: usize> {
    pub layer: Layer<INPUTS, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> LayerLink<INPUTS, OUTPUTS>
    for OutputLayer<INPUTS, OUTPUTS>
{
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.layer.forward(input)
    }

    // toute dernière couche
    fn backward(
        &mut self,
        aLprev: Matrix<INPUTS, 1>,
        loss: MeanSquaredError<OUTPUTS>,
        y: Matrix<OUTPUTS, 1>,
    ) -> // (HCons<INPUTS, OUTPUTS, OUTPUTS, impl HList<OUTPUTS>>, 
        Matrix<INPUTS, 1>
    // ) 
    {
        let aL = self.layer.forwardonce(aLprev.clone());
        let dCaL = loss.gradient(&aL, &y);
        return self.layer.backwardonce(aLprev, dCaL);
    }

    fn apply_train(&mut self) {
        self.layer.apply_the_train();
    }

    fn printeq(&self, i: u32) {
        println!("layer {i}");
        self.layer.printequation();
    }

    fn len(&self) -> usize {
        0
    }
}

pub struct LayerChain<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const LAST_OUTPUTS: usize,
    Next: LayerLink<OUTPUTS, LAST_OUTPUTS>,
> {
    pub layer: Layer<INPUTS, OUTPUTS>,
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
        let tmp = self.layer.forward(input);
        self.next.forward(tmp)
    }

    // couche strictement interne
    fn backward(
        &mut self,
        aLprev: Matrix<INPUTS, 1>,
        loss: MeanSquaredError<LAST_OUTPUTS>,
        y: Matrix<LAST_OUTPUTS, 1>,
    ) -> 
    // (HCons<INPUTS, OUTPUTS, LAST_OUTPUTS, impl HList<LAST_OUTPUTS>>, 
        Matrix<INPUTS, 1>
    // ) 
    {
        // println!("yo");
        // Propagation avant pour la couche courante
        let aL = self.layer.forwardonce(aLprev.clone());

        // Récupération du gradient depuis la couche suivante
        let (dCaL) = self.next.backward(aL, loss, y);
        return self.layer.backwardonce(aLprev, dCaL);
    }

    fn apply_train(&mut self) {
        self.layer.apply_the_train();
        self.next.apply_train();
    }

    fn printeq(&self, i: u32) {
        println!("layer {i}");
        self.layer.printequation();
        self.next.printeq(i + 1);
    }

    fn len(&self) -> usize {
        1 + self.next.len()
    }
}
