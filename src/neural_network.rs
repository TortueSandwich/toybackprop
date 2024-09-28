use crate::{matrix::Matrix, veclayer::LayerLink};


pub struct NeuralNetwork<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS, OUTPUTS>> {
    pub network: H,
}

impl<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS,OUTPUTS>> NeuralNetwork<INPUTS,OUTPUTS, H> {
    pub fn new(n: H) -> Self {
        NeuralNetwork {
            network: n
        }
    }

    pub fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.network.forward(input) 
    }
}



// vec Layer<2,3> Layer<3,2>
// vec Layer<A,B> Layer<B,C> Layer<C,D> ...
// reseau qui prend A entrées et D outputs

// z2 = w2*a1+b2
// 3x1 = 3x2*2x1 + 3x1

// z3 = w3*a2+b3
// 2x1 = 2x3*3x1 +2x1