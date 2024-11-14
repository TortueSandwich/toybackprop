use crate::{loss::LossFunction, matrix::Matrix, veclayer::LayerLink};

#[derive(Clone)]
pub struct NeuralNetwork<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS, OUTPUTS>, L: LossFunction<OUTPUTS>> {
    pub network: H,
    pub loss_function: L,
}

impl<const INPUTS: usize, const OUTPUTS: usize, H: LayerLink<INPUTS,OUTPUTS>, L: LossFunction<OUTPUTS>> NeuralNetwork<INPUTS,OUTPUTS, H, L> {
    pub fn new(network: H, loss_function: L) -> Self {
        NeuralNetwork {
            network,
            loss_function,
        }
    }

    // forward pass
    pub fn  predict(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.network.forward(input) 
    }

    pub fn get_gradiant(&mut self, input: Matrix<INPUTS, 1>, target: Matrix<OUTPUTS, 1>) {
        let predicted = self.network.forward(input.clone());
        self.loss_function.gradient(&predicted, &target);

    }

    pub fn ducoupppp(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<OUTPUTS, 1>>) {
        self.network.backward(X, Y);
    }

    pub fn fit(&mut self, input: Matrix<INPUTS, 1>, target: Matrix<OUTPUTS, 1>) {
        let predicted = self.network.forward(input.clone());
        // let loss = self.loss_function.compute(&predicted, &target);
        
        let output_gradient = self.loss_function.gradient(&predicted, &target);
        // println!("predicted : {}   ===   {}",predicted.data[0][0], target.data[0][0]);
        // println!("outputgrad: gard {output_gradient}");
        self.network.backward(input, output_gradient);
    }
    
    pub fn printequations(&self) {
        self.network.printeq(0);
    }
}



// vec Layer<2,3> Layer<3,2>
// vec Layer<A,B> Layer<B,C> Layer<C,D> ...
// reseau qui prend A entrées et D outputs

// z2 = w2*a1+b2
// 3x1 = 3x2*2x1 + 3x1

// z3 = w3*a2+b3
// 2x1 = 2x3*3x1 +2x1