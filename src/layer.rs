use crate::{activation::Sigmoid, matrix::Matrix};
use crate::activation::ActivationFunction;

/// NEURONS=Nb of Output
#[derive(Clone)]
pub struct Layer<const INPUTS: usize, const NEURONS: usize> {
    pub weights: Matrix<NEURONS, INPUTS>,
    pub biases: Matrix<NEURONS, 1>,
}

impl<const INPUTS: usize, const NEURONS: usize> Layer<INPUTS, NEURONS> {
    pub fn rand() -> Self {
        let weights = Matrix::rand(-1.0..1.0)*0.0+2.0 ;
        let biases = Matrix::zeros();
        Layer { weights, biases }
    }

    /// z = w * a + b
    /// 3x1 = 3x2*2x1 + 3x1
    pub fn forward(&self, a: Matrix<INPUTS, 1>) -> Matrix<NEURONS, 1> {
        let weighted_sum: Matrix<NEURONS, 1> = self.weights.clone().multiply(a);
        let t: Matrix<NEURONS, 1> =  weighted_sum + self.biases.clone();
        let s = Sigmoid;
        // t.mapv(|x| s.activate(x))
        t.mapv(|x| x.max(0.0)) // ReLU activation
    }

    #[allow(non_snake_case)]
    pub fn backward(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<NEURONS, 1>>) {
        let n = X.len();

        let mut gradient_w: Matrix<NEURONS, INPUTS> = Matrix::zeros(); 
        let mut gradient_b: Matrix<NEURONS, 1> = Matrix::zeros();
    
        for i in 0..n {
            let x: Matrix<INPUTS, 1> = X[i].transpose().clone();
            let y_real: Matrix<NEURONS, 1> = Y[i].clone(); 
            let y_pred: Matrix<NEURONS, 1> = self.forward(x.clone());

            let error: Matrix<NEURONS, 1> = y_pred - y_real; 
    
            gradient_w = gradient_w + error.clone().multiply(x.clone().transpose());

            gradient_b = gradient_b + error.clone();
        }
        gradient_w = gradient_w * (1.0/ n as f64);
        gradient_b = gradient_b * (1.0/ n as f64);

        self.weights = self.weights.clone() - gradient_w.clone() * 0.01;
        self.biases = self.biases.clone() - gradient_b.clone() * 0.01;

        // println!("nabla w = {gradient_w}");
        // println!("nabla b = {gradient_b}");
        
    }


    // pub fn backward(&mut self, input: Matrix<INPUTS, 1>, output_gradient: Matrix<NEURONS, 1>, learning_rate: f64) -> Matrix<INPUTS, 1> {
    //     let weight_gradient = output_gradient.clone().multiply(input.transpose());
    //     self.weights = self.weights.clone() - weight_gradient * learning_rate;
    //     self.biases = self.biases.clone() - output_gradient.clone() * learning_rate;
    //     let input_gradient = self.weights.transpose().multiply(output_gradient);
    //     input_gradient
    // }

    pub fn printequation(&self) {
        for n in 0..NEURONS {
            print!("n{n} : ");
            for i in 0..INPUTS {
                print!("{:.5}*x{i} + ", self.weights.data[n][i])
            }
            println!("{:.5}", self.biases.data[n][0]);
        }
    }
}
