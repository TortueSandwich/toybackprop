use crate::{loss::{LossFunction, MeanSquaredError}, matrix::Matrix, veclayer::LayerLink};

#[derive(Clone)]
pub struct NeuralNetwork<
    const INPUTS: usize,
    const OUTPUTS: usize,
    H: LayerLink<INPUTS, OUTPUTS>,
    L: LossFunction<OUTPUTS>,
> {
    pub network: H,
    pub loss_function: L,
}

impl<
        const INPUTS: usize,
        const OUTPUTS: usize,
        H: LayerLink<INPUTS, OUTPUTS>,
        L: LossFunction<OUTPUTS>,
    > NeuralNetwork<INPUTS, OUTPUTS, H, L>
{
    pub fn new(network: H, loss_function: L) -> Self {
        NeuralNetwork {
            network,
            loss_function,
        }
    }

    // forward pass
    pub fn predict(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.network.forward(input)
    }

    pub fn get_gradiant(&mut self, input: Matrix<INPUTS, 1>, target: Matrix<OUTPUTS, 1>) {
        let predicted = self.network.forward(input.clone());
        self.loss_function.gradient(&predicted, &target);
    }

    fn apply_train_gradiants(&mut self) {
        self.network.apply_train();
    }

    pub fn entraine(&mut self, X: &Vec<Matrix<1, INPUTS>>, Y: &Vec<Matrix<OUTPUTS, 1>>) {
        let n = X.len();
        // let mut error_pondere = Matrix::<OUTPUTS, 1>::zeros();

        // for i in 0..n {
        //     let x: Matrix<INPUTS, 1> = X[i].transpose().clone();
        //     let y_real: Matrix<OUTPUTS, 1> = Y[i].clone(); 
        //     let y_pred: Matrix<OUTPUTS, 1> = self.predict(x.clone());

        //     let error: Matrix<OUTPUTS, 1> = y_pred - y_real; // MSE'(y_pred) (plus tard le polymorphismezzz)
        //     error_pondere = error_pondere + error;
        // }

        // error_pondere = error_pondere*(1.0/n as f64);

        

        // let gradiant_global = 
        for k in 0..n {
            self.network.backward(X[k].transpose(), MeanSquaredError{} , Y[k].clone());
        }
        self.apply_train_gradiants();
        // self.network.backward(X, Y);
        // todo faire modification des paramtres

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
