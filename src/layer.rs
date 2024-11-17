use std::marker::PhantomData;
use std::process::Output;

use num_traits::zero;

use crate::activation::ActivationFunction;
use crate::layerbuffer::LayerBuffer;
use crate::{activation::Sigmoid, matrix::Matrix};



/// NEURONS=Nb of Output
#[derive(Clone)]
pub struct Layer<const INPUTS: usize, const NEURONS: usize> {
    pub weights: Matrix<NEURONS, INPUTS>, // wl
    pub biases: Matrix<NEURONS, 1>, // bl

    lbuff : Option<LayerBuffer<INPUTS, NEURONS>>,
}

impl<const INPUTS: usize, const NEURONS: usize> Layer<INPUTS, NEURONS> {
    pub fn rand() -> Self {
        let weights = Matrix::rand(-1.0..1.0) * 0.0+1.0;
        let biases = Matrix::ones();
        Layer { weights, biases, lbuff : None}
    }

    /// z = w * a + b
    /// 3x1 = 3x2*2x1 + 3x1
    pub fn forward(&self, a: Matrix<INPUTS, 1>) -> Matrix<NEURONS, 1> {
        let weighted_sum: Matrix<NEURONS, 1> = self.weights.clone().multiply(a);
        let t: Matrix<NEURONS, 1> = weighted_sum + self.biases.clone();
        let s = Sigmoid;
        t.mapv(|x| s.activate(x))
        // t.mapv(|x| x.max(0.0)) // ReLU activation
    }

    // pub fn backward(&self, gradiantcost : Matrix<NEURONS, 1>) {
    //     //  gradiantcost = 2(ypred-y)

    //     let delta = gradiantcost *
    // }

    fn activate(&self, x: f64) -> f64 {
        // x.max(0.0)
        Sigmoid.activate(x)
    }

    fn activationderivative(&self, x: f64) -> f64 {
        // 1.0
        Sigmoid.derivative(x)
    }

    pub fn forwardonce(&self, a: Matrix<INPUTS, 1>) -> Matrix<NEURONS, 1> {
        self.weights.clone().multiply(a) + self.biases.clone()
    }

    pub fn apply_the_train(&mut self) {
        println!("ok j'applique ça :");
        let (w,b) = self.lbuff.take().unwrap().get_mats();
        println!("w : {w},b : {b}");
        let lr:f64 = 1.0;//0.01;
        self.weights = self.weights.clone() + w*lr;
        self.biases = self.biases.clone() + b*lr;
    }

    // couche quelconque
    #[allow(non_snake_case)]
    pub fn backwardonce(
        &mut self,
        aLprev: Matrix<INPUTS, 1>,
        dCaL: Matrix<NEURONS, 1>,
    ) ->  Matrix<INPUTS, 1> {

        if self.lbuff.is_none() {
            self.lbuff = Some(LayerBuffer::default());
        }

        let zL = self.weights.clone().multiply(aLprev.clone()) + self.biases.clone(); // ok
        let aL = zL.mapv(|x| self.activate(x)); // ok
        let daLzL = zL.mapv(|x| self.activationderivative(x)); // ok

        // let dCzL =   daLzL.multiply(dCaL);
        let dzLwL = aLprev.clone();
        
        let mut wgrad = Matrix::<NEURONS, INPUTS>::zeros();
        for k in 0..INPUTS {
            for j in 0..NEURONS {
                let alk = aL.data[j][0];
                let derivsig=self.activationderivative(zL.data[j][0]);
                let caj = dCaL.data[j][0];
                wgrad.data[j][k] = alk*derivsig*caj;
            }
        }

        // let dCwL: Matrix<NEURONS, INPUTS> = dzLwL.multiply(dCzL).transpose();
        
        let dzLbL = 1.0;
        let mut bgrad = Matrix::<NEURONS, 1>::zeros();
        for k in 0..NEURONS {
            bgrad.data[k][0] = dzLbL*self.activationderivative(zL.data[k][0])*dCaL.data[k][0]
        }
        
        // let dCbL: Matrix<NEURONS, 1> = dzLbL*dCzL;

        // let dzLaLprev = self.weights.clone();
        // let dCaLprev = dzLaLprev * dCzL;

        // let bgrad = Matrix::<NEURONS, 1>::zeros();    


        
        self.lbuff.as_mut().expect("wrf").add_weight_grad(wgrad, bgrad);

        // self.weights = self.weights.clone() - wgrad.clone()*0.005;//dCwL * 0.01;
        // self.biases = self.biases.clone() - bgrad.clone()*0.005;//dCbL * 0.01;


        let mut galprev = Matrix::<INPUTS, 1>::zeros();
        for k in 0..INPUTS {
            let mut acc = 0.0;
            for j in 0..NEURONS {
                let mut wi = 0.0;
                for k2 in 0..INPUTS {
                    wi += aLprev.clone().data[k2][0];
                }
                acc += wi*self.activationderivative(zL.data[j][0])*dCaL.data[j][0]
            }
            galprev.data[k][0] = acc;
        }

        return galprev;

        // return dCaLprev;
    }

    // #[allow(non_snake_case)]
    // pub fn backward(&mut self, X: &Vec<Matrix<1, INPUTS>> ,  Y: &Vec<Matrix<NEURONS, 1>>)
    // // -> gradiant
    // {
    //     // NN(xi) -> yi

    //     // POUR L'instant ce n'est qu'une descente de gradiant !!!!
    //     // IL FAUT RETOURNER LE GRADIANT ET NE PAS LE MODIF les parametres Là MTN
    //     let n = X.len();

    //     let mut gradient_w: Matrix<NEURONS, INPUTS> = Matrix::zeros();
    //     let mut gradient_b: Matrix<NEURONS, 1> = Matrix::zeros();

    //     for i in 0..n {
    //         let x: Matrix<INPUTS, 1> = X[i].transpose().clone();
    //         let y_real: Matrix<NEURONS, 1> = Y[i].clone();
    //         let y_pred: Matrix<NEURONS, 1> = self.forward(x.clone());

    //         let error: Matrix<NEURONS, 1> = y_pred - y_real; // MSE'(y_pred) (plus tard le polymorphismezzz)

    //         gradient_w = gradient_w + error.clone().multiply(x.clone().transpose());

    //         gradient_b = gradient_b + error.clone();
    //     }

    //     gradient_w = gradient_w * (1.0/ n as f64);
    //     gradient_b = gradient_b * (1.0/ n as f64);

    //     // TODO d'abord calculer gradiant entier avant demodifier
    //     self.weights = self.weights.clone() - gradient_w.clone() * 0.01;
    //     self.biases = self.biases.clone() - gradient_b.clone() * 0.01;

    //     // println!("nabla w = {gradient_w}");
    //     // println!("nabla b = {gradient_b}");

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
