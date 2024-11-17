#![allow(dead_code, unused_variables)]

use std::ops::Div;

use rand::{thread_rng, Rng};

pub trait ActivationFunction : Clone {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64; // For backpropagation
}

/// Rectified Linear Units
#[derive(Clone)]
struct ReLU;
impl ActivationFunction for ReLU {
    fn activate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            0.0
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Gaussian Error Linear Units
/// GELU(x) = xP(X<=x) = (x/2) * (1+erf(x/sqrt(2)))
/// Where X ~ N(0,1)
#[derive(Clone)]
struct GELU;
impl ActivationFunction for GELU {
    fn activate(&self, x: f64) -> f64 {
        todo!("choose approx")
    }
    fn derivative(&self, x: f64) -> f64 {
        todo!("tf")
    }
}

#[derive(Clone)]
pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0.div(1.0 + (-x).exp())
    }
    fn derivative(&self, x: f64) -> f64 {
        self.activate(x) * (1.0 - self.activate(x))
    }
}

#[derive(Clone)]
struct Tanh;
impl ActivationFunction for Tanh {
    fn activate(&self, x: f64) -> f64 {
        (x.exp() - (-x).exp()).div(x.exp() + (-x).exp())
    }
    fn derivative(&self, x: f64) -> f64 {
        1.0 - Tanh.activate(x).powi(2)
    }
}

/// Leaky ReLU
#[derive(Clone)]
struct LeakyReLU;
impl ActivationFunction for LeakyReLU {
    fn activate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            0.1 * x
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            0.1
        }
    }
}

#[derive(Clone)]
struct GLU;
// todo!();

#[derive(Clone)]
struct Swish {
    beta: f64, // can be LEARNABLE PARAMETER
}
impl ActivationFunction for Swish {
    fn activate(&self, x: f64) -> f64 {
        x * Sigmoid.activate(self.beta * x)
    }
    fn derivative(&self, x: f64) -> f64 {
        self.beta * self.activate(x)
            + Sigmoid.activate(self.beta * x) * (1.0 - self.beta * self.activate(x))
    }
}

#[derive(Clone)]
struct SoftPlus;
impl ActivationFunction for SoftPlus {
    fn activate(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    fn derivative(&self, x: f64) -> f64 {
        Sigmoid.activate(x)
    }
}

/// Exponential Linear Unit
#[derive(Clone)]
struct ELU {
    alpha: f64, // > 0 todo
}
impl ActivationFunction for ELU {
    fn activate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            self.alpha * (x.exp() - 1.0)
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            self.alpha * x.exp()
        }
    }
}

/// Scaled Exponential Linear Unit
#[derive(Clone)]
struct SELU {
    alpha: f64, // > 0 todo
    lambda: f64,
}
impl Default for SELU {
    fn default() -> Self {
        SELU {
            alpha: 1.6733,
            lambda: 1.0507,
        }
    }
}
impl ActivationFunction for SELU {
    fn activate(&self, x: f64) -> f64 {
        self.lambda
            * if x >= 0.0 {
                x
            } else {
                self.alpha * (x.exp() - 1.0)
            }
    }
    fn derivative(&self, x: f64) -> f64 {
        self.lambda * if x >= 0.0 { 1.0 } else { self.alpha * x.exp() }
    }
}

#[derive(Clone)]
struct SiLU;
impl ActivationFunction for SiLU {
    fn activate(&self, x: f64) -> f64 {
        x * Sigmoid.activate(x)
    }
    fn derivative(&self, x: f64) -> f64 {
        Sigmoid.activate(x) + x * Sigmoid.derivative(x)
    }
}

#[derive(Clone)]
struct Mish;
impl ActivationFunction for Mish {
    fn activate(&self, x: f64) -> f64 {
        x * Tanh.activate(SoftPlus.activate(x))
    }
    fn derivative(&self, x: f64) -> f64 {
        let omega =
            (3.0 * x).exp() + 4.0 * (2.0 * x).exp() + (6.0 + 4.0 * x) * x.exp() + 4.0 * (1.0 + x);
        let delta = 1.0 + (x.exp() + 1.0).powi(2);
        x.exp() * omega / delta.powi(2)
    }
}

/// Parametric Rectified Linear Unit,
#[derive(Clone)]
struct PReLU {
    parameter: f64, // 0< parameter <1
}
impl ActivationFunction for PReLU {
    fn activate(&self, x: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            self.parameter * x
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        } else {
            self.parameter
        }
    }
}

#[derive(Clone)]
struct ReLU6;
impl ActivationFunction for ReLU6 {
    fn activate(&self, x: f64) -> f64 {
        let x = ReLU.activate(x);
        if x >= 6.0 {
            6.0
        } else {
            x
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x >= 6.0 || x <= 0.0 {
            0.0
        } else {
            1.0
        }
    }
}

#[derive(Clone)]
struct HardSwish;
impl ActivationFunction for HardSwish {
    fn activate(&self, x: f64) -> f64 {
        (1.0 / 6.0) * x * ReLU6.activate(x + 3.0)
    }
    fn derivative(&self, x: f64) -> f64 {
        todo!()
    }
}

#[derive(Clone)]
struct Maxout;
// todo

/// Adaptive Richard's Curve Weighted Activation
/// wtf
///  Swish is a special case of ARiA, manifested at ARiA=f(x, 1, 0, 1, 1, β, 1)
#[derive(Clone)]
struct ARiA {
    a: f64, // lower asymptote
    k: f64, // upper asymptote   (a<k) ? todo
    b: f64, // exponential growth rate
    v: f64, // > 0 todo decides the direction of growth
    q: f64, // related to the initial value of function
    c: f64, // constant which typically is chosen as one
}
impl ActivationFunction for ARiA {
    fn activate(&self, x: f64) -> f64 {
        let num = self.k - self.a;
        let denomnotpow = self.c + self.q * (-self.b * x).exp();
        x * (self.a + num / denomnotpow.powf(1.0 / self.v))
    }

    fn derivative(&self, x: f64) -> f64 {
        todo!()
    }
}

/// Adaptive Richard's Curve Weighted Activation 2 (simplified)
/// wtf
#[derive(Clone)]
struct ARiA2 {
    alpha: f64, //contraint? todo
    beta: f64,
}
impl ActivationFunction for ARiA2 {
    fn activate(&self, x: f64) -> f64 {
        let sigma = (1.0 + (-self.beta * x).exp()) * self.alpha;
        x * sigma
    }
    fn derivative(&self, x: f64) -> f64 {
        let tmp = ARiA2 {
            alpha: self.alpha + 1.0,
            beta: self.beta,
        };
        let t = tmp.activate(x);
        self.activate(x) + (-self.beta * x).exp() * t * x * self.alpha * self.beta
    }
}

#[derive(Clone)]
struct ShiftedSoftplus;
impl ActivationFunction for ShiftedSoftplus {
    fn activate(&self, x: f64) -> f64 {
        (0.5 * (x.exp() + 1.0)).ln()
    }
    fn derivative(&self, x: f64) -> f64 {
        0.5 * Sigmoid.activate(x)
    }
}

#[derive(Clone)]
struct Softsign;
impl ActivationFunction for Softsign {
    fn activate(&self, x: f64) -> f64 {
        x.div(x.abs() + 1.0)
    }

    fn derivative(&self, x: f64) -> f64 {
        1.0.div((x.abs() + 1.0).powi(2))
    }
}

#[derive(Clone)]
struct TanhExp;
impl ActivationFunction for TanhExp {
    fn activate(&self, x: f64) -> f64 {
        x * Tanh.activate(x.exp())
    }
    fn derivative(&self, x: f64) -> f64 {
        Tanh.activate(x) + x * x.exp() - x * x.exp() * Tanh.activate(x.exp()).powi(2)
    }
}

#[derive(Clone)]
struct ModReLU {
    bias: f64,
}
impl ActivationFunction for ModReLU {
    // should be a  complex nb
    fn activate(&self, x: f64) -> f64 {
        if x.abs() + self.bias >= 0.0 {
            (x.abs() + self.bias) * (x / x.abs())
        } else {
            0.0
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        todo!()
    }
}

#[derive(Clone)]
struct HardSigmoid;
impl ActivationFunction for HardSigmoid {
    fn activate(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x > 1.0 {
            1.0
        } else {
            0.5 * (x + 1.0)
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            0.0
        } else {
            1.0
        }
    }
}

/// Randomized Leaky Rectified Linear Units
#[derive(Clone)]
struct RReLU {
    l: f64,
    u: f64, // l<u && l,u \in [0,1) todo
}
impl Default for RReLU {
    fn default() -> Self {
        RReLU { l: 3.0, u: 8.0 }
    }
}
impl ActivationFunction for RReLU {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            todo!("le hasard n'est fait qu'one fois");
            //thread_rng().gen_range(self.l..self.u) * x
            thread_rng().gen_range((1.0 / self.l)..(1.0 / self.u)) * x
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            todo!("le hasard n'est fait qu'one fois");
            //thread_rng().gen_range(self.l..self.u) * x
            thread_rng().gen_range((1.0 / self.l)..(1.0 / self.u))
        }
    }
}

#[derive(Clone)]
struct Serf;
impl ActivationFunction for Serf {
    fn activate(&self, x: f64) -> f64 {
        todo!("missing erf")
        // x*(1.0+x.exp()).ln().erf()
    }
    fn derivative(&self, x: f64) -> f64 {
        todo!()
    }
}

#[derive(Clone)]
struct DELU {
    n: f64, // reel ? LEARNABLE PARAMETER
}
impl ActivationFunction for DELU {
    fn activate(&self, x: f64) -> f64 {
        if x <= 0.0 {
            SiLU.activate(x)
        } else {
            (self.n + 0.5) * x + ((-x).exp() - 1.0).abs()
        }
    }
    fn derivative(&self, x: f64) -> f64 {
        todo!()
    }
}

/// S-shaped Rectified Linear Unit
#[derive(Clone)]
struct SReLU {
    // LEARNABLE PARAMETER
    tl: f64,
    tr: f64, // tl < tr
    a: f64, // > 0 ?
}
impl ActivationFunction for SReLU {
    fn activate(&self, x: f64) -> f64 {
        if x >= self.tr { self.tr + self.a*(x-self.tr) }
        else if x <= self.tl {self.tl + self.a*(x-self.tl)}
        else {x}
    }

    fn derivative(&self, x: f64) -> f64 {
        if x >= self.tr || x <= self.tl {self.a}
        else {1.0}
    }
}

/// Parameterized Exponential Linear Units
#[derive(Clone)]
struct PELU ;
// todo

/// Adaptive Parametric activation 
#[derive(Clone)]
struct APA;
// todo

/// Margin Rectified Linear Unit
#[derive(Clone)]
struct MarginReLU;
//todo

/// Cosine Linear Unit
#[derive(Clone)]
struct CosLU;
//todo

/// Scaled Exponentially-Regularized Linear Unit
#[derive(Clone)]
struct SERLU;
//todo

/// Shifted Rectified Linear Unit
#[derive(Clone)]
struct ShiLU;
//todo

/// Collapsing Linear Unit
#[derive(Clone)]
struct CoLU;
// todo

/// Continuously Differentiable Exponential Linear Units
#[derive(Clone)]
struct CELU;
//todo


/// Rectified Linear Unit N
#[derive(Clone)]
struct ReLUN {
    n:f64, //trainable parameter
}
//todo

// Gumbel Cross Entropy
#[derive(Clone)]
struct GCE;
//todo

/// ScaledSoftSign
#[derive(Clone)]
struct ScaledSoftSign;
//todo

#[derive(Clone)]
struct Smish;
//todo

/// Exponential Linear Squashing Activation
#[derive(Clone)]
struct ELiSH;
//todo

#[derive(Clone)]
struct HardELiSH;
// todo


/// Optimizer Activation Function
#[derive(Clone)]
struct NIPUNA;
//todo

#[derive(Clone)]
struct StarReLU;
//todo


/// Lecun's Tanh
#[derive(Clone)]
struct LecunTanh;
//todo

#[derive(Clone)]
struct Hardtanh ;
// todo
