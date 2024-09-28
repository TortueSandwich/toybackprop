#![allow(unused)]

use crate::{layer::Layer, matrix::Matrix};

pub trait LayerLink<const INPUTS: usize, const LAST_OUTPUTS: usize> {
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<LAST_OUTPUTS, 1>;
}

/// La derniere couche de neurone
/// OUTPUTS = LAST_OUTPUTS
pub struct OutputLayer<const INPUTS: usize, const OUTPUTS: usize> {
    pub layer: Layer<INPUTS, OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> LayerLink<INPUTS, OUTPUTS>
    for OutputLayer<INPUTS, OUTPUTS>
{
    fn forward(&self, input: Matrix<INPUTS, 1>) -> Matrix<OUTPUTS, 1> {
        self.layer.forward(input)
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
}

#[macro_export]
macro_rules! create_network {
    // Cas de base : dernière couche avec OutputLayer
    ($first:expr, $second:expr) => {
        $crate::veclayer::OutputLayer::<$first, $second> {
            layer: $crate::layer::Layer::<$first, $second>::rand(),
        }
    };

    // Cas récursif : capture les deux premiers éléments et continue avec le reste
    ($first:expr, $second:expr, $($rest:expr),+) => {{
        $crate::veclayer::LayerChain::<$first, $second, { create_network!(@last $($rest),+) }, _> {
            layer: $crate::layer::Layer::<$first, $second>::rand(),
            next: create_network!($second, $($rest),+),
        }
    }};

    // Capture le dernier argument de la liste
    (@last $last:expr) => {
        $last
    };

    // Cas récursif avec plusieurs arguments pour capturer la dernière valeur
    (@last $first:expr, $($rest:expr),+) => {
        create_network!(@last $($rest),+)
    };
}
