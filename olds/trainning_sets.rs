#![allow(unused)]

type Expected = f64;
type TrainEntry<const N: usize> = [f64; N];

pub struct DataLine<const N: usize> {
    pub entry: TrainEntry<N>,
    pub expected: Expected,
}

pub struct TrainDataSet<const N: usize> {
    pub data: &'static [DataLine<N>],
}

macro_rules! dataset {
    ( $( [$($input:expr),*] => $expected:expr ),+ $(,)? ) => {
        TrainDataSet {
            data: &[ $( DataLine { entry: [$($input),*], expected: $expected } ),+]
        }
    };
}

pub struct TrainDataSetIterator<'a, const N: usize> {
    dataset: &'a TrainDataSet<N>,
    index: usize,
}

impl<'a, const N: usize> Iterator for TrainDataSetIterator<'a, N> {
    type Item = &'a DataLine<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dataset.data.len() {
            let item = &self.dataset.data[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl<const N: usize> TrainDataSet<N> {
    pub fn iter(&self) -> TrainDataSetIterator<N> {
        TrainDataSetIterator {
            dataset: self,
            index: 0,
        }
    }
}

pub static LINEAR_TRAIN: TrainDataSet<1> = dataset![
    [0.] => 0.,
    [1.] => 2.,
    [2.] => 4.,
    [3.] => 6.,
    [4.] => 8.,
];

pub static OR_TRAIN: TrainDataSet<2> = dataset!(
    [0., 0.] => 0.,
    [1., 0.] => 1.,
    [0., 1.] => 1.,
    [1., 1.] => 1.,
);

pub static XOR_TRAIN: TrainDataSet<2> = dataset!(
    [0., 0.] => 0.,
    [1., 0.] => 1.,
    [0., 1.] => 1.,
    [1., 1.] => 0.,
);

pub static NAND_TRAIN: TrainDataSet<2> = dataset!(
    [0., 0.] => 1.,
    [1., 0.] => 0.,
    [0., 1.] => 0.,
    [1., 1.] => 0.,
);

pub static AND_TRAIN: TrainDataSet<2> = dataset!(
    [0., 0.]=> 0.,
    [1., 0.]=> 1.,
    [0., 1.]=> 1.,
    [1., 1.]=> 1.,
);

pub static NOR_TRAIN: TrainDataSet<2> = dataset!(
    [0., 0.]=> 1.,
    [1., 0.]=> 0.,
    [0., 1.]=> 0.,
    [1., 1.]=> 0.,
);
