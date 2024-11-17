use crate::matrix::Matrix;

pub trait HList {}


pub struct Nil;
impl HList for Nil {}
impl<const I : usize, const O : usize> Push<I, O> for Nil {
    type Output = HCons<I,O, Nil>;
    fn push(self, data: Matrix<O, I>) -> Self::Output {
        HCons {
            data, next : self,
        }
    }

}

pub struct HLeaf<const I: usize, const O: usize> {
    pub data : Matrix<O, I>,
}
impl<const I: usize, const O: usize> HList for HLeaf<I,O> {
}

pub struct HCons<const I: usize, const O: usize, T: HList> {
    pub data : Matrix<O, I>,
    pub next: T,
}
impl<const I: usize, const O: usize, T: HList> HList for HCons<I, O,T> {}

// pub trait ComputePrepend<const I: usize, const O: usize> {
//     type Output;
// }
// impl<const NewI: usize, const I: usize,const O: usize, const LO: usize,> ComputePrepend<NewI, I> for HLeaf<I, O> {
//     type Output = HCons<NewI, I, HLeaf<I,O>>;
// }

// impl<const NI: usize, const I: usize, const O: usize, T: HList> ComputePrepend<NI, I>
//     for HCons<I, O, T>
// {
//     type Output = HCons<NI, I, HCons<I, O, T>>;
// }


// pub type Prepend<const I:usize,const O:usize, L> = <L as ComputePrepend<I,O>>::Output;


pub trait Push<const NI: usize, const NO: usize>: HList {
    type Output: HList;
    fn push(self, data: Matrix<NO, NI>) -> Self::Output;
}

impl<const NewI: usize, const I: usize, const O: usize> Push<NewI, I> for HLeaf<I,O> {
    type Output = HCons<NewI, I, HLeaf<I,O>>;

    fn push(self, data: Matrix<I,NewI>) -> Self::Output {
        HCons { data, next: self }
    }
}

impl<const NI: usize, const NO: usize, const I: usize, const O: usize,  T: HList> Push<NI, NO>
    for HCons<I, O, T>
where
    T: HList,
{
    type Output = HCons<NI, NO, HCons<I, O, T>>;

    fn push(self, data: Matrix<NO, NI>) -> Self::Output {
        HCons {
            data,
            next: self,
        }
    }
}

// pub trait Pop: HList<LO> {
//     type Head; // Le type des données du premier élément
//     type Tail: HList; // La liste restante
//     fn pop(self) -> (Self::Head, Option<Self::Tail>);
// }

// impl<const I: usize, const O: usize> Pop for HLeaf<I,O> {
//     type Head = Matrix<O,I>;
//     type Tail = HLeaf<I,O>;

//     fn pop(self) -> (Self::Head, Option<Self::Tail>) {
//         (self.data, None)
//     }
// }

// impl<const I: usize, const O: usize, T: HList> Pop for HCons<I, O, T> {
//     type Head = Matrix<O, I>;
//     type Tail = T;

//     fn pop(self) -> (Self::Head, Option<Self::Tail>) {
//         (self.data, Some(self.next))
//     }
// }

