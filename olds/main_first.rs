use rand::Rng;

const SIZE: usize = 5; 
const TRAIN : [[f32;2]; SIZE] = [
    [0., 0.],
    [1., 2.],
    [2., 4.],
    [3., 6.],
    [4., 8.],
]; 
const TRAIN_COUNT: usize = TRAIN.len(); 

fn rand_float() -> f32 {
    let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
    rng.gen()
}

fn cost(w: f32, b: f32) -> f32 {
    let mut result: f32 = 0.;
    for i in 0..TRAIN_COUNT {
        let x: f32 = TRAIN[i][0];
        let y: f32 = x*w + b;
        let d:f32 = y -TRAIN[i][1];
        result += d*d;
        //println!("actual : {}   expected : {}", y, TRAIN[i][1]);
    }
    result /= TRAIN_COUNT as f32;
    result
}

fn main() {
    let mut w: f32 = rand_float()*10.0;
    let mut b: f32 = rand_float()*5.0;

    let eps = 1e-3;
    let rate = 1e-3;
    for _ in 0..500 {
        let c: f32 = cost(w,b);
        println!("cost : {},  w : {}, b : {}", c, w, b);

        let dw = (cost(w + eps, b) - c)/eps;
        let db = (cost(w, b + eps) - c)/eps;
        w -= rate*dw;
        b -= rate*db;
    }

    println!("---------------");
    println!("w = {}, b = {}", w, b);
}
