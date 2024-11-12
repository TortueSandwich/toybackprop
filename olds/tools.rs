#![allow(dead_code)]

pub fn sigmoid<To: num_traits::real::Real>(x:To) -> To {
    To::one() /(To::one() + (-x).exp())
}

// #[allow(dead_code)]
// fn cost(w1 : T, w2 : T, b:T) -> T {// w3: T, w4: T, b :T) -> T {
//     let mut result = 0.0;
//     for i in 0..TRAIN_PTR.len() {
//         let x1 = TRAIN_PTR[i][0].clone();
//         let x2 = TRAIN_PTR[i][1].clone();
//         //let x3 = x1 + x2;
//         //let x4 = x1 * x2;
//         let y = sigmoid(w1 * x1+ w2 * x2  + b);

//         let y_expected = TRAIN_PTR[i][2].clone();
//         let d = y - y_expected;
//         result += d.powi(2); 
//         //println!("Get : {:.2},  Expected : {:.2}", y , y_expected);
//     }
//     result /= TRAIN_PTR.len() as f64;
//     result
// }

// pub trait Model {
//     fn rand() -> Self;
//     fn forward(&self, x1: T,x2: T) -> T;
//     fn cost(&self) -> T {
//         let mut result = 0 as T;
//         for i in 0..TRAIN_PTR.len() {
//             let x1 = TRAIN_PTR[i][0].clone();
//             let x2 = TRAIN_PTR[i][1].clone();
//             let y = self.forward(x1, x2);
//             let y_expected = TRAIN_PTR[i][2].clone();
//             let d = y - y_expected;
//             result += d.powi(2); 
//         }
//         result /= TRAIN_PTR.len() as f64;
//         result
//     }
//     fn print_table(&self, set: TrainSet) {
//         let red = "\x1b[41m";
//         let green = "\x1b[42m";
//         let orange = "\x1b[43m";
    
//         for arr in set {
//             let (i,j) = (arr[0], arr[1]);
//             let res = self.forward(i,j);
//             print!("{}|{} = ",i,j);
//             if res > 0.9 { print!("{green}");
//             } else if res < 0.1 { print!("{red}");
//             } else { print!("{orange}"); }
//             println!("{}\x1b[0m", res);
//         }
//     }
//     fn finite_diff(&mut self) -> Self;
//     fn learn_with(&mut self, g : Self);
// }



// pub fn parse_cursor_position(response: &str) -> Option<(usize, usize)> {
//     if let Some(start) = response.find('\x1B') {
//         if let Some(end) = response.find('R') {
//             // La réponse est de la forme "\x1B[n;mR", où n et m sont les coordonnées
//             let coordinates = &response[start + 2..end];
//             let mut parts = coordinates.split(';');
//             if let (Some(row), Some(col)) = (parts.next(), parts.next()) {
//                 if let (Ok(row), Ok(col)) = (row.parse(), col.parse()) {
//                     return Some((row, col));
//                 }
//             }
//         }
//     }
//     None
// }