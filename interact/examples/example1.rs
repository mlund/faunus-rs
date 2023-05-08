use interact::twobody::{LennardJones, TwobodyEnergy};
use interact::Citation;

fn main() {
    let lj = LennardJones::new(1.5, 2.0);
    let s = serde_json::to_string(&lj).unwrap();

    println!("{}", lj.url().unwrap());
    println!("h {}", s);
    println!("{:?}", lj);
}
