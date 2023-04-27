use interact::lj;
use interact::TwobodyEnergy;

fn main() {
    let lj = lj::LennardJones::new(1.5, 2.0);
    let s = serde_json::to_string(&lj).unwrap();

    println!("{}", lj.cite().unwrap());
    println!("h {}", s);
    println!("{:?}", lj);
}
