use interatomic::twobody::{Combined, LennardJones, WeeksChandlerAndersen};
use interatomic::Info;

fn main() {
    let lj = LennardJones::new(1.5, 2.0);
    let wca = WeeksChandlerAndersen::new(lj.clone());
    let pot = Combined::new(lj.clone(), wca.clone());
    let pot2 = Combined::new(pot.clone(), wca.clone());

    let s = serde_json::to_string(&pot2).unwrap();

    println!("{}", lj.url().unwrap());
    println!("h {}", s);
    println!("{:?}", lj);
}
