use interatomic::twobody::{Combined, LennardJones, WeeksChandlerAndersen};

fn main() {
    let lj = LennardJones::new(1.5, 2.0);
    let wca = WeeksChandlerAndersen::new(1.5, 2.0);
    let pot = Combined::new(lj.clone(), wca.clone());
    let _pot2 = Combined::new(pot.clone(), wca.clone());

    println!("{:?}", lj);

    // #[cfg(feature = "serde")]
    // let s = serde_json::to_string(&pot2).unwrap();
    // #[cfg(feature = "serde")]
    // println!("h {}", s);
}
