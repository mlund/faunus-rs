fn main() {
    if let Err(err) = faunus::cli::do_main() {
        eprintln!("Error: {}", &err);
        std::process::exit(1);
    }
}
