use std::env;
use std::process::Command;
use std::path::PathBuf;

fn main() {
    // Determine the build mode (default to "debug")
    let build_mode = env::var("CARGO_PROFILE").unwrap_or_else(|_| "debug".to_string());

    // CARGO_BUILD_TARGET_DIR
    let build_target_dir = env::var("CARGO_MANIFEST_PATH").unwrap_or_else(|_| "empty".to_string());
    println!("Build mode: {}", build_target_dir);

    // let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());

    // let mut executable_path = PathBuf::from(target_dir);
    // executable_path.push(build_mode);
    // executable_path.push(env!("CARGO_PKG_NAME")); // Use the package name as the executable name

    // if cfg!(windows) {
    //     executable_path.set_extension("exe");
    // }

    // Print the executable path for debugging
    // println!("Executable path: {}", executable_path.display());

    // Run the executable with arguments
    // let output = Command::new(&executable_path)
    //     .arg("--example-arg")
    //     .output()
    //     .expect("Failed to execute process");

    // // Display the output
    // println!("Status: {}", output.status);
    // println!("Stdout: {}", String::from_utf8_lossy(&output.stdout));
    // println!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
}
