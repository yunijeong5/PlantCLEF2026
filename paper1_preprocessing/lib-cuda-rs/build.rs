// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
extern crate anyhow;

//use anyhow::{Context, Result};
use anyhow::Result;
use std::path::PathBuf;

const KERNEL_FILES: [&str; 6] = [
    "kernels/api.cu",
    "kernels/misc.cu",
    "kernels/matmult.cu",
    "kernels/softmax.cu",
    "kernels/interpolate2d.cu",
    "kernels/attn.cu",
];

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo:rerun-if-changed={kernel_file}");
    }

    /*
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };
    */

    //let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);

    let build_dir = "cuda_build_dir";
    let build_dir = PathBuf::from(build_dir).canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                build_dir,
                std::env::current_dir()?.display()
                ));

    let kernels = KERNEL_FILES.iter().collect();
    let mut builder = bindgen_cuda::Builder::default()
        .kernel_paths(kernels)
        .out_dir(build_dir.clone())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("-Icutlass/include")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose");

    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    //let out_file = build_dir.join("libflashattention.a");
    let out_file = build_dir.join("libcudars.a");
    builder.build_lib(out_file);

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=cudars");    // Librairie contenant les kernels
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
