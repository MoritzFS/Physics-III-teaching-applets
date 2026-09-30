//! Optics Bench: a GPU ray-optics bench, a 4f Fourier-optics bench, a
//! dispersion bench (with thunder and whistlers) and a rainbow bench.
//!
//! The same app runs as a desktop program (`main.rs`) and in the browser
//! (WebAssembly + WebGPU, also started from `main.rs`).

mod app;
mod audio;
mod configs;
mod dispersion;
mod dispersion_ui;
mod fourier;
mod fourier_ui;
mod gpu;
mod rainbow;
mod rainbow_ui;
mod scene;
mod sound;
mod sound_ui;
mod trace;
mod worker;
#[cfg(target_arch = "wasm32")]
mod web;

pub use app::OpticsApp;
