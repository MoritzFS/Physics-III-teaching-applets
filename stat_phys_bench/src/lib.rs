//! Stat Phys Bench: the five worked examples of partition functions, and
//! light in a box, as simulations in contact with a heat bath, next to the
//! formulas.
//!
//! 1. particles in boxes of fixed energy (`levels`)
//! 2. a gas of free particles in a box (`gas`)
//! 3. the same gas of atoms with a ground and an excited level (`gas`)
//! 4. classical modes: masses on springs or strings (`oscillators`)
//! 5. the same modes with quantised energy (`oscillators`)
//! 6. light in a box: the modes of a cavity, Rayleigh–Jeans and Planck (`cavity`)
//!
//! The same app runs as a desktop program and in the browser (WebAssembly
//! with WebGL 2), both started from `main.rs`.

mod app;
mod cavity;
mod cavity_ui;
mod gas;
mod gas_ui;
mod levels;
mod levels_ui;
mod oscillators;
mod oscillators_ui;
mod plot;
mod rng;
mod stats;
#[cfg(target_arch = "wasm32")]
mod web;
mod widgets;

pub use app::StatApp;
