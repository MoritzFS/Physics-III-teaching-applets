//! Clock for UI animations (flashes, fades): [`wall_time`].
//!
//! Interactively it is real time. In screenshot mode it is a virtual clock
//! that the shell advances with the simulation (one second per `time_scale`
//! units of simulated time, as when playing at speed 1), so a shot at QB_T
//! shows animations as a viewer would have seen them at that moment.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::Instant;

static START: OnceLock<Instant> = OnceLock::new();
static VIRTUAL: AtomicBool = AtomicBool::new(false);
static VIRTUAL_T: AtomicU64 = AtomicU64::new(0);

/// Seconds since start-up, for animations. Use this instead of `Instant::now()`
/// in anything a screenshot should capture.
pub fn wall_time() -> f64 {
    if VIRTUAL.load(Ordering::Relaxed) {
        f64::from_bits(VIRTUAL_T.load(Ordering::Relaxed))
    } else {
        START.get_or_init(Instant::now).elapsed().as_secs_f64()
    }
}

/// Switches to the virtual clock (screenshot mode).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn use_virtual() {
    VIRTUAL.store(true, Ordering::Relaxed);
}

/// Advances the virtual clock by `dt` seconds.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn advance_virtual(dt: f64) {
    let t = f64::from_bits(VIRTUAL_T.load(Ordering::Relaxed)) + dt.max(0.0);
    VIRTUAL_T.store(t.to_bits(), Ordering::Relaxed);
}
