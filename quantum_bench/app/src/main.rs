//! Quantum Bench, the full app: all four tabs in one window (desktop) or in
//! the page's canvas (web). Everything else lives in `qb_core::run_app`.

fn main() {
    qb_core::run_app(vec![
        Box::new(qb_wave1d::Wave1dTab::new()),
        Box::new(qb_box2d::Box2dTab::new()),
        Box::new(qb_hydrogen::HydrogenTab::new()),
        Box::new(qb_spin::SpinTab::new()),
    ]);
}
