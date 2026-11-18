//! The "1D waves" tab on its own: `cargo run --release -p qb_wave1d`.

fn main() {
    qb_core::run_app(vec![Box::new(qb_wave1d::Wave1dTab::new())]);
}
