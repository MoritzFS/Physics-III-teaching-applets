//! The "Spin" tab on its own: `cargo run --release -p qb_spin`.

fn main() {
    qb_core::run_app(vec![Box::new(qb_spin::SpinTab::new())]);
}
