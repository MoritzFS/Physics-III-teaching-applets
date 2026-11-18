//! The "Hydrogen" tab on its own: `cargo run --release -p qb_hydrogen`.

fn main() {
    qb_core::run_app(vec![Box::new(qb_hydrogen::HydrogenTab::new())]);
}
