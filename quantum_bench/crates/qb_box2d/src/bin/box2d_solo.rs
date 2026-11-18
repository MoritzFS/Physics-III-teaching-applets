//! The "2D box" tab on its own: `cargo run --release -p qb_box2d`.

fn main() {
    qb_core::run_app(vec![Box::new(qb_box2d::Box2dTab::new())]);
}
