mod app;
mod configs;
mod fourier;
mod fourier_ui;
mod gpu;
mod scene;
mod trace;

use eframe::egui;

fn main() -> eframe::Result {
    env_logger::init();
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Optics Bench")
            .with_inner_size([1500.0, 1050.0])
            .with_min_inner_size([900.0, 700.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    if let Ok(shot) = std::env::var("OPTICS_SHOT") {
        // debug screenshots must not read or overwrite the user's saved state
        let dir = std::path::Path::new(&shot).with_extension("state");
        options.persistence_path = Some(dir);
        options.persist_window = false;
    }
    eframe::run_native("Optics Bench", options, Box::new(|cc| Ok(Box::new(app::OpticsApp::new(cc)))))
}
