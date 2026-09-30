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

use eframe::egui;

fn main() -> eframe::Result {
    env_logger::init();
    // a double-clicked app has no terminal: keep panic messages in ~/Library/Logs/Optics Bench.log
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(home) = std::env::var_os("HOME") {
            let path = std::path::Path::new(&home).join("Library/Logs/Optics Bench.log");
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                use std::io::Write;
                let bt = std::backtrace::Backtrace::force_capture();
                let _ = writeln!(f, "{:?}: {info}\n{bt}\n", std::time::SystemTime::now());
            }
        }
        default_hook(info);
    }));
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
