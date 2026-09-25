//! Entry points: a native window on the desktop, a canvas in the browser.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    use eframe::egui;

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
    eframe::run_native("Optics Bench", options, Box::new(|cc| Ok(Box::new(optics_bench::OpticsApp::new(cc)))))
}

/// The browser entry: runs the app in the canvas of `index.html` with WebGPU.
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::egui_wgpu::{WgpuSetup, wgpu};
    use wasm_bindgen::JsCast as _;

    eframe::WebLogger::init(log::LevelFilter::Info).ok();
    // eframe's own panic hook (which logs to the console) calls this one afterwards
    std::panic::set_hook(Box::new(|info| show_crash(&info.to_string())));

    // the renderer needs compute shaders and storage textures: WebGPU only, no WebGL fallback
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
    }
    let web_options = eframe::WebOptions { wgpu_options, ..Default::default() };

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().expect("no window").document().expect("no document");
        let Some(canvas) = document.get_element_by_id("optics_canvas") else {
            // index.html removes the canvas when the browser has no WebGPU
            log::warn!("no canvas: not starting");
            return;
        };
        let canvas = canvas.dyn_into::<web_sys::HtmlCanvasElement>().expect("optics_canvas is not a canvas");
        let result = eframe::WebRunner::new()
            .start(canvas, web_options, Box::new(|cc| Ok(Box::new(optics_bench::OpticsApp::new(cc)))))
            .await;
        if let Some(loading) = document.get_element_by_id("loading") {
            loading.remove();
        }
        if let Err(e) = result {
            // most likely there is no usable WebGPU adapter: show the explanation from index.html
            let detail = e.as_string().unwrap_or_else(|| format!("{e:?}"));
            log::error!("could not start Optics Bench: {detail}");
            if let Some(canvas) = document.get_element_by_id("optics_canvas") {
                canvas.remove();
            }
            if let Some(msg) = document.get_element_by_id("no_webgpu") {
                let _ = msg.remove_attribute("hidden");
            }
            if let Some(el) = document.get_element_by_id("no_webgpu_detail") {
                el.set_text_content(Some(&detail));
            }
        }
    });
}

/// A panic ends the app: say so in the page instead of leaving it frozen.
#[cfg(target_arch = "wasm32")]
fn show_crash(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else { return };
    if let Some(canvas) = document.get_element_by_id("optics_canvas") {
        let _ = canvas.set_attribute("data-crashed", "");
    }
    if let Some(el) = document.get_element_by_id("crashed") {
        let _ = el.remove_attribute("hidden");
    }
    if let Some(el) = document.get_element_by_id("crashed_detail") {
        el.set_text_content(Some(&format!("{message} (more in the browser's developer console)")));
    }
}
