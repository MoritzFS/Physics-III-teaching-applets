//! Entry points: a native window on the desktop, a canvas in the browser.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    use eframe::egui;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Stat Phys Bench").with_inner_size([1500.0, 1000.0]).with_min_inner_size([900.0, 650.0]),
        ..Default::default()
    };
    eframe::run_native("Stat Phys Bench", options, Box::new(|cc| Ok(Box::new(stat_phys_bench::StatApp::new(cc)))))
}

/// The browser entry: runs the app in the canvas of `index.html` with WebGL 2.
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast as _;

    eframe::WebLogger::init(log::LevelFilter::Info).ok();
    // eframe's own panic hook (which logs to the console) calls this one afterwards
    std::panic::set_hook(Box::new(|info| show_crash(&info.to_string())));

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().expect("no window").document().expect("no document");
        let Some(canvas) = document.get_element_by_id("stat_canvas") else {
            log::warn!("no canvas: not starting");
            return;
        };
        let canvas = canvas.dyn_into::<web_sys::HtmlCanvasElement>().expect("stat_canvas is not a canvas");
        let result =
            eframe::WebRunner::new().start(canvas, eframe::WebOptions::default(), Box::new(|cc| Ok(Box::new(stat_phys_bench::StatApp::new(cc))))).await;
        if let Some(loading) = document.get_element_by_id("loading") {
            loading.remove();
        }
        if let Err(e) = result {
            // most likely the browser has no WebGL 2: show the explanation from index.html
            let detail = e.as_string().unwrap_or_else(|| format!("{e:?}"));
            log::error!("could not start Stat Phys Bench: {detail}");
            if let Some(canvas) = document.get_element_by_id("stat_canvas") {
                canvas.remove();
            }
            if let Some(msg) = document.get_element_by_id("no_webgl") {
                let _ = msg.remove_attribute("hidden");
            }
            if let Some(el) = document.get_element_by_id("no_webgl_detail") {
                el.set_text_content(Some(&detail));
            }
        }
    });
}

/// A panic ends the app: say so in the page instead of leaving it frozen.
#[cfg(target_arch = "wasm32")]
fn show_crash(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else { return };
    if let Some(loading) = document.get_element_by_id("loading") {
        loading.remove();
    }
    if let Some(canvas) = document.get_element_by_id("stat_canvas") {
        let _ = canvas.set_attribute("data-crashed", "");
    }
    if let Some(el) = document.get_element_by_id("crashed") {
        let _ = el.remove_attribute("hidden");
    }
    if let Some(el) = document.get_element_by_id("crashed_detail") {
        el.set_text_content(Some(&format!("{message} (more in the browser's developer console)")));
    }
}
