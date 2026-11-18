//! WGSL validation with naga (the same front end wgpu uses natively; browsers
//! use their own compilers, which can be stricter). Copy `validate` into a
//! test of your tab crate for your own shaders:
//!
//! ```text
//! validate(&qb_core::gpu::shader_source(MY_WGSL, false));
//! ```

fn validate(src: &str) {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("{}", e.emit_to_string(src)),
    };
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default());
    if let Err(e) = v.validate(&module) {
        panic!("{}", e.emit_to_string(src));
    }
}

#[test]
fn phase_colour_snippet_validates() {
    validate(qb_core::widgets::PHASE_WGSL);
}

#[test]
fn shader_rect_demo_validates() {
    for srgb in [false, true] {
        validate(&qb_core::gpu::shader_source(qb_core::gpu::PHASE_WHEEL_WGSL, srgb));
    }
}

#[test]
fn cpu_and_wgsl_phase_colours_agree_on_the_formula() {
    // the CPU function at a few angles (the WGSL uses the same expressions)
    use qb_core::widgets::phase_rgb;
    let tau = std::f32::consts::TAU;
    assert_eq!(phase_rgb(0.0), [1.0, 0.0, 0.0]);
    let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
    assert!(close(phase_rgb(tau / 3.0), [0.0, 1.0, 0.0]));
    assert!(close(phase_rgb(2.0 * tau / 3.0), [0.0, 0.0, 1.0]));
    assert!(close(phase_rgb(tau / 2.0), [0.0, 1.0, 1.0]));
    assert!(close(phase_rgb(-tau / 6.0), phase_rgb(5.0 * tau / 6.0)));
}
