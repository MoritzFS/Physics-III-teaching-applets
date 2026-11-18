//! Custom GPU drawing inside egui: [`ShaderRect`], a fragment shader that fills
//! a rectangle of the UI (e.g. a ray-marched 3D density).
//!
//! Rules (the web build may run on WebGL2):
//! * render pipelines only — **no compute shaders, no storage buffers or
//!   storage textures**;
//! * data goes in through one uniform buffer (≤ 16 KiB) and one 3D data
//!   texture of `Rgba32Float` texels read with `textureLoad` (WebGL2 cannot
//!   filter 32-bit float textures; interpolate yourself if needed);
//! * validate your WGSL with naga in a unit test (see `tests/wgsl.rs` in qb_core).
//!
//! The WGSL you pass must define `struct Uniforms { … }` and
//! `fn shade(uv: vec2<f32>) -> vec4<f32>`; `uv` runs over [0, 1]² across the
//! rectangle (y down) and the result is an sRGB-encoded colour with straight
//! alpha (like `Color32`). Available bindings:
//!
//! ```text
//! qb_frame.rect_px : vec4<f32>   rectangle in physical pixels (x, y, w, h)
//! qb_frame.misc    : vec4<f32>   (pixels_per_point, 0, 0, 0)
//! u                : Uniforms    your uniform struct
//! qb_data          : texture_3d<f32>   your data (1×1×1 if none)
//! ```
//!
//! plus `qb_phase_color(arg, intensity)` and `qb_density_color(t)` from
//! [`crate::widgets::PHASE_WGSL`].
//!
//! ```ignore
//! // once (Tab::init_gpu, or lazily from ctx.render_state):
//! self.view = Some(ShaderRect::new(rs, "orbital", MY_WGSL, std::mem::size_of::<MyUniforms>(), Some([64, 64, 64])));
//! // when the data changes:
//! view.write_data(rs, &texels);          // 64·64·64 [f32; 4] texels
//! // every frame:
//! view.paint(ui.painter(), rect, bytemuck::bytes_of(&my_uniforms));
//! ```
//!
//! [`PhaseWheelDemo`] is the smallest working example (used by the hydrogen stub).

use eframe::egui;
use eframe::egui_wgpu::{self, CallbackTrait, RenderState, wgpu};

use crate::widgets::PHASE_WGSL;

const PRELUDE: &str = r#"
struct QbFrame {
    rect_px: vec4<f32>,
    misc: vec4<f32>,
};
@group(0) @binding(0) var<uniform> qb_frame: QbFrame;
@group(0) @binding(1) var<uniform> u: Uniforms;
@group(0) @binding(2) var qb_data: texture_3d<f32>;

struct QbVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn qb_vs(@builtin(vertex_index) i: u32) -> QbVsOut {
    // one triangle that covers the viewport (egui sets the viewport to the rectangle)
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: QbVsOut;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

fn qb_srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

@fragment
fn qb_fs(v: QbVsOut) -> @location(0) vec4<f32> {
    let c = shade(v.uv);
    var rgb = clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    if (QB_SRGB_TARGET) {
        rgb = qb_srgb_to_linear(rgb);
    }
    let a = clamp(c.a, 0.0, 1.0);
    return vec4<f32>(rgb * a, a);
}
"#;

/// The complete WGSL module built from your code (for naga validation in tests).
/// `srgb_target` is the render target's `is_srgb()`.
pub fn shader_source(user_wgsl: &str, srgb_target: bool) -> String {
    format!("const QB_SRGB_TARGET: bool = {srgb_target};\n{PHASE_WGSL}\n{user_wgsl}\n{PRELUDE}")
}

/// A fragment shader drawn into a rectangle of the UI. Cheap to clone.
#[derive(Clone)]
pub struct ShaderRect {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    frame_buf: wgpu::Buffer,
    user_buf: wgpu::Buffer,
    user_size: u64,
    data: wgpu::Texture,
    data_size: [u32; 3],
}

impl ShaderRect {
    /// Compiles `user_wgsl` (see the module docs). `uniform_size` is the byte
    /// size of your `Uniforms` (rounded up to 16); `data_size` the size of the
    /// 3D data texture in texels, or None.
    pub fn new(rs: &RenderState, label: &str, user_wgsl: &str, uniform_size: usize, data_size: Option<[u32; 3]>) -> Self {
        let device = &rs.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_source(user_wgsl, rs.target_format.is_srgb()).into()),
        });
        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("qb_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("qb_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: rs.target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qb_frame"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let user_size = (uniform_size.max(16) as u64).div_ceil(16) * 16;
        let user_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qb_uniforms"),
            size: user_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let data_size = data_size.unwrap_or([1, 1, 1]).map(|s| s.max(1));
        let data = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qb_data"),
            size: wgpu::Extent3d {
                width: data_size[0],
                height: data_size[1],
                depth_or_array_layers: data_size[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bind_group = Self::make_bind_group(device, &bind_group_layout, &frame_buf, &user_buf, &data);
        ShaderRect {
            pipeline,
            bind_group_layout,
            bind_group,
            frame_buf,
            user_buf,
            user_size,
            data,
            data_size,
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        frame: &wgpu::Buffer,
        user: &wgpu::Buffer,
        data: &wgpu::Texture,
    ) -> wgpu::BindGroup {
        let view = data.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qb_shader_rect"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: user.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        })
    }

    /// Size of the data texture in texels.
    pub fn data_size(&self) -> [u32; 3] {
        self.data_size
    }

    /// Uploads the whole data texture: `texels.len()` must be w·h·d, x fastest, then y, then z.
    pub fn write_data(&self, rs: &RenderState, texels: &[[f32; 4]]) {
        let [w, h, d] = self.data_size;
        assert_eq!(texels.len(), (w * h * d) as usize, "ShaderRect::write_data: wrong number of texels");
        rs.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.data,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 16),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: d,
            },
        );
    }

    /// Replaces the data texture by one of a different size.
    pub fn resize_data(&mut self, rs: &RenderState, size: [u32; 3]) {
        let size = size.map(|s| s.max(1));
        if size == self.data_size {
            return;
        }
        self.data = rs.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("qb_data"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: size[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.data_size = size;
        self.bind_group = Self::make_bind_group(&rs.device, &self.bind_group_layout, &self.frame_buf, &self.user_buf, &self.data);
    }

    /// Draws the shader into `rect` with these uniform bytes (`bytemuck::bytes_of(&u)`).
    /// One paint per `ShaderRect` per frame (the uniform buffer is shared).
    pub fn paint(&self, painter: &egui::Painter, rect: egui::Rect, uniforms: &[u8]) {
        let mut bytes = uniforms.to_vec();
        bytes.resize(self.user_size as usize, 0);
        let cb = Callback {
            rect,
            shader: self.clone(),
            uniforms: bytes,
        };
        painter.add(egui_wgpu::Callback::new_paint_callback(rect, cb));
    }
}

struct Callback {
    rect: egui::Rect,
    shader: ShaderRect,
    uniforms: Vec<u8>,
}

impl CallbackTrait for Callback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        _resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let ppp = screen.pixels_per_point;
        let r = self.rect;
        let frame: [f32; 8] = [r.min.x * ppp, r.min.y * ppp, r.width() * ppp, r.height() * ppp, ppp, 0.0, 0.0, 0.0];
        queue.write_buffer(&self.shader.frame_buf, 0, bytemuck::cast_slice(&frame));
        queue.write_buffer(&self.shader.user_buf, 0, &self.uniforms);
        vec![]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, _resources: &egui_wgpu::CallbackResources) {
        pass.set_pipeline(&self.shader.pipeline);
        pass.set_bind_group(0, &self.shader.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The smallest complete example: a phase colour wheel drawn by the GPU with
/// `qb_phase_color` (hue = angle, brightness = radius), rotating with `turn`.
pub struct PhaseWheelDemo {
    shader: ShaderRect,
}

/// WGSL of [`PhaseWheelDemo`].
pub const PHASE_WHEEL_WGSL: &str = r#"
// keep uniform structs made of vec4s (or scalars in multiples of four):
// a vec3 after a scalar is aligned to 16 bytes in WGSL, unlike in Rust
struct Uniforms {
    params: vec4<f32>, // x = turn
};
fn shade(uv: vec2<f32>) -> vec4<f32> {
    let aspect = qb_frame.rect_px.z / max(qb_frame.rect_px.w, 1.0);
    let p = vec2<f32>((uv.x - 0.5) * aspect, 0.5 - uv.y) * 2.0;
    let r = length(p);
    if (r > 1.0) {
        return vec4<f32>(0.0);
    }
    // texel (0,0,0) of the data texture, unused here: shows how to read it
    let d = textureLoad(qb_data, vec3<i32>(0, 0, 0), 0);
    let arg = atan2(p.y, p.x) - u.params.x + 0.0 * d.x;
    return vec4<f32>(qb_phase_color(arg, r), 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WheelUniforms {
    params: [f32; 4],
}

impl PhaseWheelDemo {
    pub fn new(rs: &RenderState) -> Self {
        PhaseWheelDemo {
            shader: ShaderRect::new(rs, "phase wheel demo", PHASE_WHEEL_WGSL, std::mem::size_of::<WheelUniforms>(), None),
        }
    }

    /// Draws the wheel into `rect`, rotated by `turn` radians.
    pub fn paint(&self, painter: &egui::Painter, rect: egui::Rect, turn: f32) {
        self.shader.paint(
            painter,
            rect,
            bytemuck::bytes_of(&WheelUniforms {
                params: [turn, 0.0, 0.0, 0.0],
            }),
        );
    }
}
