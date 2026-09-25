//! wgpu (Metal on macOS) compute renderer with progressive accumulation.

use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

use crate::scene::{GpuLens, GpuObj, GpuPrim, GpuPrism};

pub const MODE_SCREEN: u32 = 0;
pub const MODE_EYE: u32 = 1;
pub const MODE_OVERVIEW: u32 = 2;

/// indices of the render targets
pub const T_SCREEN: usize = 0;
pub const T_EYE: usize = 1;
pub const T_OVERVIEW: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Globals {
    pub sun_dir: [f32; 4],
    pub sun_col: [f32; 4],
    pub counts: [u32; 4],
    pub scr_center: [f32; 4],
    pub scr_right: [f32; 4],
    pub scr_up: [f32; 4],
    pub scr_normal: [f32; 4],
    pub scr_info: [u32; 4],
    pub misc: [f32; 4],
    pub counts2: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ViewUniform {
    pub info: [u32; 4],
    pub info2: [u32; 4],
    pub origin: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub fwd: [f32; 4],
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
}

struct DynBuf {
    buf: wgpu::Buffer,
    cap: u64,
}

impl DynBuf {
    fn new(device: &wgpu::Device, label: &str, cap: u64) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: cap,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        DynBuf { buf, cap }
    }

    /// returns true if the buffer had to be reallocated
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> bool {
        let mut realloc = false;
        if bytes.len() as u64 > self.cap {
            *self = DynBuf::new(device, label, (bytes.len() as u64).next_power_of_two());
            realloc = true;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buf, 0, bytes);
        }
        realloc
    }
}

pub struct Target {
    pub width: u32,
    pub height: u32,
    _texture: wgpu::Texture,
    accum: wgpu::Buffer,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub tex_id: egui::TextureId,
    pub samples: u32,
    hash: u64,
}

pub struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    globals: wgpu::Buffer,
    prims: DynBuf,
    objs: DynBuf,
    lenses: DynBuf,
    prisms: DynBuf,
    dummy: wgpu::Buffer,
    bg0_plain: Option<wgpu::BindGroup>,
    bg0_full: Option<wgpu::BindGroup>,
    pub targets: [Option<Target>; 3],
    scene_hash: u64,
    frame: u32,
}

pub struct FrameData<'a> {
    pub prims: &'a [GpuPrim],
    pub objs: &'a [GpuObj],
    pub lenses: &'a [GpuLens],
    pub prisms: &'a [GpuPrism],
    pub globals: Globals,
    /// `None` = do not render this target this frame
    pub views: [Option<ViewUniform>; 3],
    pub spp: u32,
    pub max_samples: u32,
}

fn hash_bytes<H: Hasher>(h: &mut H, b: &[u8]) {
    b.hash(h);
}

impl Gpu {
    pub fn new(rs: &egui_wgpu::RenderState) -> Self {
        let device = &rs.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("optics tracer"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[
                uniform(0),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, true),
                storage(5, true),
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("view"),
            entries: &[
                uniform(0),
                storage(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tracer layout"),
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Gpu {
            pipeline,
            bgl0,
            bgl1,
            globals,
            prims: DynBuf::new(device, "prims", 64 * 256),
            objs: DynBuf::new(device, "objs", 32 * 64),
            lenses: DynBuf::new(device, "lenses", 48 * 32),
            prisms: DynBuf::new(device, "prisms", 96 * 16),
            dummy,
            bg0_plain: None,
            bg0_full: None,
            targets: [None, None, None],
            scene_hash: 0,
            frame: 0,
        }
    }

    /// Makes sure render target `which` exists with the given size and returns its texture id.
    pub fn target(&mut self, rs: &egui_wgpu::RenderState, which: usize, width: u32, height: u32) -> egui::TextureId {
        let width = width.clamp(8, 4096);
        let height = height.clamp(8, 4096);
        if let Some(t) = &self.targets[which] {
            if t.width == width && t.height == height {
                return t.tex_id;
            }
        }
        let device = &rs.device;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("view"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let accum = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accum"),
            size: (width as u64) * (height as u64) * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view uniform"),
            size: std::mem::size_of::<ViewUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view bg"),
            layout: &self.bgl1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        let mut renderer = rs.renderer.write();
        let tex_id = match &self.targets[which] {
            Some(old) => {
                renderer.update_egui_texture_from_wgpu_texture(device, &view, wgpu::FilterMode::Linear, old.tex_id);
                old.tex_id
            }
            None => renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear),
        };
        self.targets[which] = Some(Target {
            width,
            height,
            _texture: texture,
            accum,
            uniform,
            bind_group,
            tex_id,
            samples: 0,
            hash: 0,
        });
        if which == T_SCREEN {
            self.bg0_full = None;
        }
        tex_id
    }

    pub fn samples(&self, which: usize) -> u32 {
        self.targets[which].as_ref().map_or(0, |t| t.samples)
    }

    fn make_bg0(&self, device: &wgpu::Device, screen: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene bg"),
            layout: &self.bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.prims.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.objs.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.lenses.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: screen.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.prisms.buf.as_entire_binding() },
            ],
        })
    }

    /// Renders all requested views. Returns true while some view is still converging.
    pub fn render(&mut self, rs: &egui_wgpu::RenderState, data: &FrameData) -> bool {
        let device = &rs.device;
        let queue = &rs.queue;
        self.frame = self.frame.wrapping_add(1);

        // ---- scene buffers
        let mut h = std::collections::hash_map::DefaultHasher::new();
        hash_bytes(&mut h, bytemuck::cast_slice(data.prims));
        hash_bytes(&mut h, bytemuck::cast_slice(data.objs));
        hash_bytes(&mut h, bytemuck::cast_slice(data.lenses));
        hash_bytes(&mut h, bytemuck::cast_slice(data.prisms));
        let scene_hash = h.finish();
        if scene_hash != self.scene_hash || self.bg0_plain.is_none() {
            self.scene_hash = scene_hash;
            let mut realloc = false;
            realloc |= self.prims.upload(device, queue, "prims", bytemuck::cast_slice(data.prims));
            realloc |= self.objs.upload(device, queue, "objs", bytemuck::cast_slice(data.objs));
            realloc |= self.lenses.upload(device, queue, "lenses", bytemuck::cast_slice(data.lenses));
            realloc |= self.prisms.upload(device, queue, "prisms", bytemuck::cast_slice(data.prisms));
            if realloc {
                self.bg0_plain = None;
                self.bg0_full = None;
            }
        }
        if self.bg0_plain.is_none() {
            self.bg0_plain = Some(self.make_bg0(device, &self.dummy));
        }
        if self.bg0_full.is_none() {
            let bg = match &self.targets[T_SCREEN] {
                Some(t) => self.make_bg0(device, &t.accum),
                None => self.make_bg0(device, &self.dummy),
            };
            self.bg0_full = Some(bg);
        }

        // ---- decide which views need samples
        let mut g = data.globals;
        g.scr_info[2] = 0;
        g.counts[3] = 0;
        let mut gh = std::collections::hash_map::DefaultHasher::new();
        scene_hash.hash(&mut gh);
        hash_bytes(&mut gh, bytemuck::bytes_of(&g));
        let globals_hash = gh.finish();

        let mut work: Vec<(usize, ViewUniform)> = vec![];
        for (i, v) in data.views.iter().enumerate() {
            let (Some(v), Some(t)) = (v, self.targets[i].as_mut()) else { continue };
            let mut key = *v;
            key.info[3] = 0;
            key.info2 = [0, 0, key.info2[2], key.info2[3]];
            let mut vh = std::collections::hash_map::DefaultHasher::new();
            globals_hash.hash(&mut vh);
            hash_bytes(&mut vh, bytemuck::bytes_of(&key));
            (t.width, t.height).hash(&mut vh);
            let hash = vh.finish();
            if hash != t.hash {
                t.hash = hash;
                t.samples = 0;
            }
            if t.samples >= data.max_samples {
                continue;
            }
            let spp = data.spp.min(data.max_samples - t.samples).max(1);
            let mut u = *v;
            u.info[1] = t.width;
            u.info[2] = t.height;
            u.info[3] = spp;
            u.info2[0] = t.samples;
            u.info2[1] = self.frame.wrapping_mul(7919).wrapping_add(i as u32 * 104729);
            work.push((i, u));
        }

        // screen samples after this frame (other views read the screen image)
        let mut g = data.globals;
        g.counts[3] = self.frame;
        if let Some(t) = &self.targets[T_SCREEN] {
            let extra = work.iter().find(|(i, _)| *i == T_SCREEN).map_or(0, |(_, u)| u.info[3]);
            g.scr_info[0] = t.width;
            g.scr_info[1] = t.height;
            g.scr_info[2] = t.samples + extra;
        } else {
            g.counts[2] &= !2;
        }
        if work.is_empty() {
            return false;
        }
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&g));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("trace") });
        for (i, u) in &work {
            let t = self.targets[*i].as_mut().unwrap();
            queue.write_buffer(&t.uniform, 0, bytemuck::bytes_of(u));
            t.samples += u.info[3];
            let bg0 = if *i == T_SCREEN { self.bg0_plain.as_ref() } else { self.bg0_full.as_ref() };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("trace"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg0, &[]);
            pass.set_bind_group(1, Some(&t.bind_group), &[]);
            pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()));
        true
    }
}
