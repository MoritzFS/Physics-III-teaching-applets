// Optics bench: progressive GPU ray tracer.
//
// One compute shader renders three kinds of "sensors":
//   mode 0 = screen : every pixel is a point on a diffuse screen. We sample the
//                     aperture of the lens in front of it and trace backwards.
//   mode 1 = eye    : every pixel is a point on the retina. We sample the pupil,
//                     refract through the (accommodating) eye lens, and trace.
//   mode 2 = overview camera (pinhole) that looks at the whole bench.
//
// Lenses are ideal thin lenses (slope form t' = t - h P), optionally with a
// third order spherical aberration term and chromatic dispersion. Apertures
// are lenses without power whose rim is an opaque plate. Prisms are real
// glass bodies (Snell's law, Cauchy dispersion).
//
// Colour: when something dispersive is on the bench, every sample carries a
// random wavelength. Paths that never meet dispersive glass return plain RGB;
// paths that do are converted with CIE colour matching functions.

const PI: f32 = 3.14159265;
const INF: f32 = 1e30;
const EPS: f32 = 1e-4;
const NONE: u32 = 0xffffffffu;

struct Globals {
    sun_dir: vec4<f32>,    // xyz: direction to sun, w: angular radius (rad)
    sun_col: vec4<f32>,    // rgb: sun irradiance, w: sky ambient strength
    counts: vec4<u32>,     // n_objs, n_lenses, flags (1 chromatic, 2 screen present), frame
    scr_center: vec4<f32>, // xyz, w: half width
    scr_right: vec4<f32>,  // xyz, w: half height
    scr_up: vec4<f32>,     // xyz, w: half thickness
    scr_normal: vec4<f32>, // xyz (front), w: display gain
    scr_info: vec4<u32>,   // px width, px height, samples, unused
    misc: vec4<f32>,       // fog density, grass height, grass cell size, unused
    counts2: vec4<u32>,    // n_prisms, unused...
};

struct View {
    info: vec4<u32>,   // mode, width, height, spp
    info2: vec4<u32>,  // samples so far, seed, visibility mask, lenses a screen ray must pass (tube)
    origin: vec4<f32>, // w: exposure
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
};

struct Prim {
    hdr: vec4<u32>,  // kind (0 sphere, 1 capsule, 2 box), packed rgb, visibility, unused
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
};

struct Obj {
    sphere: vec4<f32>,
    range: vec4<u32>,
};

struct Lens {
    center: vec4<f32>, // xyz, w: aperture radius
    axis: vec4<f32>,   // xyz, w: power (dioptres)
    params: vec4<f32>, // spherical aberration, dispersion, rim width, 1 = aperture plate
};

struct Prism {
    center: vec4<f32>, // centroid, w: half height
    axes: vec4<f32>,   // right.xz, fwd.xz
    p1: vec4<f32>,     // side planes in local (x, z): normal.x, normal.z, offset
    p2: vec4<f32>,
    p3: vec4<f32>,
    glass: vec4<f32>,  // Cauchy A, B, n_d
};

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var<storage, read> prims: array<Prim>;
@group(0) @binding(2) var<storage, read> objs: array<Obj>;
@group(0) @binding(3) var<storage, read> lenses: array<Lens>;
@group(0) @binding(4) var<storage, read> screen_img: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> prisms: array<Prism>;

@group(1) @binding(0) var<uniform> V: View;
@group(1) @binding(1) var<storage, read_write> accum: array<vec4<f32>>;
@group(1) @binding(2) var out_tex: texture_storage_2d<rgba8unorm, write>;

// ---------------------------------------------------------------- random

var<private> rng: u32;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rnd() -> f32 {
    rng = pcg(rng);
    return f32(rng >> 8u) * (1.0 / 16777216.0);
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i++) {
        v += a * vnoise(q);
        q = q * 2.03 + vec2<f32>(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

fn unpack_rgb(c: u32) -> vec3<f32> {
    return vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
}

fn basis_x(n: vec3<f32>) -> vec3<f32> {
    let a = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    return normalize(cross(a, n));
}

fn cosine_dir(n: vec3<f32>) -> vec3<f32> {
    let r = sqrt(rnd());
    let phi = 2.0 * PI * rnd();
    let t = basis_x(n);
    let b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - r * r)));
}

// ---------------------------------------------------------------- sky

fn sky_gradient(d: vec3<f32>) -> vec3<f32> {
    let zenith = vec3<f32>(0.13, 0.30, 0.78);
    let horizon = vec3<f32>(0.62, 0.74, 0.90);
    return mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), 0.5));
}

fn haze() -> vec3<f32> {
    return vec3<f32>(0.60, 0.70, 0.84);
}

fn sky(d: vec3<f32>) -> vec3<f32> {
    var c = sky_gradient(d);
    if d.y > 0.0 {
        let uv = d.xz / (d.y + 0.12);
        let cl = smoothstep(0.52, 0.85, fbm(uv * 1.1 + vec2<f32>(3.1, 1.7)));
        c = mix(c, vec3<f32>(1.0, 1.0, 1.02), cl * 0.8 * smoothstep(0.0, 0.2, d.y));
    }
    let sd = G.sun_dir.xyz;
    let cs = dot(d, sd);
    c += vec3<f32>(1.0, 0.8, 0.55) * (0.35 * pow(max(cs, 0.0), 40.0));
    if cs > cos(G.sun_dir.w) {
        c += vec3<f32>(9.0, 8.0, 6.5);
    }
    // two ranges of distant hills (periodic in azimuth)
    let az = atan2(d.z, d.x);
    let ring = vec2<f32>(cos(az), sin(az));
    let far_hill = 0.035 + 0.05 * fbm(ring * 2.2 + vec2<f32>(7.0, 3.0));
    let near_hill = 0.012 + 0.035 * fbm(ring * 3.4 + vec2<f32>(1.0, 11.0));
    if d.y < far_hill {
        c = mix(vec3<f32>(0.30, 0.42, 0.52), haze(), 0.45);
    }
    if d.y < near_hill {
        c = mix(vec3<f32>(0.10, 0.20, 0.10), haze(), 0.45);
    }
    return c;
}

fn fog(c: vec3<f32>, dist: f32) -> vec3<f32> {
    let f = 1.0 - exp(-dist * G.misc.x);
    return mix(c, haze(), f);
}

// ---------------------------------------------------------------- primitives

fn isect_sphere(ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cc = dot(oc, oc) - r * r;
    let h = b * b - cc;
    if h < 0.0 {
        return -1.0;
    }
    let s = sqrt(h);
    let t = -b - s;
    if t > EPS {
        return t;
    }
    return -b + s;
}

fn isect_capsule(ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, r: f32) -> f32 {
    let ba = pb - pa;
    let oa = ro - pa;
    let baba = dot(ba, ba);
    let bard = dot(ba, rd);
    let baoa = dot(ba, oa);
    let rdoa = dot(rd, oa);
    let oaoa = dot(oa, oa);
    let a = baba - bard * bard;
    let b = baba * rdoa - baoa * bard;
    let c = baba * oaoa - baoa * baoa - r * r * baba;
    let h = b * b - a * c;
    if a < 1e-9 * baba {
        // ray parallel to the axis: only the end caps can be hit
        let t1 = isect_sphere(ro, rd, pa, r);
        let t2 = isect_sphere(ro, rd, pb, r);
        if t1 > EPS && (t2 <= EPS || t1 < t2) {
            return t1;
        }
        return t2;
    }
    if h >= 0.0 {
        let t = (-b - sqrt(h)) / a;
        let y = baoa + t * bard;
        if y > 0.0 && y < baba {
            return t;
        }
        let oc = select(ro - pb, oa, y <= 0.0);
        let b2 = dot(rd, oc);
        let c2 = dot(oc, oc) - r * r;
        let h2 = b2 * b2 - c2;
        if h2 > 0.0 {
            return -b2 - sqrt(h2);
        }
    }
    return -1.0;
}

fn capsule_normal(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> vec3<f32> {
    let ba = b - a;
    let pa = p - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return (pa - h * ba) / r;
}

// Oriented box in its local frame. Returns (t, local normal).
fn isect_box_local(ro: vec3<f32>, rd: vec3<f32>, half: vec3<f32>) -> vec4<f32> {
    let d = select(rd, vec3<f32>(1e-8), abs(rd) < vec3<f32>(1e-8));
    let m = 1.0 / d;
    let n = m * ro;
    let k = abs(m) * half;
    let t1 = -n - k;
    let t2 = -n + k;
    let tn = max(max(t1.x, t1.y), t1.z);
    let tf = min(min(t2.x, t2.y), t2.z);
    if tn > tf || tf < EPS || tn < EPS {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    let nrm = -sign(d) * step(t1.yzx, t1.xyz) * step(t1.zxy, t1.xyz);
    return vec4<f32>(tn, nrm);
}

// Box prim: a = center, b = half extents, c = (right.x, right.z, fwd.x, fwd.z)
fn box_local_ray(pr: Prim, ro: vec3<f32>, rd: vec3<f32>) -> array<vec3<f32>, 2> {
    let r = vec3<f32>(pr.c.x, 0.0, pr.c.y);
    let f = vec3<f32>(pr.c.z, 0.0, pr.c.w);
    let o = ro - pr.a.xyz;
    return array<vec3<f32>, 2>(
        vec3<f32>(dot(o, r), o.y, dot(o, f)),
        vec3<f32>(dot(rd, r), rd.y, dot(rd, f)),
    );
}

fn isect_prim(pr: Prim, ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    switch pr.hdr.x {
        case 0u: {
            return isect_sphere(ro, rd, pr.a.xyz, pr.a.w);
        }
        case 1u: {
            return isect_capsule(ro, rd, pr.a.xyz, pr.b.xyz, pr.a.w);
        }
        default: {
            let lr = box_local_ray(pr, ro, rd);
            return isect_box_local(lr[0], lr[1], pr.b.xyz).x;
        }
    }
}

fn prim_normal(pr: Prim, ro: vec3<f32>, rd: vec3<f32>, t: f32) -> vec3<f32> {
    let p = ro + rd * t;
    switch pr.hdr.x {
        case 0u: {
            return normalize(p - pr.a.xyz);
        }
        case 1u: {
            return normalize(capsule_normal(p, pr.a.xyz, pr.b.xyz, pr.a.w));
        }
        default: {
            let lr = box_local_ray(pr, ro, rd);
            let nl = isect_box_local(lr[0], lr[1], pr.b.xyz).yzw;
            let r = vec3<f32>(pr.c.x, 0.0, pr.c.y);
            let f = vec3<f32>(pr.c.z, 0.0, pr.c.w);
            return normalize(r * nl.x + vec3<f32>(0.0, nl.y, 0.0) + f * nl.z);
        }
    }
}

// Nearest prim along the ray. Returns (t, prim index as float bits).
fn isect_prims(ro: vec3<f32>, rd: vec3<f32>, tmax: f32, vis: u32) -> vec2<f32> {
    var best_t = tmax;
    var best_i = NONE;
    let n_objs = G.counts.x;
    for (var oi = 0u; oi < n_objs; oi++) {
        let ob = objs[oi];
        let oc = ro - ob.sphere.xyz;
        let b = dot(oc, rd);
        let c = dot(oc, oc) - ob.sphere.w * ob.sphere.w;
        let disc = b * b - c;
        if disc < 0.0 || (c > 0.0 && b > 0.0) {
            continue;
        }
        if -b - sqrt(disc) > best_t {
            continue;
        }
        let end = ob.range.x + ob.range.y;
        for (var pi = ob.range.x; pi < end; pi++) {
            let pr = prims[pi];
            if (pr.hdr.z & vis) == 0u {
                continue;
            }
            let t = isect_prim(pr, ro, rd);
            if t > EPS && t < best_t {
                best_t = t;
                best_i = pi;
            }
        }
    }
    return vec2<f32>(best_t, bitcast<f32>(best_i));
}

fn occluded_prims(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> bool {
    let n_objs = G.counts.x;
    for (var oi = 0u; oi < n_objs; oi++) {
        let ob = objs[oi];
        let oc = ro - ob.sphere.xyz;
        let b = dot(oc, rd);
        let c = dot(oc, oc) - ob.sphere.w * ob.sphere.w;
        let disc = b * b - c;
        if disc < 0.0 || (c > 0.0 && b > 0.0) {
            continue;
        }
        let end = ob.range.x + ob.range.y;
        for (var pi = ob.range.x; pi < end; pi++) {
            let pr = prims[pi];
            if (pr.hdr.z & 1u) == 0u {
                continue;
            }
            let t = isect_prim(pr, ro, rd);
            if t > EPS && t < tmax {
                return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------- screen board

fn screen_present() -> bool {
    return (G.counts.z & 2u) != 0u;
}

// returns (t, local normal)
fn isect_screen(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let o = ro - G.scr_center.xyz;
    let r = G.scr_right.xyz;
    let u = G.scr_up.xyz;
    let n = G.scr_normal.xyz;
    let lo = vec3<f32>(dot(o, r), dot(o, u), dot(o, n));
    let ld = vec3<f32>(dot(rd, r), dot(rd, u), dot(rd, n));
    return isect_box_local(lo, ld, vec3<f32>(G.scr_center.w, G.scr_right.w, G.scr_up.w));
}

fn screen_radiance(p: vec3<f32>) -> vec3<f32> {
    let n = G.scr_info.z;
    if n == 0u {
        return vec3<f32>(0.02);
    }
    let d = p - G.scr_center.xyz;
    let u = dot(d, G.scr_right.xyz) / G.scr_center.w;
    let v = dot(d, G.scr_up.xyz) / G.scr_right.w;
    let w = G.scr_info.x;
    let h = G.scr_info.y;
    let px = u32(clamp((u * 0.5 + 0.5) * f32(w), 0.0, f32(w - 1u)));
    let py = u32(clamp((0.5 - v * 0.5) * f32(h), 0.0, f32(h - 1u)));
    return screen_img[py * w + px].rgb / f32(n) * G.scr_normal.w;
}

// ---------------------------------------------------------------- lenses

// returns (t, lens index bits, rim flag)
fn isect_lenses(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> vec3<f32> {
    var best_t = tmax;
    var best_i = NONE;
    var rim = 0.0;
    let n = G.counts.y;
    for (var li = 0u; li < n; li++) {
        let L = lenses[li];
        let ax = L.axis.xyz;
        let dn = dot(rd, ax);
        if abs(dn) < 1e-7 {
            continue;
        }
        let t = dot(L.center.xyz - ro, ax) / dn;
        if t <= EPS || t >= best_t {
            continue;
        }
        let p = ro + rd * t;
        let q = p - L.center.xyz;
        let r2 = dot(q, q);
        let R = L.center.w;
        let Rr = R + L.params.z;
        if r2 < Rr * Rr {
            best_t = t;
            best_i = li;
            rim = select(0.0, 1.0, r2 > R * R);
        }
    }
    return vec3<f32>(best_t, bitcast<f32>(best_i), rim);
}

// relative power change with wavelength: 0 at the d line, 1 at the F line
fn chromatic_shape(lam: f32) -> f32 {
    let lf = 486.1;
    let ld = 587.6;
    return (1.0 / (lam * lam) - 1.0 / (ld * ld)) / (1.0 / (lf * lf) - 1.0 / (ld * ld));
}

// Ideal thin lens in slope form: t' = t - h P(h, lambda).
fn lens_refract(L: Lens, d: vec3<f32>, p: vec3<f32>, lam: f32) -> vec3<f32> {
    var n = L.axis.xyz;
    var dn = dot(d, n);
    if dn < 0.0 {
        n = -n;
        dn = -dn;
    }
    let h = p - L.center.xyz;
    let slope = d / dn - n;
    let R = L.center.w;
    let rho2 = dot(h, h) / (R * R);
    var power = L.axis.w * (1.0 + L.params.x * rho2);
    if lam > 0.0 {
        power *= 1.0 + L.params.y * chromatic_shape(lam);
    }
    return normalize(n + slope - h * power);
}

// ---------------------------------------------------------------- prisms

fn prism_local(P: Prism, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(v.x * P.axes.x + v.z * P.axes.y, v.y, v.x * P.axes.z + v.z * P.axes.w);
}

fn prism_world(P: Prism, n: vec3<f32>) -> vec3<f32> {
    let r = vec3<f32>(P.axes.x, 0.0, P.axes.y);
    let f = vec3<f32>(P.axes.z, 0.0, P.axes.w);
    return normalize(r * n.x + vec3<f32>(0.0, n.y, 0.0) + f * n.z);
}

// local outward normal and offset of the i-th bounding plane
fn prism_plane(P: Prism, i: i32) -> vec4<f32> {
    switch i {
        case 0: {
            return vec4<f32>(P.p1.x, 0.0, P.p1.y, P.p1.z);
        }
        case 1: {
            return vec4<f32>(P.p2.x, 0.0, P.p2.y, P.p2.z);
        }
        case 2: {
            return vec4<f32>(P.p3.x, 0.0, P.p3.y, P.p3.z);
        }
        case 3: {
            return vec4<f32>(0.0, 1.0, 0.0, P.center.w);
        }
        default: {
            return vec4<f32>(0.0, -1.0, 0.0, P.center.w);
        }
    }
}

// ray from outside: (t, local normal of the entry face)
fn prism_enter(P: Prism, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let o = prism_local(P, ro - P.center.xyz);
    let d = prism_local(P, rd);
    var tn = -INF;
    var tf = INF;
    var nn = vec3<f32>(0.0);
    for (var i = 0; i < 5; i++) {
        let pl = prism_plane(P, i);
        let denom = dot(pl.xyz, d);
        let dist = pl.w - dot(pl.xyz, o);
        if abs(denom) < 1e-9 {
            if dist < 0.0 {
                return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
            }
            continue;
        }
        let t = dist / denom;
        if denom < 0.0 {
            if t > tn {
                tn = t;
                nn = pl.xyz;
            }
        } else {
            tf = min(tf, t);
        }
    }
    if tn > tf || tn < EPS {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    return vec4<f32>(tn, nn);
}

// ray from inside: (t, local normal of the exit face)
fn prism_exit(P: Prism, ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    let o = prism_local(P, ro - P.center.xyz);
    let d = prism_local(P, rd);
    var tf = INF;
    var nf = vec3<f32>(0.0, 1.0, 0.0);
    for (var i = 0; i < 5; i++) {
        let pl = prism_plane(P, i);
        let denom = dot(pl.xyz, d);
        if denom > 1e-9 {
            let t = (pl.w - dot(pl.xyz, o)) / denom;
            if t < tf {
                tf = t;
                nf = pl.xyz;
            }
        }
    }
    return vec4<f32>(max(tf, 0.0), nf);
}

// (t, prism index bits)
fn isect_prisms(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> vec2<f32> {
    var best_t = tmax;
    var best_i = NONE;
    for (var i = 0u; i < G.counts2.x; i++) {
        let h = prism_enter(prisms[i], ro, rd);
        if h.x > 0.0 && h.x < best_t {
            best_t = h.x;
            best_i = i;
        }
    }
    return vec2<f32>(best_t, bitcast<f32>(best_i));
}

fn prism_index(P: Prism, lam: f32) -> f32 {
    if lam <= 0.0 {
        return P.glass.z;
    }
    let l = lam * 1e-3;
    return P.glass.x + P.glass.y / (l * l);
}

// ---------------------------------------------------------------- spectral colour

fn gauss2(l: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let t = (l - mu) / select(s2, s1, l < mu);
    return exp(-0.5 * t * t);
}

// CIE 1931 colour matching (Wyman et al. 2013 fit) -> linear sRGB
fn cie_rgb(l: f32) -> vec3<f32> {
    let x = 1.056 * gauss2(l, 599.8, 37.9, 31.0) + 0.362 * gauss2(l, 442.0, 16.0, 26.7) - 0.065 * gauss2(l, 501.1, 20.4, 26.2);
    let y = 0.821 * gauss2(l, 568.8, 46.9, 40.5) + 0.286 * gauss2(l, 530.9, 16.3, 31.1);
    let z = 1.217 * gauss2(l, 437.0, 11.8, 36.0) + 0.681 * gauss2(l, 459.0, 26.0, 13.8);
    return vec3<f32>(
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    );
}

// One wavelength sample of an RGB radiance, as seen by the camera.
// The matrix makes sure that the average over wavelengths returns the input RGB.
fn spectral(l: f32, rgb: vec3<f32>) -> vec3<f32> {
    let bb = 1.0 - smoothstep(470.0, 520.0, l);
    let br = smoothstep(565.0, 615.0, l);
    let s = dot(rgb, vec3<f32>(br, 1.0 - bb - br, bb));
    let m = mat3x3<f32>(
        vec3<f32>(2.41892, -0.00496, 0.05875),
        vec3<f32>(-0.15389, 2.94007, 0.14481),
        vec3<f32>(0.05363, 0.0224, 2.87732),
    );
    return m * (cie_rgb(l) * s);
}

// ---------------------------------------------------------------- grass

fn grass_tint(xz: vec2<f32>, cell: vec2<f32>) -> vec3<f32> {
    var c = vec3<f32>(1.0);
    c *= 0.75 + 0.5 * hash21(cell + vec2<f32>(71.0, 3.0));
    if hash21(cell + vec2<f32>(13.0, 5.0)) > 0.86 {
        c *= vec3<f32>(1.35, 1.1, 0.55);
    }
    c *= 0.78 + 0.44 * fbm(xz * 0.45);
    // mowing stripes: a gentle depth cue
    let stripe = (i32(floor(xz.x / 1.5)) & 1) == 0;
    c *= select(0.9, 1.1, stripe);
    return c;
}

fn grass_albedo(xz: vec2<f32>, hf: f32, cell: vec2<f32>) -> vec3<f32> {
    let base = vec3<f32>(0.035, 0.10, 0.015);
    let tip = vec3<f32>(0.16, 0.32, 0.045);
    return mix(base, tip, hf) * grass_tint(xz, cell) * mix(0.3, 1.0, hf);
}

struct Surf {
    t: f32,
    n: vec3<f32>,
    alb: vec3<f32>,
    hit: bool,
};

fn march_grass(ro: vec3<f32>, rd: vec3<f32>, tmax: f32) -> Surf {
    var s: Surf;
    s.hit = false;
    s.t = INF;
    if rd.y > -1e-6 || ro.y <= 0.0 {
        return s;
    }
    let H = G.misc.y;
    let cs = G.misc.z;
    let t_bot = -ro.y / rd.y;
    let t_top = (H - ro.y) / rd.y;
    if max(t_top, 0.0) >= tmax {
        return s;
    }
    if t_bot > 45.0 {
        // far away: the blades are sub-pixel, use their average colour
        if t_bot < tmax {
            let p = ro + rd * t_bot;
            s.hit = true;
            s.t = t_bot;
            s.n = vec3<f32>(0.0, 1.0, 0.0);
            let c = floor(p.xz / cs);
            s.alb = grass_albedo(p.xz, 0.8, c);
        }
        return s;
    }
    let K = 24;
    let jit = rnd();
    for (var i = 0; i < K; i++) {
        let hk = H * (1.0 - (f32(i) + jit) / f32(K));
        if hk >= ro.y {
            continue;
        }
        let t = (hk - ro.y) / rd.y;
        if t >= tmax {
            return s;
        }
        let p = ro + rd * t;
        // the blade grid is rotated so it never lines up with a line of sight along x or z
        let q = vec2<f32>(0.914 * p.x - 0.405 * p.z, 0.405 * p.x + 0.914 * p.z) / cs;
        // blades may stand anywhere in their cell: test the 2x2 cells around q
        let base = floor(q - 0.5);
        for (var k = 0; k < 4; k++) {
            let cell = base + vec2<f32>(f32(k & 1), f32(k >> 1u));
            let hb = H * (0.45 + 0.55 * hash21(cell + vec2<f32>(3.7, 1.1)));
            if hk < hb {
                let hf = hk / hb;
                let c = cell + hash22(cell) + (hash22(cell + vec2<f32>(11.3, 4.1)) - 0.5) * 0.5 * hf * hf;
                let w = 0.31 * (1.0 - hf) + 0.03;
                let dv = q - c;
                if dot(dv, dv) < w * w {
                    s.hit = true;
                    s.t = t;
                    let lat = dv / w;
                    s.n = normalize(vec3<f32>(lat.x * 0.6, 1.0, lat.y * 0.6));
                    s.alb = grass_albedo(p.xz, hf, cell);
                    return s;
                }
            }
        }
    }
    if t_bot < tmax {
        s.hit = true;
        s.t = t_bot;
        s.n = vec3<f32>(0.0, 1.0, 0.0);
        s.alb = vec3<f32>(0.03, 0.025, 0.015);
    }
    return s;
}

// ---------------------------------------------------------------- shading

fn sun_dir_sample() -> vec3<f32> {
    let s = G.sun_dir.xyz;
    let t = basis_x(s);
    let b = cross(s, t);
    let r = G.sun_dir.w * sqrt(rnd());
    let phi = 2.0 * PI * rnd();
    return normalize(s + t * (r * cos(phi)) + b * (r * sin(phi)));
}

fn occluded(p: vec3<f32>, d: vec3<f32>, tmax: f32) -> bool {
    if occluded_prims(p, d, tmax) {
        return true;
    }
    if screen_present() {
        let sh = isect_screen(p, d);
        if sh.x > 0.0 && sh.x < tmax {
            return true;
        }
    }
    return false;
}

fn shade(p: vec3<f32>, n: vec3<f32>, alb: vec3<f32>) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    let l = sun_dir_sample();
    let ndl = dot(n, l);
    let po = p + n * 2e-3;
    if ndl > 0.0 && !occluded(po, l, INF) {
        light += G.sun_col.rgb * ndl;
    }
    let a = cosine_dir(n);
    var amb = sky_gradient(a);
    if a.y < 0.0 {
        amb = vec3<f32>(0.05, 0.08, 0.03);
    }
    if occluded(po, a, 6.0) {
        amb *= 0.12;
    }
    light += amb * G.sun_col.w;
    return alb * light;
}

// ---------------------------------------------------------------- tracing

// returns (radiance, 1 if the path depended on the wavelength)
fn trace(ro_in: vec3<f32>, rd_in: vec3<f32>, vis: u32, lam: f32, skip_screen: bool, glassy: bool) -> vec4<f32> {
    var ro = ro_in;
    var rd = rd_in;
    // "tube": light reaching a screen must have passed all lenses of its optical train
    let need = V.info2.w;
    var passed = 0u;
    var thr = vec3<f32>(1.0);
    var add = vec3<f32>(0.0);
    var dist = 0.0;
    var disp = 0.0;
    for (var bounce = 0; bounce < 16; bounce++) {
        let ph = isect_prims(ro, rd, INF, vis);
        var t = ph.x;
        var kind = 0u; // 0 none, 1 prim, 2 lens, 3 rim, 4 screen, 5 prism
        if bitcast<u32>(ph.y) != NONE {
            kind = 1u;
        }
        let lh = isect_lenses(ro, rd, t);
        if bitcast<u32>(lh.y) != NONE {
            t = lh.x;
            kind = select(2u, 3u, lh.z > 0.5);
        }
        let gh_p = isect_prisms(ro, rd, t);
        if bitcast<u32>(gh_p.y) != NONE {
            t = gh_p.x;
            kind = 5u;
        }
        var sh = vec4<f32>(-1.0);
        if screen_present() && !skip_screen {
            sh = isect_screen(ro, rd);
            if sh.x > 0.0 && sh.x < t {
                t = sh.x;
                kind = 4u;
            }
        }
        let gh = march_grass(ro, rd, t);
        let terminal = gh.hit || (kind != 2u && kind != 5u);
        if terminal && (passed & need) != need {
            return vec4<f32>(0.0, 0.0, 0.0, disp);
        }
        if gh.hit {
            let p = ro + rd * gh.t;
            return vec4<f32>(add + thr * fog(shade(p, gh.n, gh.alb), dist + gh.t), disp);
        }
        switch kind {
            case 0u: {
                return vec4<f32>(add + thr * sky(rd), disp);
            }
            case 1u: {
                let pr = prims[bitcast<u32>(ph.y)];
                let n = prim_normal(pr, ro, rd, t);
                let p = ro + rd * t;
                var alb = unpack_rgb(pr.hdr.y);
                let mat = pr.hdr.w & 255u;
                let param = f32(pr.hdr.w >> 8u);
                if mat == 1u {
                    // glowing surface
                    return vec4<f32>(add + thr * fog(alb * (param * 0.01), dist + t), disp);
                }
                if mat == 2u {
                    // checkerboard on the front face
                    let lr = box_local_ray(pr, ro, rd);
                    let lp = lr[0] + lr[1] * t;
                    if lp.z > pr.b.z * 0.5 {
                        let cell = param * 0.001;
                        let c = i32(floor(lp.x / cell) + floor(lp.y / cell));
                        if (c & 1) == 1 {
                            alb = vec3<f32>(0.015);
                        }
                    } else {
                        alb = vec3<f32>(0.25);
                    }
                }
                return vec4<f32>(add + thr * fog(shade(p, n, alb), dist + t), disp);
            }
            case 2u: {
                let L = lenses[bitcast<u32>(lh.y)];
                let p = ro + rd * t;
                if glassy && L.params.w < 0.5 {
                    // Fresnel reflection + a slight tint so the glass is visible in the overview
                    let n = L.axis.xyz;
                    let c = abs(dot(rd, n));
                    let fr = 0.04 + 0.96 * pow(1.0 - c, 5.0);
                    let rr = reflect(rd, select(n, -n, dot(rd, n) > 0.0));
                    add += thr * fr * sky(rr);
                    thr *= (1.0 - fr) * vec3<f32>(0.80, 0.90, 0.97);
                }
                if L.params.y > 0.0 && lam > 0.0 {
                    disp = 1.0;
                }
                let li = bitcast<u32>(lh.y);
                if li < 32u {
                    passed |= 1u << li;
                }
                rd = lens_refract(L, rd, p, lam);
                ro = p;
                dist += t;
            }
            case 3u: {
                let L = lenses[bitcast<u32>(lh.y)];
                let p = ro + rd * t;
                var n = L.axis.xyz;
                if dot(n, rd) > 0.0 {
                    n = -n;
                }
                let col = select(vec3<f32>(0.05, 0.05, 0.055), vec3<f32>(0.018, 0.018, 0.02), L.params.w > 0.5);
                return vec4<f32>(add + thr * fog(shade(p, n, col), dist + t), disp);
            }
            case 5u: {
                let P = prisms[bitcast<u32>(gh_p.y)];
                let hit = prism_enter(P, ro, rd);
                var p = ro + rd * hit.x;
                let n_in = prism_world(P, hit.yzw);
                let ng = prism_index(P, lam);
                if lam > 0.0 {
                    disp = 1.0;
                }
                if glassy {
                    let c = abs(dot(rd, n_in));
                    let fr = 0.04 + 0.96 * pow(1.0 - c, 5.0);
                    add += thr * fr * sky(reflect(rd, n_in));
                    thr *= (1.0 - fr) * vec3<f32>(0.88, 0.94, 0.98);
                }
                var d = refract(rd, n_in, 1.0 / ng);
                var inside = 0.0;
                for (var k = 0; k < 8; k++) {
                    let ex = prism_exit(P, p, d);
                    p = p + d * ex.x;
                    inside += ex.x;
                    let nw = prism_world(P, ex.yzw);
                    let d2 = refract(d, -nw, ng);
                    if dot(d2, d2) < 0.5 {
                        // total internal reflection
                        d = reflect(d, -nw);
                        continue;
                    }
                    d = d2;
                    break;
                }
                dist += hit.x + inside;
                ro = p;
                rd = normalize(d);
            }
            default: {
                let p = ro + rd * t;
                let nl = sh.yzw;
                if nl.z > 0.5 {
                    // front face: the image formed on the screen + a little ambient
                    let img = screen_radiance(p);
                    return vec4<f32>(add + thr * fog(img + vec3<f32>(0.01), dist + t), disp);
                }
                let n = normalize(G.scr_right.xyz * nl.x + G.scr_up.xyz * nl.y + G.scr_normal.xyz * nl.z);
                return vec4<f32>(add + thr * fog(shade(p, n, vec3<f32>(0.25, 0.25, 0.27)), dist + t), disp);
            }
        }
    }
    if (passed & need) != need {
        return vec4<f32>(0.0, 0.0, 0.0, disp);
    }
    return vec4<f32>(add + thr * sky(rd), disp);
}

fn sample_pixel(px: vec2<u32>) -> vec3<f32> {
    let W = f32(V.info.y);
    let H = f32(V.info.z);
    let sx = ((f32(px.x) + rnd()) / W) * 2.0 - 1.0;
    let sy = 1.0 - ((f32(px.y) + rnd()) / H) * 2.0;
    var lam = 0.0;
    if (G.counts.z & 1u) != 0u {
        lam = 400.0 + 300.0 * rnd();
    }
    let mode = V.info.x;
    let vis = V.info2.z;
    var ro: vec3<f32>;
    var rd: vec3<f32>;
    var w = 1.0;
    if mode == 0u {
        // screen: sample the opening in front of it (lens, aperture or prism)
        let P = V.origin.xyz + V.right.xyz * (sx * V.p0.x) + V.up.xyz * (sy * V.p0.y) + V.fwd.xyz * 2e-3;
        ro = P;
        if V.p1.w > 0.0 {
            let ax = V.p2.xyz;
            let e1 = normalize(cross(ax, vec3<f32>(0.0, 1.0, 0.0)));
            let e2 = cross(e1, ax);
            let r = V.p1.w * sqrt(rnd());
            let phi = 2.0 * PI * rnd();
            let Q = V.p1.xyz + e1 * (r * cos(phi)) + e2 * (r * sin(phi));
            let dv = Q - P;
            let r2 = dot(dv, dv);
            rd = dv * inverseSqrt(r2);
            let cs = dot(rd, V.fwd.xyz);
            if cs <= 0.0 {
                return vec3<f32>(0.0);
            }
            w = cs * abs(dot(rd, ax)) * V.p0.z / r2 * V.p2.w;
        } else {
            rd = cosine_dir(V.fwd.xyz);
        }
    } else if mode == 1u {
        // eye: retina point -> pupil point -> eye lens -> world
        let dr = V.p1.x;
        let Rp = V.origin.xyz - V.fwd.xyz * dr - V.right.xyz * (sx * V.p0.x * dr) - V.up.xyz * (sy * V.p0.y * dr);
        let r = V.p0.z * sqrt(rnd());
        let phi = 2.0 * PI * rnd();
        let h = V.right.xyz * (r * cos(phi)) + V.up.xyz * (r * sin(phi));
        let Pp = V.origin.xyz + h;
        let din = normalize(Pp - Rp);
        let dn = dot(din, V.fwd.xyz);
        let slope = din / dn - V.fwd.xyz;
        rd = normalize(V.fwd.xyz + slope - h * V.p0.w);
        ro = Pp;
    } else {
        rd = normalize(V.fwd.xyz + V.right.xyz * (sx * V.p0.x) + V.up.xyz * (sy * V.p0.y));
        ro = V.origin.xyz;
    }
    let res = trace(ro, rd, vis, lam, mode == 0u, mode == 2u);
    var c = res.xyz * w;
    if res.w > 0.5 {
        c = spectral(lam, c);
    }
    // clamp fireflies (direct views of the sun through a lens)
    return clamp(c, vec3<f32>(-40.0), vec3<f32>(40.0));
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c < vec3<f32>(0.0031308));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let W = V.info.y;
    let H = V.info.z;
    if gid.x >= W || gid.y >= H {
        return;
    }
    let idx = gid.y * W + gid.x;
    rng = pcg(idx ^ pcg(V.info2.y + 0x9e3779b9u));
    let spp = V.info.w;
    var acc = vec3<f32>(0.0);
    for (var s = 0u; s < spp; s++) {
        acc += sample_pixel(gid.xy);
    }
    let base = V.info2.x;
    if base > 0u {
        acc += accum[idx].xyz;
    }
    accum[idx] = vec4<f32>(acc, 1.0);
    let c = max(acc / f32(base + spp) * V.origin.w, vec3<f32>(0.0));
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(to_srgb(aces(c)), 1.0));
}
