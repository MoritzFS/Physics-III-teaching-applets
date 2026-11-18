// Hydrogen orbitals: ray marching of |ψ|² inside a sphere of radius L.
//
// ψ(r) = Σ_k c_k e^(−r/n_k) r^l_k P_rad,k(r) · sin^|m_k|θ P_ang,k(cos θ) · e^(i m_k φ)
// is evaluated analytically (no data texture), with the same polynomial
// coefficients as orbital.rs. A term with kind = 1 is a Gaussian packet
// exp(−|r − r₀|²/(4σ²)) (the state right after a position measurement).
//
// Modes: 0 = cloud (front-to-back compositing; opacity ∝ (|ψ|²/ρ_max)^γ, fading out
// in the outer 18 % of the sphere so that the view's edge does not show,
// colour = phase of ψ), 1 = isosurface |ψ|² = level·ρ_max (lit, phase colour),
// 3 = background only. Cut: the half-space on the eye's side of the plane
// n·r = d is removed; with "cap" the cut face shows ψ on the plane (phase
// colour, brightness |ψ|/max|ψ|), like the 2D slice.

struct Uniforms {
    cam_pos: vec4<f32>,   // xyz eye, w = radius L of the bounding sphere
    cam_fwd: vec4<f32>,   // xyz forward, w = tan(fov / 2)
    cam_right: vec4<f32>, // xyz right, w = number of ray steps across 2L
    cam_up: vec4<f32>,    // xyz up, w = mode
    look: vec4<f32>,      // x = 1/ρ_max, y = exposure, z = gamma, w = iso level (fraction of ρ_max)
    cut: vec4<f32>,       // xyz unit normal of the cut plane, w = offset d
    flags: vec4<f32>,     // x = cut on, y = cap on, z = number of terms, w = opacity of the cut face
    bg: vec4<f32>,        // rgb background (sRGB), w = unused
    terms: array<vec4<f32>, 128>, // 4 vec4 per term, at most 32 terms
};

fn hy_cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn hy_ipow(x: f32, k: i32) -> f32 {
    var r = 1.0;
    for (var i = 0; i < k; i++) {
        r = r * x;
    }
    return r;
}

fn hy_psi(p: vec3<f32>) -> vec2<f32> {
    let r = length(p);
    let rho = length(p.xy);
    var eip = vec2<f32>(1.0, 0.0);
    if (rho > 1e-7) {
        eip = p.xy / rho;
    }
    var cu = 1.0;
    var s = 0.0;
    if (r > 1e-7) {
        cu = p.z / r;
        s = rho / r;
    }
    var acc = vec2<f32>(0.0, 0.0);
    let nt = i32(u.flags.z + 0.5);
    for (var k = 0; k < nt; k++) {
        let t0 = u.terms[4 * k];
        let t1 = u.terms[4 * k + 1];
        if (t0.w > 0.5) {
            let d = p - t1.xyz;
            acc = acc + t0.xy * exp(-dot(d, d) / (4.0 * t1.w * t1.w));
            continue;
        }
        let a = u.terms[4 * k + 2];
        let b = u.terms[4 * k + 3];
        let l = i32(t1.x + 0.5);
        let m = i32(round(t1.y));
        let am = abs(m);
        let radial = exp(-r * t0.z) * hy_ipow(r, l) * (((a.w * r + a.z) * r + a.y) * r + a.x);
        let ang = hy_ipow(s, am) * (((b.w * cu + b.z) * cu + b.y) * cu + b.x);
        var ph = vec2<f32>(1.0, 0.0);
        for (var j = 0; j < am; j++) {
            ph = hy_cmul(ph, eip);
        }
        if (m < 0) {
            ph.y = -ph.y;
        }
        acc = acc + hy_cmul(t0.xy, ph) * (radial * ang);
    }
    return acc;
}

// |ψ|²/ρ_max
fn hy_density(p: vec3<f32>) -> f32 {
    let q = hy_psi(p);
    return dot(q, q) * u.look.x;
}

fn hy_hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hy_arg(q: vec2<f32>) -> f32 {
    return atan2(q.y, q.x);
}

fn shade(uv: vec2<f32>) -> vec4<f32> {
    // (the data texture is not used; reading one texel keeps the binding alive on every backend)
    let unused = textureLoad(qb_data, vec3<i32>(0, 0, 0), 0).x * 0.0;
    let bg = u.bg.rgb + vec3<f32>(unused);
    let aspect = qb_frame.rect_px.z / max(qb_frame.rect_px.w, 1.0);
    let tf = u.cam_fwd.w;
    let sx = (2.0 * uv.x - 1.0) * aspect * tf;
    let sy = (1.0 - 2.0 * uv.y) * tf;
    let dir = normalize(u.cam_fwd.xyz + sx * u.cam_right.xyz + sy * u.cam_up.xyz);
    let ro = u.cam_pos.xyz;
    let big_l = u.cam_pos.w;
    let mode = i32(u.cam_up.w + 0.5);
    let bq = dot(ro, dir);
    let cq = dot(ro, ro) - big_l * big_l;
    let h = bq * bq - cq;
    if (h <= 0.0) {
        return vec4<f32>(bg, 1.0);
    }
    let sq = sqrt(h);
    var t0 = max(-bq - sq, 0.0);
    let t1 = -bq + sq;
    var cap = false;
    if (u.flags.x > 0.5) {
        let nrm = u.cut.xyz;
        let side = dot(ro, nrm) - u.cut.w;
        let sgn = select(-1.0, 1.0, side >= 0.0);
        let on = side * sgn;
        let dn = dot(dir, nrm) * sgn;
        if (dn >= 0.0) {
            return vec4<f32>(bg, 1.0);
        }
        let tp = -on / dn;
        if (tp > t0) {
            t0 = tp;
            cap = u.flags.y > 0.5;
        }
    }
    if (t0 >= t1) {
        return vec4<f32>(bg, 1.0);
    }
    let gam = u.look.z;
    var col = vec3<f32>(0.0, 0.0, 0.0);
    var trans = 1.0;
    // the cut face: ψ on the plane
    if (cap) {
        let p = ro + t0 * dir;
        let q = hy_psi(p);
        let d = dot(q, q) * u.look.x;
        if (mode == 1) {
            if (d >= u.look.w) {
                let v = 0.5 + 0.5 * clamp(pow(d, gam), 0.0, 1.0);
                return vec4<f32>(qb_phase_rgb(hy_arg(q)) * v, 1.0);
            }
        } else {
            // a painted section, like the 2D slice: hue = arg ψ, brightness = |ψ|/max|ψ|;
            // nearly opaque, so the radial and angular nodes show as dark lines
            let b = sqrt(clamp(d, 0.0, 1.0));
            let a = u.flags.w * (1.0 - smoothstep(0.85 * big_l, big_l, length(p)));
            col = col + trans * a * mix(bg, qb_phase_rgb(hy_arg(q)), b);
            trans = trans * (1.0 - a);
        }
    }
    if (mode == 3 || u.look.y <= 0.0) {
        return vec4<f32>(col + trans * bg, 1.0);
    }
    let steps = max(u.cam_right.w, 16.0);
    let ds = 2.0 * big_l / steps;
    let jitter = hy_hash(uv * qb_frame.rect_px.zw);
    if (mode == 1) {
        // isosurface: first crossing of the level, refined by bisection
        let lev = u.look.w;
        var t = t0 + ds * jitter;
        var t_prev = t0;
        var hit = false;
        for (var i = 0; i < 1024; i++) {
            if (t >= t1) {
                break;
            }
            if (hy_density(ro + t * dir) >= lev) {
                hit = true;
                break;
            }
            t_prev = t;
            t = t + ds;
        }
        if (!hit) {
            return vec4<f32>(col + trans * bg, 1.0);
        }
        var lo = t_prev;
        var hi = t;
        for (var i = 0; i < 8; i++) {
            let mid = 0.5 * (lo + hi);
            if (hy_density(ro + mid * dir) >= lev) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let p = ro + hi * dir;
        let e = big_l * 0.003;
        let g = vec3<f32>(
            hy_density(p + vec3<f32>(e, 0.0, 0.0)) - hy_density(p - vec3<f32>(e, 0.0, 0.0)),
            hy_density(p + vec3<f32>(0.0, e, 0.0)) - hy_density(p - vec3<f32>(0.0, e, 0.0)),
            hy_density(p + vec3<f32>(0.0, 0.0, e)) - hy_density(p - vec3<f32>(0.0, 0.0, e)),
        );
        var n = -g / max(length(g), 1e-20);
        if (dot(n, dir) > 0.0) {
            n = -n;
        }
        let key = normalize(-dir + 0.55 * u.cam_up.xyz - 0.45 * u.cam_right.xyz);
        let diff = max(dot(n, key), 0.0);
        let half_v = normalize(key - dir);
        let spec = pow(max(dot(n, half_v), 0.0), 40.0);
        let hue = qb_phase_rgb(hy_arg(hy_psi(p)));
        let lit = hue * (0.22 + 0.78 * diff) + vec3<f32>(0.35 * spec);
        return vec4<f32>(col + trans * lit, 1.0);
    }
    // glowing cloud: front-to-back compositing
    let kappa = u.look.y * 8.0 / big_l;
    var t = t0 + ds * jitter;
    for (var i = 0; i < 1024; i++) {
        if (t >= t1 || trans < 0.004) {
            break;
        }
        let p = ro + t * dir;
        let q = hy_psi(p);
        let d = dot(q, q) * u.look.x;
        if (d > 1e-9) {
            // soft edge: the cloud fades out in the outer 18 % of the view sphere
            let fade = 1.0 - smoothstep(0.82 * big_l, big_l, length(p));
            let a = 1.0 - exp(-kappa * pow(d, gam) * fade * ds);
            col = col + trans * a * qb_phase_rgb(hy_arg(q));
            trans = trans * (1.0 - a);
        }
        t = t + ds;
    }
    return vec4<f32>(col + trans * bg, 1.0);
}
