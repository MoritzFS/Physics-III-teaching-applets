//! CPU mirror of the GPU tracer, used for the eye's auto focus, the
//! ray diagram in the top view and for picking in the 3D view.

use glam::Vec3;

use crate::scene::{chromatic_shape, GpuPrim, LensGeom, PrismGeom, World, LAMBDA_D, VIS_PHYSICAL};

const EPS: f32 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surf {
    Sky,
    Ground,
    /// element id of the object that was hit
    Object(u32),
    /// lens or aperture hole
    Lens(usize),
    /// lens holder or aperture plate
    Rim(usize),
    Prism(usize),
    ScreenFront,
    ScreenBack,
}

fn isect_sphere(ro: Vec3, rd: Vec3, c: Vec3, r: f32) -> f32 {
    let oc = ro - c;
    let b = oc.dot(rd);
    let cc = oc.dot(oc) - r * r;
    let h = b * b - cc;
    if h < 0.0 {
        return -1.0;
    }
    let s = h.sqrt();
    if -b - s > EPS { -b - s } else { -b + s }
}

fn isect_capsule(ro: Vec3, rd: Vec3, pa: Vec3, pb: Vec3, r: f32) -> f32 {
    let ba = pb - pa;
    let oa = ro - pa;
    let baba = ba.dot(ba);
    let bard = ba.dot(rd);
    let baoa = ba.dot(oa);
    let rdoa = rd.dot(oa);
    let oaoa = oa.dot(oa);
    let a = baba - bard * bard;
    let b = baba * rdoa - baoa * bard;
    let c = baba * oaoa - baoa * baoa - r * r * baba;
    let h = b * b - a * c;
    if a < 1e-9 * baba {
        let t1 = isect_sphere(ro, rd, pa, r);
        let t2 = isect_sphere(ro, rd, pb, r);
        return if t1 > EPS && (t2 <= EPS || t1 < t2) { t1 } else { t2 };
    }
    if h >= 0.0 {
        let t = (-b - h.sqrt()) / a;
        let y = baoa + t * bard;
        if y > 0.0 && y < baba {
            return t;
        }
        let oc = if y <= 0.0 { oa } else { ro - pb };
        let b2 = rd.dot(oc);
        let c2 = oc.dot(oc) - r * r;
        let h2 = b2 * b2 - c2;
        if h2 > 0.0 {
            return -b2 - h2.sqrt();
        }
    }
    -1.0
}

/// returns (t, local normal)
pub fn isect_box_local(ro: Vec3, rd: Vec3, half: Vec3) -> (f32, Vec3) {
    let d = Vec3::new(
        if rd.x.abs() < 1e-8 { 1e-8 } else { rd.x },
        if rd.y.abs() < 1e-8 { 1e-8 } else { rd.y },
        if rd.z.abs() < 1e-8 { 1e-8 } else { rd.z },
    );
    let m = d.recip();
    let n = m * ro;
    let k = m.abs() * half;
    let t1 = -n - k;
    let t2 = -n + k;
    let tn = t1.x.max(t1.y).max(t1.z);
    let tf = t2.x.min(t2.y).min(t2.z);
    if tn > tf || tf < EPS || tn < EPS {
        return (-1.0, Vec3::ZERO);
    }
    let nrm = if tn == t1.x {
        Vec3::new(-d.x.signum(), 0.0, 0.0)
    } else if tn == t1.y {
        Vec3::new(0.0, -d.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, -d.z.signum())
    };
    (tn, nrm)
}

fn isect_prim(p: &GpuPrim, ro: Vec3, rd: Vec3) -> f32 {
    let a = Vec3::new(p.a[0], p.a[1], p.a[2]);
    match p.meta[0] {
        0 => isect_sphere(ro, rd, a, p.a[3]),
        1 => isect_capsule(ro, rd, a, Vec3::new(p.b[0], p.b[1], p.b[2]), p.a[3]),
        _ => {
            let r = Vec3::new(p.c[0], 0.0, p.c[1]);
            let f = Vec3::new(p.c[2], 0.0, p.c[3]);
            let o = ro - a;
            let lo = Vec3::new(o.dot(r), o.y, o.dot(f));
            let ld = Vec3::new(rd.dot(r), rd.y, rd.dot(f));
            isect_box_local(lo, ld, Vec3::new(p.b[0], p.b[1], p.b[2])).0
        }
    }
}

/// Ideal thin lens, identical to `lens_refract` in the shader.
pub fn lens_refract(l: &LensGeom, d: Vec3, p: Vec3, lambda: f32) -> Vec3 {
    let mut n = l.axis;
    let mut dn = d.dot(n);
    if dn < 0.0 {
        n = -n;
        dn = -dn;
    }
    let h = p - l.center;
    let slope = d / dn - n;
    (n + slope - h * local_power(l, p, lambda)).normalize()
}

pub fn local_power(l: &LensGeom, p: Vec3, lambda: f32) -> f32 {
    let rho2 = (p - l.center).length_squared() / (l.radius * l.radius);
    l.power * (1.0 + l.spherical * rho2) * (1.0 + l.chromatic * chromatic_shape(lambda))
}

// ---------------------------------------------------------------- prisms

fn prism_planes(p: &PrismGeom) -> [(Vec3, f32); 5] {
    let side = |i: usize| (Vec3::new(p.planes[i].0.x, 0.0, p.planes[i].0.y), p.planes[i].1);
    [side(0), side(1), side(2), (Vec3::Y, p.half_h), (-Vec3::Y, p.half_h)]
}

fn prism_local(p: &PrismGeom, v: Vec3) -> Vec3 {
    Vec3::new(v.dot(p.right), v.y, v.dot(p.fwd))
}

fn prism_world(p: &PrismGeom, n: Vec3) -> Vec3 {
    (p.right * n.x + Vec3::Y * n.y + p.fwd * n.z).normalize()
}

/// ray from outside: (t, world normal of the entry face)
pub fn prism_enter(p: &PrismGeom, ro: Vec3, rd: Vec3) -> Option<(f32, Vec3)> {
    let o = prism_local(p, ro - p.center);
    let d = prism_local(p, rd);
    let mut tn = f32::NEG_INFINITY;
    let mut tf = f32::INFINITY;
    let mut nn = Vec3::ZERO;
    for (n, off) in prism_planes(p) {
        let denom = n.dot(d);
        let dist = off - n.dot(o);
        if denom.abs() < 1e-9 {
            if dist < 0.0 {
                return None;
            }
            continue;
        }
        let t = dist / denom;
        if denom < 0.0 {
            if t > tn {
                tn = t;
                nn = n;
            }
        } else {
            tf = tf.min(t);
        }
    }
    (tn <= tf && tn >= EPS).then(|| (tn, prism_world(p, nn)))
}

fn prism_exit(p: &PrismGeom, ro: Vec3, rd: Vec3) -> (f32, Vec3) {
    let o = prism_local(p, ro - p.center);
    let d = prism_local(p, rd);
    let mut tf = f32::INFINITY;
    let mut nf = Vec3::Y;
    for (n, off) in prism_planes(p) {
        let denom = n.dot(d);
        if denom > 1e-9 {
            let t = (off - n.dot(o)) / denom;
            if t < tf {
                tf = t;
                nf = n;
            }
        }
    }
    (tf.max(0.0), prism_world(p, nf))
}

/// GLSL-style refraction; `None` on total internal reflection.
fn refract(i: Vec3, n: Vec3, eta: f32) -> Option<Vec3> {
    let c = n.dot(i);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    (k >= 0.0).then(|| eta * i - (eta * c + k.sqrt()) * n)
}

fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - 2.0 * n.dot(i) * n
}

/// Passage through a prism: returns the points where the ray meets a face
/// from inside (exit, or internal reflections) and the outgoing direction.
pub fn prism_pass(p: &PrismGeom, n_in: Vec3, entry: Vec3, d: Vec3, lambda: f32) -> (Vec<Vec3>, Vec3) {
    let ng = p.index(lambda);
    let mut dir = refract(d, n_in, 1.0 / ng).unwrap_or(d);
    let mut pos = entry;
    let mut pts = vec![];
    for _ in 0..8 {
        let (t, nw) = prism_exit(p, pos, dir);
        pos += dir * t;
        pts.push(pos);
        match refract(dir, -nw, ng) {
            Some(out) => return (pts, out.normalize()),
            None => dir = reflect(dir, -nw),
        }
    }
    (pts, dir)
}

impl World {
    /// Nearest surface along a ray (no grass, the ground is the plane y = 0).
    pub fn intersect(&self, ro: Vec3, rd: Vec3, vis: u32, skip_screen: bool) -> (f32, Surf) {
        self.intersect_skip(ro, rd, vis, skip_screen, None)
    }

    /// Like [`Self::intersect`], optionally ignoring one element (e.g. the source of a ray fan).
    pub fn intersect_skip(&self, ro: Vec3, rd: Vec3, vis: u32, skip_screen: bool, skip: Option<u32>) -> (f32, Surf) {
        self.intersect_full(ro, rd, vis, skip_screen, skip, false)
    }

    /// `through_stops`: central stops are ignored (used for the paraxial focus of the chief ray)
    pub fn intersect_full(
        &self,
        ro: Vec3,
        rd: Vec3,
        vis: u32,
        skip_screen: bool,
        skip: Option<u32>,
        through_stops: bool,
    ) -> (f32, Surf) {
        let mut best = f32::INFINITY;
        let mut surf = Surf::Sky;
        for (oi, ob) in self.objs.iter().enumerate() {
            if skip == Some(self.obj_ids[oi]) {
                continue;
            }
            let c = Vec3::new(ob.sphere[0], ob.sphere[1], ob.sphere[2]);
            let oc = ro - c;
            let b = oc.dot(rd);
            let cc = oc.dot(oc) - ob.sphere[3] * ob.sphere[3];
            if b * b - cc < 0.0 || (cc > 0.0 && b > 0.0) {
                continue;
            }
            let start = ob.range[0] as usize;
            let end = start + ob.range[1] as usize;
            for p in &self.prims[start..end] {
                if p.meta[2] & vis == 0 {
                    continue;
                }
                let t = isect_prim(p, ro, rd);
                if t > EPS && t < best {
                    best = t;
                    surf = Surf::Object(self.obj_ids[oi]);
                }
            }
        }
        for (i, l) in self.lenses.iter().enumerate() {
            if through_stops && l.obstruction {
                continue;
            }
            let dn = rd.dot(l.axis);
            if dn.abs() < 1e-7 {
                continue;
            }
            let t = (l.center - ro).dot(l.axis) / dn;
            if t <= EPS || t >= best {
                continue;
            }
            let r2 = (ro + rd * t - l.center).length_squared();
            let rr = l.radius + l.rim;
            if r2 < rr * rr {
                best = t;
                surf = if r2 > l.radius * l.radius { Surf::Rim(i) } else { Surf::Lens(i) };
            }
        }
        for (i, p) in self.prisms.iter().enumerate() {
            if let Some((t, _)) = prism_enter(p, ro, rd) {
                if t < best {
                    best = t;
                    surf = Surf::Prism(i);
                }
            }
        }
        if let (Some(s), false) = (&self.screen, skip_screen) {
            let o = ro - s.center;
            let lo = Vec3::new(o.dot(s.right), o.dot(s.up), o.dot(s.normal));
            let ld = Vec3::new(rd.dot(s.right), rd.dot(s.up), rd.dot(s.normal));
            let (t, n) = isect_box_local(lo, ld, Vec3::new(s.hw, s.hh, s.ht));
            if t > 0.0 && t < best {
                best = t;
                surf = if n.z > 0.5 { Surf::ScreenFront } else { Surf::ScreenBack };
            }
        }
        if rd.y < -1e-6 && ro.y > 0.0 {
            let t = -ro.y / rd.y;
            if t < best {
                best = t;
                surf = Surf::Ground;
            }
        }
        (best, surf)
    }

    /// Follows a ray through all lenses, apertures and prisms until it hits something opaque.
    pub fn trace_path_ex(
        &self,
        ro: Vec3,
        rd: Vec3,
        max_dist: f32,
        skip_screen: bool,
        skip: Option<u32>,
        lambda: f32,
    ) -> Path {
        self.trace_path_full(ro, rd, max_dist, skip_screen, skip, lambda, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn trace_path_full(
        &self,
        ro: Vec3,
        rd: Vec3,
        max_dist: f32,
        skip_screen: bool,
        skip: Option<u32>,
        lambda: f32,
        through_stops: bool,
    ) -> Path {
        let mut path =
            Path { points: vec![ro], dirs: vec![rd], events: vec![], seg_n: vec![], end: Surf::Sky, dispersed: false };
        let mut o = ro;
        let mut d = rd;
        let mut travelled = 0.0;
        for _ in 0..24 {
            let (t, s) = self.intersect_full(o, d, VIS_PHYSICAL, skip_screen, skip, through_stops);
            if !t.is_finite() || travelled + t > max_dist {
                path.points.push(o + d * (max_dist - travelled).max(0.0));
                path.seg_n.push(1.0);
                path.end = Surf::Sky;
                return path;
            }
            let p = o + d * t;
            travelled += t;
            path.points.push(p);
            path.seg_n.push(1.0);
            match s {
                Surf::Lens(i) => {
                    let l = &self.lenses[i];
                    if l.aperture {
                        path.events.push(Event::Stop(i));
                    } else {
                        path.events.push(Event::Lens(i));
                        path.dispersed |= l.chromatic > 0.0;
                        d = lens_refract(l, d, p, lambda);
                    }
                    path.dirs.push(d);
                    o = p;
                }
                Surf::Prism(i) => {
                    let pr = &self.prisms[i];
                    let n_in = prism_enter(pr, o, d).map_or(-d, |h| h.1);
                    let (inner, out) = prism_pass(pr, n_in, p, d, lambda);
                    path.dispersed = true;
                    path.events.push(Event::Glass(i));
                    // segments inside the glass
                    let ng = pr.index(lambda);
                    let mut prev = p;
                    for q in &inner {
                        path.dirs.push((*q - prev).normalize_or_zero());
                        travelled += (*q - prev).length();
                        path.points.push(*q);
                        path.seg_n.push(ng);
                        path.events.push(Event::Glass(i));
                        prev = *q;
                    }
                    d = out;
                    path.dirs.push(d);
                    o = prev;
                }
                other => {
                    path.end = other;
                    return path;
                }
            }
        }
        path
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    Lens(usize),
    /// passing through an aperture
    Stop(usize),
    /// entering, reflecting inside or leaving a prism
    Glass(usize),
}

pub struct Path {
    /// start, every event, end
    pub points: Vec<Vec3>,
    /// direction of each segment
    pub dirs: Vec<Vec3>,
    /// `events[k]` happens at `points[k + 1]`
    pub events: Vec<Event>,
    /// refractive index along segment k (points[k] -> points[k + 1])
    pub seg_n: Vec<f32>,
    pub end: Surf,
    /// the path depends on the wavelength
    pub dispersed: bool,
}

impl Path {
    pub fn segment_lengths(&self) -> Vec<f32> {
        self.points.windows(2).map(|w| (w[1] - w[0]).length()).collect()
    }

    /// index of the last event where the direction changed (lens or prism)
    pub fn last_refraction(&self) -> Option<usize> {
        self.events.iter().rposition(|e| matches!(e, Event::Lens(_) | Event::Glass(_)))
    }
}

/// Result of following the line of sight of an eye / the axis of the screen.
pub struct FocusInfo {
    /// vergence of the light arriving at the start point (D).
    /// 0 = parallel light, negative = diverging, positive = converging.
    pub vergence: f32,
    /// what the chief ray hits in the end
    pub end: Surf,
    pub path: Path,
    /// vergence right after the lens nearest to the start point (towards the start)
    pub vergence_after_first_lens: Option<f32>,
    /// (reduced) distance from the start point to that lens
    pub first_lens_distance: Option<f32>,
}

/// Paraxial vergence of light from the point where the chief ray ends, carried
/// back along the path to the start point. Thin lenses add their power, inside
/// glass the reduced distance d/n is used, flat faces keep the reduced vergence.
pub fn focus_along(world: &World, start: Vec3, dir: Vec3, skip_screen: bool) -> FocusInfo {
    let path = world.trace_path_full(start, dir, 5000.0, skip_screen, None, LAMBDA_D, true);
    let seg = path.segment_lengths();
    let tau: Vec<f32> = seg.iter().zip(&path.seg_n).map(|(d, n)| d / n).collect();
    let n = tau.len();
    let mut verg = if path.end == Surf::Sky { 0.0 } else { -1.0 / tau[n - 1].max(1e-4) };
    let first_lens = path.events.iter().position(|e| matches!(e, Event::Lens(_)));
    let mut after_first = None;
    // walk backwards: event k sits between segment k and k+1
    for k in (0..path.events.len()).rev() {
        if let Event::Lens(li) = path.events[k] {
            verg += local_power(&world.lenses[li], path.points[k + 1], LAMBDA_D);
            if Some(k) == first_lens {
                after_first = Some(verg);
            }
        }
        let denom = 1.0 - tau[k] * verg;
        verg = if denom.abs() < 1e-6 { 1e6 * verg.signum() } else { verg / denom };
    }
    let first_lens_distance = first_lens.map(|k| tau[..=k].iter().sum());
    FocusInfo { vergence: verg, end: path.end, path, vergence_after_first_lens: after_first, first_lens_distance }
}
