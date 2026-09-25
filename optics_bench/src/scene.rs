//! Scene description: what is standing on the lawn, and how it turns into
//! GPU primitives.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

pub const UP: Vec3 = Vec3::Y;
/// Default height of the optical axis above the grass (m).
pub const AXIS_H: f32 = 0.95;
/// Retina distance of the reduced eye model (m).
pub const RETINA: f32 = 0.017;

pub const VIS_PHYSICAL: u32 = 1;
pub const VIS_OVERVIEW: u32 = 2;

/// Fraunhofer lines used to define dispersion (nm)
pub const LAMBDA_F: f32 = 486.1;
pub const LAMBDA_D: f32 = 587.6;
pub const LAMBDA_C: f32 = 656.3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjKind {
    Stickman,
    Tree,
    Sign,
    SlitLamp,
    Checker,
}

impl ObjKind {
    pub fn label(self) -> &'static str {
        match self {
            ObjKind::Stickman => "Stickman",
            ObjKind::Tree => "Tree",
            ObjKind::Sign => "F-sign",
            ObjKind::SlitLamp => "Slit lamp",
            ObjKind::Checker => "Checkerboard",
        }
    }
    /// Approximate half width in the object's own right direction (m, before scaling).
    pub fn half_width(self) -> f32 {
        match self {
            ObjKind::Stickman => 0.45,
            ObjKind::Tree => 1.0,
            ObjKind::Sign => 0.35,
            ObjKind::SlitLamp => 0.7,
            ObjKind::Checker => 0.5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LensParams {
    /// focal length in m, negative for a diverging lens
    pub focal: f32,
    pub diameter: f32,
    /// relative power increase at the rim (third order spherical aberration)
    pub spherical: f32,
    /// relative power difference between blue (F line) and yellow (d line)
    pub chromatic: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApertureParams {
    /// diameter of the hole (m)
    pub hole: f32,
    /// diameter of the opaque plate (m)
    pub plate: f32,
}

/// An opaque disk that blocks the centre of a beam ("anti-aperture").
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StopParams {
    pub diameter: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrismParams {
    /// angle between the two refracting faces
    pub apex_deg: f32,
    /// length of the refracting faces (m)
    pub side: f32,
    /// vertical size (m)
    pub height: f32,
    /// refractive index at 587.6 nm
    pub n_d: f32,
    /// Abbe number V = (n_d - 1) / (n_F - n_C); small = strong dispersion
    pub abbe: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScreenParams {
    pub width: f32,
    pub height: f32,
    pub exposure: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeParams {
    pub auto_focus: bool,
    /// current accommodation in dioptres (set by auto focus or by hand)
    pub accommodation: f32,
    /// amplitude of accommodation (dioptres); 4 D = near point 25 cm
    pub max_accommodation: f32,
    /// spectacle prescription (D): negative = myopic, positive = hyperopic
    pub prescription: f32,
    pub pupil_mm: f32,
    pub fov_deg: f32,
    pub pitch_deg: f32,
    pub exposure: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Kind {
    Object { kind: ObjKind, scale: f32 },
    Lens(LensParams),
    Aperture(ApertureParams),
    Stop(StopParams),
    Prism(PrismParams),
    Screen(ScreenParams),
    Eye(EyeParams),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub id: u32,
    /// position on the lawn: (x, z)
    pub pos: Vec2,
    /// height of the centre above ground (lenses, apertures, prisms, screen, eye)
    pub height: f32,
    /// orientation around the vertical axis (rad); facing = (cos, 0, sin)
    pub yaw: f32,
    pub kind: Kind,
}

pub fn facing(yaw: f32) -> Vec3 {
    Vec3::new(yaw.cos(), 0.0, yaw.sin())
}

/// right-hand direction for something facing `facing(yaw)`
pub fn right_of(yaw: f32) -> Vec3 {
    facing(yaw).cross(UP)
}

impl Element {
    pub fn name(&self) -> String {
        match &self.kind {
            Kind::Object { kind, .. } => kind.label().to_string(),
            Kind::Lens(l) => {
                if l.focal >= 0.0 {
                    format!("Converging lens (f = {:.0} cm)", l.focal * 100.0)
                } else {
                    format!("Diverging lens (f = {:.0} cm)", l.focal * 100.0)
                }
            }
            Kind::Aperture(a) => format!("Aperture (Ø {:.1} cm)", a.hole * 100.0),
            Kind::Stop(st) => format!("Central stop (Ø {:.1} cm)", st.diameter * 100.0),
            Kind::Prism(p) => format!("Prism ({:.0}°, n = {:.2})", p.apex_deg, p.n_d),
            Kind::Screen(_) => "Screen".into(),
            Kind::Eye(_) => "Eye".into(),
        }
    }

    pub fn center3(&self) -> Vec3 {
        match &self.kind {
            Kind::Object { kind, scale } => {
                let h = match kind {
                    ObjKind::Stickman => 0.9,
                    ObjKind::Tree => 1.8,
                    ObjKind::Sign => 0.8,
                    ObjKind::SlitLamp | ObjKind::Checker => AXIS_H,
                };
                Vec3::new(self.pos.x, h * scale, self.pos.y)
            }
            _ => Vec3::new(self.pos.x, self.height, self.pos.y),
        }
    }

    pub fn eye_frame(&self) -> Option<(Vec3, Vec3, Vec3, Vec3)> {
        if let Kind::Eye(e) = &self.kind {
            let p = e.pitch_deg.to_radians();
            let f = Vec3::new(p.cos() * self.yaw.cos(), p.sin(), p.cos() * self.yaw.sin());
            let r = f.cross(UP).normalize();
            let u = r.cross(f);
            Some((self.center3(), r, u, f))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------- glass

/// Cauchy coefficients (A, B) with n(λ) = A + B/λ², λ in µm.
pub fn cauchy(n_d: f32, abbe: f32) -> (f32, f32) {
    let inv2 = |l: f32| 1.0 / (l * 1e-3) / (l * 1e-3);
    let b = (n_d - 1.0) / (abbe.max(1.0) * (inv2(LAMBDA_F) - inv2(LAMBDA_C)));
    (n_d - b * inv2(LAMBDA_D), b)
}

pub fn prism_index(p: &PrismParams, lambda_nm: f32) -> f32 {
    let (a, b) = cauchy(p.n_d, p.abbe);
    let l = lambda_nm * 1e-3;
    a + b / (l * l)
}

/// Minimum deviation (rad) of a prism for refractive index n.
pub fn min_deviation(apex_deg: f32, n: f32) -> Option<f32> {
    let half = (apex_deg * 0.5).to_radians();
    let s = n * half.sin();
    (s < 1.0).then(|| 2.0 * s.asin() - 2.0 * half)
}

/// Relative change of lens power with wavelength: 0 at the d line, 1 at the F line.
pub fn chromatic_shape(lambda_nm: f32) -> f32 {
    let inv2 = |l: f32| 1.0 / (l * l);
    (inv2(lambda_nm) - inv2(LAMBDA_D)) / (inv2(LAMBDA_F) - inv2(LAMBDA_D))
}

// ---------------------------------------------------------------- presets

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Preset {
    Projector,
    Magnifier,
    DepthOfField,
    CentralStop,
    Pinhole,
    Kepler,
    Galilei,
    Diverging,
    Myopia,
    Aberrations,
    PrismLook,
    Spectrometer,
}

impl Preset {
    pub const ALL: [Preset; 12] = [
        Preset::Projector,
        Preset::Magnifier,
        Preset::DepthOfField,
        Preset::CentralStop,
        Preset::Pinhole,
        Preset::Kepler,
        Preset::Galilei,
        Preset::Diverging,
        Preset::Myopia,
        Preset::Aberrations,
        Preset::PrismLook,
        Preset::Spectrometer,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Preset::Projector => "Real image on a screen + eye",
            Preset::Magnifier => "Magnifying glass",
            Preset::DepthOfField => "Aperture: depth of field",
            Preset::CentralStop => "Central stop: ring-shaped blur",
            Preset::Pinhole => "Pinhole camera (camera obscura)",
            Preset::Kepler => "Keplerian telescope (eye + screen)",
            Preset::Galilei => "Galilean telescope (eye + screen)",
            Preset::Diverging => "Diverging lens (virtual image)",
            Preset::Myopia => "Short-sighted eye + glasses",
            Preset::Aberrations => "Spherical & chromatic aberration",
            Preset::PrismLook => "Looking through a prism",
            Preset::Spectrometer => "Prism spectrometer",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub elements: Vec<Element>,
    pub next_id: u32,
    pub sun_azimuth_deg: f32,
    pub sun_elevation_deg: f32,
    /// description / instructions shown in the inspector
    #[serde(default)]
    pub notes: String,
}

impl Default for Scene {
    fn default() -> Self {
        Scene::preset(Preset::Projector)
    }
}

pub fn default_lens(focal: f32) -> LensParams {
    LensParams { focal, diameter: 0.5, spherical: 0.0, chromatic: 0.0 }
}

pub fn default_aperture() -> ApertureParams {
    ApertureParams { hole: 0.1, plate: 0.8 }
}

pub fn default_prism() -> PrismParams {
    // dense flint glass
    PrismParams { apex_deg: 60.0, side: 0.4, height: 0.5, n_d: 1.62, abbe: 36.0 }
}

pub fn default_screen() -> ScreenParams {
    ScreenParams { width: 1.2, height: 1.0, exposure: 1.0 }
}

pub fn default_eye() -> EyeParams {
    EyeParams {
        auto_focus: true,
        accommodation: 0.0,
        max_accommodation: 4.0,
        prescription: 0.0,
        pupil_mm: 5.0,
        fov_deg: 60.0,
        pitch_deg: 0.0,
        exposure: 1.0,
    }
}

fn yaw_deg_of(v: Vec2) -> f32 {
    v.y.atan2(v.x).to_degrees()
}

/// Centres of the two refracting faces of a prism at `center` with the given
/// yaw, on the lawn (x, z). At minimum deviation the ray runs from one to the other.
fn prism_faces(p: &PrismParams, center: Vec2, yaw_deg: f32) -> (Vec2, Vec2) {
    let half = (p.apex_deg * 0.5).to_radians();
    let h = p.side * half.cos();
    let zb = p.side * half.sin();
    let x_mid = h / 6.0; // halfway between apex (2h/3) and base (-h/3)
    let yaw = yaw_deg.to_radians();
    let f = Vec2::new(yaw.cos(), yaw.sin());
    let r = Vec2::new(-yaw.sin(), yaw.cos());
    (center + r * x_mid - f * (zb * 0.5), center + r * x_mid + f * (zb * 0.5))
}

impl Scene {
    pub fn empty() -> Self {
        Scene { elements: vec![], next_id: 1, sun_azimuth_deg: 80.0, sun_elevation_deg: 42.0, notes: String::new() }
    }

    pub fn add(&mut self, kind: Kind, x: f32, z: f32, yaw_deg: f32) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let height = match &kind {
            Kind::Object { .. } => 0.0,
            _ => AXIS_H,
        };
        self.elements.push(Element { id, pos: Vec2::new(x, z), height, yaw: yaw_deg.to_radians(), kind });
        id
    }

    pub fn get(&self, id: u32) -> Option<&Element> {
        self.elements.iter().find(|e| e.id == id)
    }

    pub fn get_mut(&mut self, id: u32) -> Option<&mut Element> {
        self.elements.iter_mut().find(|e| e.id == id)
    }

    pub fn remove(&mut self, id: u32) {
        self.elements.retain(|e| e.id != id);
    }

    pub fn screen(&self) -> Option<&Element> {
        self.elements.iter().find(|e| matches!(e.kind, Kind::Screen(_)))
    }

    pub fn eye(&self) -> Option<&Element> {
        self.elements.iter().find(|e| matches!(e.kind, Kind::Eye(_)))
    }

    pub fn eye_mut(&mut self) -> Option<&mut Element> {
        self.elements.iter_mut().find(|e| matches!(e.kind, Kind::Eye(_)))
    }

    pub fn sun_dir(&self) -> Vec3 {
        let az = self.sun_azimuth_deg.to_radians();
        let el = self.sun_elevation_deg.to_radians();
        Vec3::new(el.cos() * az.cos(), el.sin(), el.cos() * az.sin()).normalize()
    }

    /// Two telescopes pointing at a tree 30 m away: row A (z < 0) ends in an eye,
    /// row B (z > 0) projects onto a screen.
    fn telescopes(&mut self, f_obj: f32, f_eyepiece: f32, eyepiece_eye: f32, eye_at: f32, eyepiece_screen: f32, screen_at: f32) {
        let obj = |kind| Kind::Object { kind, scale: 1.0 };
        let tree = Vec2::new(-30.0, 0.0);
        self.add(obj(ObjKind::Tree), tree.x, tree.y, 0.0);
        self.add(obj(ObjKind::Stickman), -22.0, -3.0, 10.0);
        self.add(obj(ObjKind::Sign), -26.0, 3.5, 0.0);
        for (row, z0) in [(0, -0.9f32), (1, 0.9f32)] {
            let origin = Vec2::new(0.0, z0);
            let u = (tree - origin).normalize(); // towards the tree
            let yaw = yaw_deg_of(u);
            let at = |s: f32| origin - u * s;
            let mut objective = default_lens(f_obj);
            objective.diameter = 0.6;
            let p = at(0.0);
            self.add(Kind::Lens(objective), p.x, p.y, yaw);
            let mut eyepiece = default_lens(f_eyepiece);
            eyepiece.diameter = 0.3;
            if row == 0 {
                let p = at(eyepiece_eye);
                self.add(Kind::Lens(eyepiece), p.x, p.y, yaw);
                let mut e = default_eye();
                e.fov_deg = 50.0;
                let p = at(eye_at);
                self.add(Kind::Eye(e), p.x, p.y, yaw);
            } else {
                let p = at(eyepiece_screen);
                self.add(Kind::Lens(eyepiece), p.x, p.y, yaw);
                let p = at(screen_at);
                self.add(Kind::Screen(default_screen()), p.x, p.y, yaw);
            }
        }
    }

    pub fn preset(p: Preset) -> Scene {
        let mut s = Scene::empty();
        let obj = |kind| Kind::Object { kind, scale: 1.0 };
        match p {
            Preset::Projector => {
                // row 1: object at 3 m, f = 1 m -> sharp image 1.5 m behind the lens
                s.add(obj(ObjKind::Stickman), 0.0, -1.4, 0.0);
                s.add(Kind::Lens(default_lens(1.0)), 3.0, -1.4, 0.0);
                s.add(Kind::Screen(default_screen()), 4.52, -1.4, 180.0);
                // row 2: tree seen by the eye
                s.add(obj(ObjKind::Tree), -1.0, 1.6, 0.0);
                s.add(obj(ObjKind::Sign), 1.2, 2.4, 0.0);
                s.add(Kind::Eye(default_eye()), 5.5, 1.6, 180.0);
                s.notes = "The stickman stands 3 m in front of a lens with f = 1 m, so a sharp image forms \
                           1.5 m behind the lens (1/f = 1/g + 1/b), inverted and half as large (m = −b/g). \
                           Move the screen or the stickman and watch the image blur."
                    .into();
            }
            Preset::Magnifier => {
                // sign inside the focal length: upright, magnified virtual image
                s.add(obj(ObjKind::Sign), 0.0, 0.0, 0.0);
                let mut l = default_lens(1.0);
                l.diameter = 0.6;
                s.add(Kind::Lens(l), 0.75, 0.0, 0.0);
                s.add(Kind::Eye(default_eye()), 1.0, 0.0, 180.0);
                s.add(obj(ObjKind::Tree), -6.0, 3.0, 0.0);
                s.add(obj(ObjKind::Stickman), -4.0, -2.0, 30.0);
                s.notes = "The sign is inside the focal length (g = 0.75 m < f = 1 m): the lens forms an upright, \
                           magnified virtual image 3 m in front of it, which the eye can focus on. \
                           Move the sign beyond F and the image flips."
                    .into();
            }
            Preset::DepthOfField => {
                let mut l = default_lens(0.5);
                l.diameter = 0.4;
                s.add(Kind::Lens(l), 0.0, 0.0, 0.0);
                s.add(Kind::Aperture(default_aperture()), -0.04, 0.0, 0.0);
                s.add(obj(ObjKind::Sign), -1.5, 0.3, 0.0);
                s.add(obj(ObjKind::Stickman), -3.0, -0.4, 0.0);
                s.add(obj(ObjKind::Tree), -8.0, 1.2, 0.0);
                s.add(obj(ObjKind::Checker), -5.0, -2.2, 20.0);
                s.add(Kind::Screen(default_screen()), 0.775, 0.0, 180.0);
                s.notes = "The screen is focused on the F-sign (1.5 m). The stickman (3 m) and the tree (8 m) \
                           are out of focus. Select the aperture and make its hole smaller: the blur circles \
                           shrink (larger f-number N = f/D) and the depth of field grows. \
                           (Exposure is automatic, a real image would get darker.)"
                    .into();
            }
            Preset::CentralStop => {
                let mut l = default_lens(0.5);
                l.diameter = 0.4;
                s.add(Kind::Lens(l), 0.0, 0.0, 0.0);
                s.add(Kind::Stop(StopParams { diameter: 0.28 }), -0.04, 0.0, 0.0);
                s.add(obj(ObjKind::Sign), -1.5, 0.3, 0.0);
                s.add(obj(ObjKind::Stickman), -3.0, -0.4, 0.0);
                s.add(obj(ObjKind::Tree), -8.0, 1.2, 0.0);
                s.add(Kind::Screen(default_screen()), 0.775, 0.0, 180.0);
                s.notes = "Like 'depth of field', but the centre of the lens is blocked instead of the rim \
                           (as by the secondary mirror of a reflecting telescope). The focused F-sign stays sharp, \
                           only dimmer. Out-of-focus points become rings instead of discs, so the stickman and \
                           the tree get double contours. Note: in ray optics this is not a filter for fine \
                           details — that needs diffraction (Fourier optics)."
                    .into();
            }
            Preset::Pinhole => {
                s.add(obj(ObjKind::Stickman), -2.0, 0.0, 0.0);
                s.add(obj(ObjKind::Tree), -6.0, 1.3, 0.0);
                s.add(obj(ObjKind::Sign), -3.5, -1.3, 10.0);
                s.add(Kind::Aperture(ApertureParams { hole: 0.015, plate: 1.2 }), 0.0, 0.0, 0.0);
                s.add(Kind::Screen(default_screen()), 1.0, 0.0, 180.0);
                s.notes = "A camera obscura: no lens, just a small hole. Every point of the screen sees the scene \
                           through the hole, so the image is always 'in focus' but blurred by the hole size. \
                           Try a larger hole (brighter but blurrier) and move the screen: \
                           the image just gets bigger."
                    .into();
            }
            Preset::Kepler => {
                // objective f = 2 m, eyepiece f = 0.25 m (M = -8); projection: eyepiece 0.35 m behind the image
                s.telescopes(2.0, 0.25, 2.25, 2.53, 2.493, 3.388);
                s.notes = "Two identical Keplerian telescopes (f_obj = 2 m, f_ep = 25 cm) aimed at the tree. \
                           Upper row: afocal, the eye sees an inverted image magnified 8×. \
                           Lower row: the eyepiece is pulled out a little and projects a real image \
                           onto the screen (how you would photograph the sun)."
                    .into();
            }
            Preset::Galilei => {
                // objective f = 2 m, eyepiece f = -0.5 m (M = +4); projection: eyepiece closer to the image
                s.telescopes(2.0, -0.5, 1.5, 1.6, 1.85, 2.578);
                s.notes = "Two Galilean telescopes (f_obj = 2 m, f_ep = −50 cm). Upper row: the eye sees an \
                           upright image magnified 4×. Lower row: with the eyepiece a bit further out, the \
                           converging light still forms a real image on the screen — this is how a telephoto \
                           lens works."
                    .into();
            }
            Preset::Diverging => {
                s.add(obj(ObjKind::Stickman), 0.0, 0.0, 0.0);
                let mut l = default_lens(-1.0);
                l.diameter = 0.7;
                s.add(Kind::Lens(l), 2.0, 0.0, 0.0);
                s.add(Kind::Eye(default_eye()), 3.5, 0.0, 180.0);
                s.add(obj(ObjKind::Tree), -3.0, 2.5, 0.0);
                s.notes = "A diverging lens always forms an upright, smaller, virtual image between the object \
                           and the lens. Turn on 'virtual rays' in the top view to see where it is."
                    .into();
            }
            Preset::Myopia => {
                s.add(obj(ObjKind::Sign), 0.0, 0.0, 0.0);
                s.add(obj(ObjKind::Tree), -4.0, 1.5, 0.0);
                s.add(obj(ObjKind::Stickman), -2.0, -1.2, 20.0);
                let mut e = default_eye();
                e.prescription = -2.0;
                s.add(Kind::Eye(e), 4.0, 0.0, 180.0);
                // glasses: f = -0.5 m just in front of the eye
                let mut l = default_lens(-0.5);
                l.diameter = 0.12;
                s.add(Kind::Lens(l), 3.98, 0.0, 0.0);
                s.notes = "This eye is short-sighted by 2 D: its far point is at 50 cm. The glasses (−2 D) make \
                           distant objects appear to be at the far point. Delete the glasses to see the blur."
                    .into();
            }
            Preset::Aberrations => {
                s.add(obj(ObjKind::Sign), 0.0, -1.2, 0.0);
                let mut l = default_lens(0.8);
                l.diameter = 0.9;
                l.spherical = 0.25;
                l.chromatic = 0.04;
                s.add(Kind::Lens(l), 2.4, -1.2, 0.0);
                s.add(Kind::Screen(default_screen()), 3.6, -1.2, 180.0);
                s.add(obj(ObjKind::Tree), -2.0, 1.8, 0.0);
                s.add(Kind::Eye(default_eye()), 5.0, 1.0, 190.0);
                s.notes = "A large lens with spherical aberration (the rim focuses shorter than the centre) and \
                           chromatic aberration (blue focuses shorter than red). Put an aperture in front of the \
                           lens to reduce both halos."
                    .into();
            }
            Preset::PrismLook => {
                let prism = default_prism();
                let n = prism.n_d;
                let delta = min_deviation(prism.apex_deg, n).unwrap_or(0.8);
                let (face_in, face_out) = prism_faces(&prism, Vec2::ZERO, 0.0);
                s.add(Kind::Prism(prism), 0.0, 0.0, 0.0);
                // light enters along d_in and leaves along d_out (minimum deviation)
                let d_in = Vec2::new((delta * 0.5).cos(), (delta * 0.5).sin());
                let d_out = Vec2::new((delta * 0.5).cos(), -(delta * 0.5).sin());
                let lamp = face_in - d_in * 3.0;
                s.add(obj(ObjKind::SlitLamp), lamp.x, lamp.y, yaw_deg_of(d_in));
                let side = Vec2::new(-d_in.y, d_in.x);
                let board = face_in - d_in * 3.6 - side * 1.0;
                s.add(obj(ObjKind::Checker), board.x, board.y, yaw_deg_of(d_in));
                let man = face_in - d_in * 4.5 + side * 1.4;
                s.add(obj(ObjKind::Stickman), man.x, man.y, yaw_deg_of(d_in));
                let eye = face_out + d_out * 0.6;
                let mut e = default_eye();
                e.fov_deg = 50.0;
                s.add(Kind::Eye(e), eye.x, eye.y, yaw_deg_of(-d_out));
                s.notes = "Looking through a 60° flint glass prism at a slit lamp: the prism bends the light \
                           towards its base, and blue more than red, so the white slit becomes a spectrum and \
                           every black/white edge gets coloured fringes. Try the glass presets of the prism."
                    .into();
            }
            Preset::Spectrometer => {
                let mut prism = default_prism();
                prism.abbe = 18.0;
                let delta = min_deviation(prism.apex_deg, prism.n_d).unwrap_or(0.8);
                s.add(obj(ObjKind::SlitLamp), 0.0, 0.0, 0.0);
                let mut l = default_lens(0.5);
                l.diameter = 0.3;
                s.add(Kind::Lens(l), 1.0, 0.0, 0.0);
                // prism at minimum deviation for light travelling along +x,
                // the beam enters in the middle of the first face (x = 1.3)
                let yaw = -(delta * 0.5).to_degrees();
                let (face_in, _) = prism_faces(&prism, Vec2::ZERO, yaw);
                let prism_at = Vec2::new(1.3, 0.0) - face_in;
                let (_, face_out) = prism_faces(&prism, prism_at, yaw);
                let inside = (prism.side * (prism.apex_deg * 0.5).to_radians().sin()) / prism.n_d;
                s.add(Kind::Prism(prism), prism_at.x, prism_at.y, yaw);
                let d_out = Vec2::new(delta.cos(), -delta.sin());
                // 2f - 2f imaging: 1 m of (reduced) path from the lens to the screen
                let scr = face_out + d_out * (1.0 - 0.3 - inside + 0.055);
                s.add(Kind::Screen(default_screen()), scr.x, scr.y, yaw_deg_of(-d_out));
                s.add(obj(ObjKind::Tree), -3.0, 2.5, 0.0);
                s.notes = "Newton's experiment as a spectrometer: the lens images the slit onto the screen \
                           (2f – 2f), the prism in between spreads the image by wavelength. \
                           The dispersion of the glass is exaggerated (Abbe number 18)."
                    .into();
            }
        }
        s
    }
}

// ---------------------------------------------------------------- GPU data

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuPrim {
    /// kind (0 sphere, 1 capsule, 2 box), packed rgb, visibility, material
    pub meta: [u32; 4],
    pub a: [f32; 4],
    pub b: [f32; 4],
    pub c: [f32; 4],
}

pub const MAT_DIFFUSE: u32 = 0;
pub const MAT_EMISSIVE: u32 = 1;
pub const MAT_CHECKER: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuObj {
    pub sphere: [f32; 4],
    pub range: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuLens {
    pub center: [f32; 4],
    pub axis: [f32; 4],
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuPrism {
    /// centroid (mid height), w = half height
    pub center: [f32; 4],
    /// right.x, right.z, fwd.x, fwd.z
    pub axes: [f32; 4],
    /// the three side planes in local (x, z): normal.x, normal.z, offset
    pub p1: [f32; 4],
    pub p2: [f32; 4],
    pub p3: [f32; 4],
    /// Cauchy A, B, n_d, unused
    pub glass: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct ScreenGeom {
    pub id: u32,
    pub center: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub normal: Vec3,
    pub hw: f32,
    pub hh: f32,
    pub ht: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LensGeom {
    pub id: u32,
    pub center: Vec3,
    pub axis: Vec3,
    pub radius: f32,
    pub power: f32,
    pub spherical: f32,
    pub chromatic: f32,
    pub rim: f32,
    /// an aperture is a "lens" without power whose rim is the opaque plate
    pub aperture: bool,
    /// a central stop: an aperture with no hole, only the opaque disk
    pub obstruction: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PrismGeom {
    pub id: u32,
    pub center: Vec3,
    pub right: Vec3,
    pub fwd: Vec3,
    pub half_h: f32,
    /// side planes in local (x = right, z = fwd): outward normal and offset
    pub planes: [(Vec2, f32); 3],
    pub cauchy: (f32, f32),
    pub n_d: f32,
    pub inradius: f32,
    /// triangle corners in local (x, z): apex, base -z, base +z
    pub corners: [Vec2; 3],
}

impl PrismGeom {
    pub fn index(&self, lambda_nm: f32) -> f32 {
        let l = lambda_nm * 1e-3;
        self.cauchy.0 + self.cauchy.1 / (l * l)
    }
    pub fn to_world(&self, local: Vec2) -> Vec3 {
        self.center + self.right * local.x + self.fwd * local.y
    }
}

/// Everything the tracers (GPU and CPU) need.
#[derive(Clone, Default)]
pub struct World {
    pub prims: Vec<GpuPrim>,
    pub objs: Vec<GpuObj>,
    /// element id for every entry of `objs`
    pub obj_ids: Vec<u32>,
    pub lenses: Vec<LensGeom>,
    pub prisms: Vec<PrismGeom>,
    pub screen: Option<ScreenGeom>,
}

impl World {
    pub fn gpu_lenses(&self) -> Vec<GpuLens> {
        self.lenses
            .iter()
            .map(|l| GpuLens {
                center: [l.center.x, l.center.y, l.center.z, l.radius],
                axis: [l.axis.x, l.axis.y, l.axis.z, l.power],
                params: [l.spherical, l.chromatic, l.rim, if l.aperture { 1.0 } else { 0.0 }],
            })
            .collect()
    }

    pub fn gpu_prisms(&self) -> Vec<GpuPrism> {
        self.prisms
            .iter()
            .map(|p| {
                let pl = |i: usize| [p.planes[i].0.x, p.planes[i].0.y, p.planes[i].1, 0.0];
                GpuPrism {
                    center: [p.center.x, p.center.y, p.center.z, p.half_h],
                    axes: [p.right.x, p.right.z, p.fwd.x, p.fwd.z],
                    p1: pl(0),
                    p2: pl(1),
                    p3: pl(2),
                    glass: [p.cauchy.0, p.cauchy.1, p.n_d, 0.0],
                }
            })
            .collect()
    }

    /// true if some element makes the light's path depend on its wavelength
    pub fn dispersive(&self) -> bool {
        !self.prisms.is_empty() || self.lenses.iter().any(|l| l.chromatic > 0.0)
    }
}

fn pack(c: [f32; 3]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    q(c[0]) | (q(c[1]) << 8) | (q(c[2]) << 16)
}

/// Builds primitives in an object's local frame: x = right, y = up, z = forward.
struct Builder<'a> {
    prims: &'a mut Vec<GpuPrim>,
    origin: Vec3,
    r: Vec3,
    f: Vec3,
    s: f32,
    vis: u32,
}

impl Builder<'_> {
    fn w(&self, l: Vec3) -> Vec3 {
        self.origin + self.r * (l.x * self.s) + UP * (l.y * self.s) + self.f * (l.z * self.s)
    }
    fn sphere(&mut self, c: Vec3, rad: f32, col: [f32; 3]) {
        let c = self.w(c);
        self.prims.push(GpuPrim {
            meta: [0, pack(col), self.vis, MAT_DIFFUSE],
            a: [c.x, c.y, c.z, rad * self.s],
            ..Default::default()
        });
    }
    fn capsule(&mut self, a: Vec3, b: Vec3, rad: f32, col: [f32; 3]) {
        let a = self.w(a);
        let b = self.w(b);
        self.prims.push(GpuPrim {
            meta: [1, pack(col), self.vis, MAT_DIFFUSE],
            a: [a.x, a.y, a.z, rad * self.s],
            b: [b.x, b.y, b.z, 0.0],
            ..Default::default()
        });
    }
    fn cuboid_mat(&mut self, c: Vec3, half: Vec3, col: [f32; 3], material: u32) {
        let c = self.w(c);
        let h = half * self.s;
        self.prims.push(GpuPrim {
            meta: [2, pack(col), self.vis, material],
            a: [c.x, c.y, c.z, 0.0],
            b: [h.x, h.y, h.z, 0.0],
            c: [self.r.x, self.r.z, self.f.x, self.f.z],
        });
    }
    fn cuboid(&mut self, c: Vec3, half: Vec3, col: [f32; 3]) {
        self.cuboid_mat(c, half, col, MAT_DIFFUSE);
    }
    /// glowing box, `strength` = emitted radiance relative to the colour
    fn emitter(&mut self, c: Vec3, half: Vec3, col: [f32; 3], strength: f32) {
        self.cuboid_mat(c, half, col, MAT_EMISSIVE | (((strength * 100.0) as u32) << 8));
    }
    /// black/white checkerboard on the front (+z) face
    fn checker(&mut self, c: Vec3, half: Vec3, cell: f32) {
        let cell_mm = (cell * self.s * 1000.0) as u32;
        self.cuboid_mat(c, half, [0.8, 0.8, 0.78], MAT_CHECKER | (cell_mm << 8));
    }
}

fn v(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

fn build_stickman(b: &mut Builder) {
    let skin = [0.85, 0.62, 0.38];
    let ink = [0.03, 0.03, 0.035];
    let red = [0.75, 0.05, 0.03];
    let blue = [0.04, 0.12, 0.75];
    // head with eyes and mouth (he faces +z local)
    b.sphere(v(0.0, 1.62, 0.0), 0.13, skin);
    b.sphere(v(0.045, 1.66, 0.112), 0.022, ink);
    b.sphere(v(-0.045, 1.66, 0.112), 0.022, ink);
    b.capsule(v(-0.04, 1.575, 0.118), v(0.04, 1.575, 0.118), 0.012, [0.6, 0.05, 0.05]);
    // a little yellow hat so up/down is obvious
    b.cuboid(v(0.0, 1.76, 0.0), v(0.1, 0.035, 0.1), [0.85, 0.65, 0.02]);
    b.cuboid(v(0.0, 1.735, 0.05), v(0.1, 0.012, 0.1), [0.85, 0.65, 0.02]);
    // torso
    b.capsule(v(0.0, 1.48, 0.0), v(0.0, 0.95, 0.0), 0.05, ink);
    // right arm (blue) hangs down, left arm (red) waves
    b.capsule(v(0.03, 1.40, 0.0), v(0.28, 1.12, 0.03), 0.04, blue);
    b.capsule(v(0.28, 1.12, 0.03), v(0.42, 0.88, 0.06), 0.04, blue);
    b.sphere(v(0.43, 0.86, 0.06), 0.055, skin);
    b.capsule(v(-0.03, 1.40, 0.0), v(-0.33, 1.52, 0.0), 0.04, red);
    b.capsule(v(-0.33, 1.52, 0.0), v(-0.42, 1.88, 0.02), 0.04, red);
    b.sphere(v(-0.425, 1.92, 0.02), 0.055, skin);
    // legs + feet
    b.capsule(v(0.02, 0.95, 0.0), v(0.2, 0.06, 0.0), 0.045, blue);
    b.capsule(v(-0.02, 0.95, 0.0), v(-0.2, 0.06, 0.0), 0.045, red);
    b.capsule(v(0.2, 0.05, 0.0), v(0.22, 0.05, 0.13), 0.045, ink);
    b.capsule(v(-0.2, 0.05, 0.0), v(-0.22, 0.05, 0.13), 0.045, ink);
}

fn build_tree(b: &mut Builder) {
    let bark = [0.18, 0.09, 0.035];
    let leaf1 = [0.03, 0.14, 0.03];
    let leaf2 = [0.05, 0.20, 0.03];
    let leaf3 = [0.09, 0.24, 0.02];
    let apple = [0.70, 0.03, 0.02];
    b.capsule(v(0.0, -0.1, 0.0), v(0.0, 2.0, 0.0), 0.14, bark);
    b.capsule(v(0.0, 1.5, 0.0), v(0.55, 2.15, 0.1), 0.07, bark);
    b.capsule(v(0.0, 1.7, 0.0), v(-0.5, 2.3, -0.1), 0.07, bark);
    b.sphere(v(0.0, 2.55, 0.0), 0.85, leaf2);
    b.sphere(v(0.65, 2.25, 0.15), 0.6, leaf1);
    b.sphere(v(-0.6, 2.4, -0.15), 0.62, leaf3);
    b.sphere(v(0.15, 3.2, -0.05), 0.6, leaf3);
    b.sphere(v(-0.2, 2.2, 0.55), 0.5, leaf1);
    // apples mostly on the right side: asymmetric on purpose
    for (i, p) in [
        v(0.75, 2.0, 0.62),
        v(1.12, 2.5, 0.3),
        v(0.5, 2.9, 0.62),
        v(0.3, 2.35, 0.93),
        v(1.2, 2.05, -0.1),
        v(-0.55, 2.0, 0.68),
    ]
    .iter()
    .enumerate()
    {
        let r = if i == 3 { 0.1 } else { 0.08 };
        b.sphere(*p, r, apple);
    }
}

fn build_sign(b: &mut Builder) {
    let post = [0.25, 0.25, 0.27];
    let board = [0.8, 0.8, 0.78];
    let letter = [0.75, 0.04, 0.02];
    b.capsule(v(0.0, 0.0, -0.04), v(0.0, 0.8, -0.04), 0.035, post);
    b.cuboid(v(0.0, 1.15, 0.0), v(0.36, 0.42, 0.02), board);
    // the letter F on the front (+z) side, readable for someone facing the sign:
    // their right-hand side is -x in the sign's frame.
    let z = 0.028;
    b.cuboid(v(0.14, 1.15, z), v(0.055, 0.32, 0.01), letter);
    b.cuboid(v(-0.035, 1.415, z), v(0.23, 0.055, 0.01), letter);
    b.cuboid(v(0.0, 1.16, z), v(0.19, 0.05, 0.01), letter);
    // a small blue dot in the upper-right corner of the board (seen from the front)
    b.sphere(v(-0.27, 1.49, 0.02), 0.04, [0.05, 0.2, 0.8]);
}

fn build_slit_lamp(b: &mut Builder) {
    let black = [0.012, 0.012, 0.014];
    let steel = [0.12, 0.12, 0.13];
    // a big black board so that the lamp fills the field of a spectrometer
    b.cuboid(v(0.0, AXIS_H, 0.0), v(0.7, 0.6, 0.03), black);
    // the glowing vertical slit on the front
    b.emitter(v(0.0, AXIS_H, 0.032), v(0.006, 0.3, 0.004), [1.0, 1.0, 1.0], 9.0);
    for x in [-0.55, 0.55] {
        b.capsule(v(x, 0.0, -0.05), v(x, AXIS_H - 0.6, -0.05), 0.03, steel);
    }
}

fn build_checker(b: &mut Builder) {
    let steel = [0.12, 0.12, 0.13];
    b.checker(v(0.0, AXIS_H, 0.0), v(0.5, 0.5, 0.015), 0.125);
    for x in [-0.4, 0.4] {
        b.capsule(v(x, 0.0, -0.04), v(x, AXIS_H - 0.5, -0.04), 0.025, steel);
    }
}

fn prism_geom(e: &Element, p: &PrismParams) -> PrismGeom {
    let half = (p.apex_deg * 0.5).to_radians();
    let h = p.side * half.cos(); // apex to base
    let zb = p.side * half.sin(); // half width of the base
    let xa = 2.0 * h / 3.0;
    let xb = -h / 3.0;
    let n1 = Vec2::new(half.sin(), -half.cos());
    let n2 = Vec2::new(half.sin(), half.cos());
    let n3 = Vec2::new(-1.0, 0.0);
    PrismGeom {
        id: e.id,
        center: e.center3(),
        right: right_of(e.yaw),
        fwd: facing(e.yaw),
        half_h: p.height * 0.5,
        planes: [(n1, n1.x * xa), (n2, n2.x * xa), (n3, -xb)],
        cauchy: cauchy(p.n_d, p.abbe),
        n_d: p.n_d,
        inradius: zb * h / (p.side + zb),
        corners: [Vec2::new(xa, 0.0), Vec2::new(xb, -zb), Vec2::new(xb, zb)],
    }
}

fn bounding_sphere(prims: &[GpuPrim]) -> [f32; 4] {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    let mut grow = |c: Vec3, r: Vec3| {
        lo = lo.min(c - r);
        hi = hi.max(c + r);
    };
    for p in prims {
        let a = Vec3::new(p.a[0], p.a[1], p.a[2]);
        match p.meta[0] {
            0 => grow(a, Vec3::splat(p.a[3])),
            1 => {
                let b = Vec3::new(p.b[0], p.b[1], p.b[2]);
                grow(a, Vec3::splat(p.a[3]));
                grow(b, Vec3::splat(p.a[3]));
            }
            _ => grow(a, Vec3::splat(Vec3::new(p.b[0], p.b[1], p.b[2]).length())),
        }
    }
    let c = (lo + hi) * 0.5;
    let r = (hi - lo).length() * 0.5 + 1e-3;
    [c.x, c.y, c.z, r]
}

pub fn build_world(scene: &Scene) -> World {
    let mut w = World::default();
    let steel = [0.12, 0.12, 0.13];
    for e in &scene.elements {
        let start = w.prims.len();
        let f = facing(e.yaw);
        let r = right_of(e.yaw);
        let ground = Vec3::new(e.pos.x, 0.0, e.pos.y);
        let mut b = Builder { prims: &mut w.prims, origin: ground, r, f, s: 1.0, vis: VIS_PHYSICAL | VIS_OVERVIEW };
        match &e.kind {
            Kind::Object { kind, scale } => {
                b.s = *scale;
                match kind {
                    ObjKind::Stickman => build_stickman(&mut b),
                    ObjKind::Tree => build_tree(&mut b),
                    ObjKind::Sign => build_sign(&mut b),
                    ObjKind::SlitLamp => build_slit_lamp(&mut b),
                    ObjKind::Checker => build_checker(&mut b),
                }
            }
            Kind::Lens(l) => {
                let radius = l.diameter * 0.5;
                let rim = (0.02 + 0.04 * radius).min(0.05);
                let bottom = e.height - radius - rim;
                if bottom > 0.02 {
                    b.capsule(v(0.0, 0.0, 0.0), v(0.0, bottom, 0.0), 0.02, steel);
                    b.cuboid(v(0.0, 0.02, 0.0), v(0.12, 0.02, 0.12), steel);
                }
                w.lenses.push(LensGeom {
                    id: e.id,
                    center: e.center3(),
                    axis: f,
                    radius,
                    power: 1.0 / l.focal,
                    spherical: l.spherical,
                    chromatic: l.chromatic,
                    rim,
                    aperture: false,
                    obstruction: false,
                });
            }
            Kind::Aperture(a) => {
                let radius = a.hole * 0.5;
                let outer = (a.plate * 0.5).max(radius + 0.02);
                let bottom = e.height - outer;
                if bottom > 0.02 {
                    b.capsule(v(0.0, 0.0, 0.0), v(0.0, bottom, 0.0), 0.02, steel);
                    b.cuboid(v(0.0, 0.02, 0.0), v(0.12, 0.02, 0.12), steel);
                }
                w.lenses.push(LensGeom {
                    id: e.id,
                    center: e.center3(),
                    axis: f,
                    radius,
                    power: 0.0,
                    spherical: 0.0,
                    chromatic: 0.0,
                    rim: outer - radius,
                    aperture: true,
                    obstruction: false,
                });
            }
            Kind::Stop(st) => {
                let outer = st.diameter * 0.5;
                let bottom = e.height - outer;
                if bottom > 0.02 {
                    b.capsule(v(0.0, 0.0, 0.0), v(0.0, bottom, 0.0), 0.01, steel);
                    b.cuboid(v(0.0, 0.02, 0.0), v(0.12, 0.02, 0.12), steel);
                }
                w.lenses.push(LensGeom {
                    id: e.id,
                    center: e.center3(),
                    axis: f,
                    radius: 0.0,
                    power: 0.0,
                    spherical: 0.0,
                    chromatic: 0.0,
                    rim: outer,
                    aperture: true,
                    obstruction: true,
                });
            }
            Kind::Prism(p) => {
                let bottom = e.height - p.height * 0.5;
                if bottom > 0.02 {
                    b.capsule(v(0.0, 0.0, 0.0), v(0.0, bottom, 0.0), 0.02, steel);
                    b.cuboid(v(0.0, 0.02, 0.0), v(0.12, 0.02, 0.12), steel);
                }
                w.prisms.push(prism_geom(e, p));
            }
            Kind::Screen(s) => {
                let bottom = e.height - s.height * 0.5;
                if bottom > 0.02 {
                    b.capsule(v(0.0, 0.0, -0.05), v(0.0, bottom, -0.05), 0.025, steel);
                    b.cuboid(v(0.0, 0.02, -0.05), v(0.2, 0.02, 0.15), steel);
                }
                w.screen = Some(ScreenGeom {
                    id: e.id,
                    center: e.center3(),
                    right: -r, // right as seen by someone looking at the front face
                    up: UP,
                    normal: f,
                    hw: s.width * 0.5,
                    hh: s.height * 0.5,
                    ht: 0.015,
                });
            }
            Kind::Eye(_) => {
                // A big (not to scale!) eyeball, only visible in the overview camera.
                let (c, _, _, ef) = e.eye_frame().unwrap();
                b.origin = Vec3::ZERO;
                b.r = Vec3::X;
                b.f = Vec3::Z;
                b.vis = VIS_OVERVIEW;
                b.sphere(c - ef * 0.09, 0.09, [0.85, 0.85, 0.82]);
                b.sphere(c - ef * 0.035, 0.05, [0.08, 0.3, 0.7]);
                b.sphere(c - ef * 0.012, 0.03, [0.0, 0.0, 0.0]);
                if e.height > 0.2 {
                    b.capsule(c - ef * 0.09 - UP * 0.09, Vec3::new(c.x, 0.0, c.z) - ef * 0.09, 0.015, steel);
                }
            }
        }
        let end = w.prims.len();
        if end > start {
            w.objs.push(GpuObj {
                sphere: bounding_sphere(&w.prims[start..end]),
                range: [start as u32, (end - start) as u32, e.id, 0],
            });
            w.obj_ids.push(e.id);
        }
    }
    w
}
