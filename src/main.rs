#![feature(portable_simd)]


use eframe::egui::{self, Color32};
use eframe::epaint::Vec2;
use egui_plot::{Line, Plot, PlotPoints};
use rand::Rng;
use std::f32::consts::PI;
use core::simd::f32x8;
use rayon::prelude::*;

// ===================================================================================
// Default Constants (used as initial slider defaults)
// ===================================================================================

const DEFAULT_NUM_PARTICLES: usize = 10_000;
const DEFAULT_RADIUS: f32 = 2.0;
const DEFAULT_BOX_SIZE_X: f32 = 4000.0;
const DEFAULT_BOX_SIZE_Y: f32 = 2000.0;
const DEFAULT_DT: f32 = 0.016;
const DEFAULT_MAX_SPEED: f32 = 50_000.0;
const DEFAULT_NUM_BINS: usize = 500;
const DEFAULT_MEAN_SPEED: f32 = 10_000.0;
const DEFAULT_ROLLING_FRAMES: u64 = 500;
const DEFAULT_STARTING_DISTRIBUTION: StartingDistribution = StartingDistribution::ConstantDist;

const COLLISION_BOX_COEFFICIENT: f32 = 0.5; // the size (in diameters) of the boxes used to parallelise the collision checking

// ===================================================================================
// Simulation Parameters
// ===================================================================================

#[derive(Clone)]
struct SimulationParams {
    num_particles: usize,
    radius: f32,
    box_size_x: f32,
    box_size_y: f32,
    dt: f32,
    max_speed: f32,
    num_bins: usize,
    mean_speed: f32,
    rolling_frames: u64,
    distribution: StartingDistribution
}

impl Default for SimulationParams {
    fn default() -> Self {
        SimulationParams {
            num_particles: DEFAULT_NUM_PARTICLES,
            radius: DEFAULT_RADIUS,
            box_size_x: DEFAULT_BOX_SIZE_X,
            box_size_y: DEFAULT_BOX_SIZE_Y,
            dt: DEFAULT_DT,
            max_speed: DEFAULT_MAX_SPEED,
            num_bins: DEFAULT_NUM_BINS,
            mean_speed: DEFAULT_MEAN_SPEED,
            rolling_frames: DEFAULT_ROLLING_FRAMES,
            distribution: DEFAULT_STARTING_DISTRIBUTION
        }
    }
}

// ===================================================================================
// Particle Storage
// ===================================================================================

/// Stores the particle data in a **Structure of Arrays** (SoA) layout, which
/// can be more convenient for batch updates or SIMD operations.
struct ParticleStorage {
    /// Per-particle x-coordinate in the simulation space.
    x: Vec<f32>,
    /// Per-particle y-coordinate in the simulation space.
    y: Vec<f32>,
    /// Per-particle velocity in the x-direction.
    vx: Vec<f32>,
    /// Per-particle velocity in the y-direction.
    vy: Vec<f32>,

    /// Per-particle colour used when drawing in the UI.
    colours: Vec<Color32>,
}

impl ParticleStorage {
    /// Constructs a new `ParticleStorage` with capacity for `n` particles.
    fn with_capacity(n: usize) -> Self {
        Self {
            x: vec![0.0; n],
            y: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            colours: vec![Color32::WHITE; n],
        }
    }
}

// ==================================================================================================
// The distribution of the speeds the particles are given when the simulation is initialised
// ==================================================================================================

/// Stores the selected distibution to initialise particles with in an enum
#[derive(Clone, PartialEq)]
enum StartingDistribution {
    MaxwellBoltzmann,
    ConstantDist,
    LinearRandom,
    Normal
}

trait Distribution {
    fn sample(&self, rng: &mut impl Rng, mean_speed: f32) -> (f32, f32);
    fn new() -> Self;
}

// string converter to allow us to display the options neatly in the dropdown box
impl StartingDistribution {
    fn as_str(&self) -> &'static str {
        match self {
            StartingDistribution::MaxwellBoltzmann => "Maxwell-Boltzmann",
            StartingDistribution::ConstantDist => "Constant",
            StartingDistribution::LinearRandom => "Linear Random",
            StartingDistribution::Normal => "Normal",
        }
    }
}

/// Delegate StartingDistribution sampling to the appropriate distribution object
// the idea is that the StartingDistribution enum acts as a sort of pass-through alias for the true selected distribution
// therefore any calls to the sample function need to be delegated as appropriate
impl Distribution for StartingDistribution {
    fn sample(&self, rng: &mut impl Rng, mean_speed: f32) -> (f32, f32) {
        match self {
            StartingDistribution::MaxwellBoltzmann => MaxwellBoltzmann::new().sample(rng, mean_speed),
            StartingDistribution::ConstantDist => ConstantDist::new().sample(rng, mean_speed),
            StartingDistribution::LinearRandom => LinearRandom::new().sample(rng, mean_speed),
            StartingDistribution::Normal => StandardNormal::new().sample(rng, mean_speed),
        }
    }

    fn new() -> Self {
        StartingDistribution::ConstantDist
    }
}


struct StandardNormal {}
impl Distribution for StandardNormal {
    
    fn new() -> Self {
        Self {}
    }

    fn sample(&self, rng: &mut impl Rng, mean_speed: f32) -> (f32, f32) {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();
        let r: f32 = (-2.0_f32 * u1.ln()).sqrt() * mean_speed;
        let theta: f32 = 2.0_f32 * PI * u2;
        (r * theta.cos(), r * theta.sin())
    }  
}

struct ConstantDist {}
impl Distribution for ConstantDist {
    
    fn new() -> Self {
        Self {}
    }

    fn sample(&self, rng: &mut impl Rng, mean_speed: f32) -> (f32, f32) {
        // The key here is to ensure a random angle - if we just returned mean_speed.sqrt(), all particles would be moving in the same direction
        // This is equivalent to the question of
        // 'Choose any two random numbers x and y such that sqrt(x^2 + y^2) = mean_speed
        // In theory this should include the possibility of unbounded positive and negative values for x and y
        // however this is obviously not an ideal situation here
        // instead, we choose a random angle and calculate the components accordingly

        let theta: f32 = rng.random_range(0.0..(2.0 * PI));

        (mean_speed * theta.cos(), mean_speed * theta.sin())
    }
}

struct LinearRandom {}
impl Distribution for LinearRandom {
    
    fn new() -> Self {
        Self {}
    }

    fn sample(&self, rng: &mut impl Rng, mean_speed: f32) -> (f32, f32) {
        let vx: f32 = rng.random_range(0..mean_speed as i64 * 2) as f32;
        let vy: f32 = rng.random_range(0..mean_speed as i64 * 2) as f32;

        (vx, vy)
    }
}

struct MaxwellBoltzmann {}
impl Distribution for MaxwellBoltzmann {
    
    fn new() -> Self {
        Self {}
    }

    fn sample(&self, rng: &mut impl Rng, mean_speed: f32) -> (f32, f32) {
        let sigma: f32 = PI.sqrt() / 2.0; // the mean_speed multiplication already happens in the normal distribution sampler
        let normal_gen: StandardNormal = StandardNormal::new(); 
        let (vx, vy) = normal_gen.sample(rng, mean_speed);
        let velocities: (f32, f32) = (sigma * vx, sigma * vy);

        velocities
    }
}



// ===================================================================================
// Main Application
// ===================================================================================

/// The primary application state:
/// - A configuration UI (sliders, etc.) used to set up parameters
/// - The running flag indicating if the simulation is active
/// - A reset-needed or new-init function that recreates the ParticleStorage
/// - The data for the ring-buffered/smoothed histogram
///
/// We do partial parallel updates but keep collisions single-threaded
/// for correctness and simplicity.
struct App {
    // -------------- Config / UI --------------
    params: SimulationParams, // user-chosen parameters
    running: bool,            // is the simulation running?
    needs_reset: bool,        // user requested a reset

    // -------------- Simulation Data --------------
    particles: ParticleStorage,
    speed_hist: Vec<f32>,                   // raw histogram for current frame
    hist_ring: Vec<Vec<f32>>,               // ring buffer of histograms
    hist_index: u64,                      // next slot in ring buffer
    stored_frames: u64,                   // how many frames stored so far
    smooth_speed_hist: Vec<f32>,            // rolling average of histograms

    // -------------- Derived Data --------------
    // the actual number of particles, bin count, etc. from the last init
    actual_num_particles: usize,
    actual_num_bins: usize,
}



impl App {
    /// Creates the initial `App` with default config (not yet running).
    fn new() -> Self {
        Self {
            params: SimulationParams::default(),
            running: false,
            needs_reset: true, // so it initialises uhhhh.... initially?

            particles: ParticleStorage::with_capacity(DEFAULT_NUM_PARTICLES),
            speed_hist: vec![],
            hist_ring: vec![],
            hist_index: 0,
            stored_frames: 0,
            smooth_speed_hist: vec![],
            actual_num_particles: 0,
            actual_num_bins: 0,
        }
    }

    /// initialises the simulation using the current params
    fn reset_simulation(&mut self) {
        // create new storage for the requested particle count
        self.particles = ParticleStorage::with_capacity(self.params.num_particles);

        let mut rng = rand::rng();
        for i in 0..self.params.num_particles {
            // random position
            let x: f32 = rng.random_range(self.params.radius..(self.params.box_size_x - self.params.radius));
            let y: f32 = rng.random_range(self.params.radius..(self.params.box_size_y - self.params.radius));
            self.particles.x[i] = x;
            self.particles.y[i] = y;

            // sets the velocity of a new particle according to the distribution selected
            let (vx, vy) = self.params.distribution.sample(&mut rng, self.params.mean_speed);
            self.particles.vx[i] = vx;
            self.particles.vy[i] = vy;

            // random colour
            let r: u8 = rng.random_range(0..=255) as u8;
            let g: u8 = rng.random_range(0..=255) as u8;
            let b: u8 = rng.random_range(0..=255) as u8;
            self.particles.colours[i] = Color32::from_rgb(r, g, b);
        }

        // update derived data
        self.actual_num_particles = self.params.num_particles;
        self.actual_num_bins = self.params.num_bins;

        // reinitialise histogram buffers
        self.speed_hist = vec![0.0; self.actual_num_bins];
        self.hist_ring = vec![vec![0.0; self.actual_num_bins]; self.params.rolling_frames as usize];
        self.hist_index = 0;
        self.stored_frames = 0;
        self.smooth_speed_hist = vec![0.0; self.actual_num_bins];

        self.needs_reset = false;
    }

    /// compute the speed histogram for the *current* frame
    fn compute_speed_hist(&mut self) {
        // zero out
        for val in &mut self.speed_hist {
            *val = 0.0;
        }

        let bin_count: f32 = self.actual_num_bins as f32;
        let bin_size: f32 = self.params.max_speed / bin_count;

        // gather speed data
        for i in 0..self.actual_num_particles {
            let speed: f32 = self.particles.vx[i].hypot(self.particles.vy[i]).min(self.params.max_speed);
            let bin: usize = (speed / bin_size) as usize;
            let bin: usize = bin.min(self.actual_num_bins - 1);
            self.speed_hist[bin] += 1.0;
        }
    }

    /// Push the current frame's histogram into the ring buffer, then
    /// average them to produce a smoothed histogram
    fn push_and_smooth(&mut self) {
        // copy current histogram
        let mut frame_hist: Vec<f32> = vec![0.0; self.actual_num_bins];
        frame_hist.copy_from_slice(&self.speed_hist);
    
        // make sure hist_index is wrapped
        let ring_size: u64 = self.hist_ring.len() as u64;
        let idx: u64 = self.hist_index % ring_size; // ensure it's in [0..ring_size-1]
    
        self.hist_ring[idx as usize] = frame_hist;
    
        // increment and wrap for next time
        self.hist_index += 1;
        self.hist_index %= ring_size;
    
        if self.stored_frames < ring_size {
            self.stored_frames += 1;
        }

        // smoooooooooth
        self.smooth_speed_hist.fill(0.0);
        for frame_idx in 0..self.stored_frames {
            let arr: &Vec<f32> = &self.hist_ring[frame_idx as usize];
            for (bin_idx, &val) in arr.iter().enumerate() {
                self.smooth_speed_hist[bin_idx] += val;
            }
        }
        if self.stored_frames > 0 {
            let denom: f32 = self.stored_frames as f32;
            for bin_idx in 0..self.actual_num_bins {
                self.smooth_speed_hist[bin_idx] /= denom;
            }
        }
    }

    /// Runs collision detection and resolution using broad-phase spatial partitioning - essentially just divide up the box in to mini boxes
    /// Then, only need to check for collision inside the box
    /// Collision events are collected in parallel and resolved in a single-threaded pass to avoid data races
    /// Improved from old O(n^2) system
    fn handle_collisions(&mut self) {
        use rayon::prelude::*;

        let diameter: f32 = self.params.radius * 2.0;
        let n: usize = self.actual_num_particles;

        // local struct to store collision info
        #[derive(Clone)]
        struct Collision {
            i: usize,   // index of first particle
            j: usize,   // index of second particle
            // velocity deltas
            dvx_i: f32,
            dvy_i: f32,
            dvx_j: f32,
            dvy_j: f32,
            // position corrections
            dx_i: f32,
            dy_i: f32,
            dx_j: f32,
            dy_j: f32,
        }

        // grid cell - should be at least the diameter but can be changed
        let cell_size: f32 = diameter * COLLISION_BOX_COEFFICIENT;
        // number of cells in x and y directions
        let nx: isize = (self.params.box_size_x / cell_size).ceil() as isize;
        let ny: isize = (self.params.box_size_y / cell_size).ceil() as isize;

        // build a vector of buckets, each bucket will store particle indices
        // we make it of length nx * ny (convert to usize safely)
        if nx <= 0 || ny <= 0 {
            // if the box dimensions are too small can just skip broad phase
            // fallback to single thread loop or do nothing
            return;
        }
        let mut grid: Vec<Vec<usize>> = vec![Vec::new(); (nx * ny) as usize];

        // function to map a particle at (px, py) -> grid index
        // returns None if out of bounds
        let grid_index = |px: f32, py: f32| -> Option<usize> {
            let gx: isize = (px / cell_size).floor() as isize;
            let gy: isize = (py / cell_size).floor() as isize;
            if gx < 0 || gy < 0 || gx >= nx || gy >= ny {
                return None;
            }
            let idx = gy * nx + gx;
            Some(idx as usize)
        };

        // assign particles to grid buckets (single-threaded for simplicity)
        for i in 0..n {
            let px = self.particles.x[i];
            let py = self.particles.y[i];
            if let Some(idx) = grid_index(px, py) {
                grid[idx].push(i);
            }
        }

        // detection in parallel: for each cell, check collisions in that cell + some number of neighbours
        // collect all collisions in local buffers, then flatten
        let collisions: Vec<Collision> = (0..grid.len())
            .into_par_iter()
            .flat_map(|cell_id: usize| {
                let mut local_collisions: Vec<Collision> = Vec::new();

                let cell_x: isize = cell_id as isize % nx;
                let cell_y: isize = cell_id as isize / nx;

                // we look in [cell_x-1, cell_x, cell_x+1] and [cell_y-1, cell_y, cell_y+1]
                for cx_off in (cell_x - 1).max(0)..=(cell_x + 1).min(nx - 1) {
                    for cy_off in (cell_y - 1).max(0)..=(cell_y + 1).min(ny - 1) {
                        let neighbour_id: usize = (cy_off * nx + cx_off) as usize;

                        // to avoid checking pairs twice, skip if neighbour < cell
                        if neighbour_id < cell_id {
                            continue;
                        }

                        let particles_a: &Vec<usize> = &grid[cell_id];
                        let particles_b: &Vec<usize> = &grid[neighbour_id];

                        for (index_a, &i) in particles_a.iter().enumerate() {
                            // if same cell, only check j > i in that cell
                            let start_b: usize = if neighbour_id == cell_id {
                                index_a + 1
                            } else {
                                0
                            };
                            for &j in &particles_b[start_b..] {

                                let dx: f32 = self.particles.x[j] - self.particles.x[i];
                                let dy: f32 = self.particles.y[j] - self.particles.y[i];
                                let dist2 = dx * dx + dy * dy;
                                if dist2 < diameter * diameter {
                                    let dist = dist2.sqrt();
                                    if dist == 0.0 {
                                        // perfect overlap => skip - something very bad has happened - bosons anyone?
                                        continue;
                                    }

                                    let nx: f32 = dx / dist;
                                    let ny: f32 = dy / dist;

                                    let v1n: f32 = self.particles.vx[i] * nx + self.particles.vy[i] * ny;
                                    let v2n: f32 = self.particles.vx[j] * nx + self.particles.vy[j] * ny;

                                    // swap normal components
                                    let v1n_post: f32 = v2n;
                                    let v2n_post: f32 = v1n;

                                    let dvx_i: f32 = (v1n_post - v1n) * nx;
                                    let dvy_i: f32 = (v1n_post - v1n) * ny;
                                    let dvx_j: f32 = (v2n_post - v2n) * nx;
                                    let dvy_j: f32 = (v2n_post - v2n) * ny;

                                    // push them apart
                                    let overlap: f32 = diameter - dist;
                                    let half_overlap: f32 = 0.5 * overlap;
                                    let dx_i: f32 = -nx * half_overlap;
                                    let dy_i: f32 = -ny * half_overlap;
                                    let dx_j: f32 = nx * half_overlap;
                                    let dy_j: f32 = ny * half_overlap;

                                    local_collisions.push(Collision {
                                        i,
                                        j,
                                        dvx_i,
                                        dvy_i,
                                        dvx_j,
                                        dvy_j,
                                        dx_i,
                                        dy_i,
                                        dx_j,
                                        dy_j,
                                    });
                                }
                            }
                        }
                    }
                }
                local_collisions
            })
            .collect();

        // single-threaded resolution 
        for c in collisions {
            self.particles.vx[c.i] += c.dvx_i;
            self.particles.vy[c.i] += c.dvy_i;
            self.particles.vx[c.j] += c.dvx_j;
            self.particles.vy[c.j] += c.dvy_j;

            self.particles.x[c.i] += c.dx_i;
            self.particles.y[c.i] += c.dy_i;
            self.particles.x[c.j] += c.dx_j;
            self.particles.y[c.j] += c.dy_j;
        }
    }

    /// Parallel + SIMD approach for updating positions
    fn update_positions_parallel(&mut self) {
        // position update in parallel chunks of 8.
        let n: usize = self.actual_num_particles;
        let dt: f32 = self.params.dt;

        // number of chunks
        let chunked_len: usize = n / 8;

        // update the chunks in parallel
        let new_positions: Vec<(f32x8, f32x8)> = (0..chunked_len)
            .into_par_iter()
            .map(|chunk_index| {
                let chunk_start: usize = chunk_index * 8;
                let vx_simd: std::simd::Simd<f32, 8> = f32x8::from_slice(&self.particles.vx[chunk_start..]);
                let vy_simd: std::simd::Simd<f32, 8> = f32x8::from_slice(&self.particles.vy[chunk_start..]);
                let x_simd: std::simd::Simd<f32, 8> = f32x8::from_slice(&self.particles.x[chunk_start..]);
                let y_simd: std::simd::Simd<f32, 8> = f32x8::from_slice(&self.particles.y[chunk_start..]);

                let dt_simd: std::simd::Simd<f32, 8> = f32x8::splat(dt);
                let x_new: std::simd::Simd<f32, 8> = x_simd + vx_simd * dt_simd;
                let y_new: std::simd::Simd<f32, 8> = y_simd + vy_simd * dt_simd;

                (x_new, y_new)
            })
            .collect();

        for (chunk_index, (x_new, y_new)) in new_positions.into_iter().enumerate() {
            let chunk_start = chunk_index * 8;
            let x_slice: &mut [f32] = &mut self.particles.x[chunk_start..chunk_start + 8];
            let y_slice: &mut [f32] = &mut self.particles.y[chunk_start..chunk_start + 8];
            x_new.as_array().iter().enumerate().for_each(|(i, &val)| {
                x_slice[i] = val;
            });
            y_new.as_array().iter().enumerate().for_each(|(i, &val)| {
                y_slice[i] = val;
            });
        }

        // leftover - there's always someone....
        // really I should just make the slider go up in units of 8
        // but UI design is hard....
        let leftover_start: usize = chunked_len * 8;
        for i in leftover_start..n {
            self.particles.x[i] += self.particles.vx[i] * dt;
            self.particles.y[i] += self.particles.vy[i] * dt;
        }

        // unfortunately we still need to check if we have hit any walls - I suspect this might be slowing things down quite a lot
        self.particles
            .x
            .par_iter_mut()
            .zip(self.particles.vx.par_iter_mut())
            .for_each(|(x, vx)| {
                if *x < self.params.radius {
                    *x = self.params.radius;
                    *vx = -*vx;
                } else if *x > self.params.box_size_x - self.params.radius {
                    *x = self.params.box_size_x - self.params.radius;
                    *vx = -*vx;
                }
            });

        self.particles
            .y
            .par_iter_mut()
            .zip(self.particles.vy.par_iter_mut())
            .for_each(|(y, vy)| {
                if *y < self.params.radius {
                    *y = self.params.radius;
                    *vy = -*vy;
                } else if *y > self.params.box_size_y - self.params.radius {
                    *y = self.params.box_size_y - self.params.radius;
                    *vy = -*vy;
                }
            });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --------------------------
        // Sidebar with configuration
        // --------------------------
        egui::SidePanel::left("config_panel").show(ctx, |ui| {
            ui.heading("Simulation Controls");

            // sliders: only matter if we haven't started or we want to reset
            if !self.running {
                ui.add(egui::Slider::new(&mut self.params.num_particles, 1..=50_000).text("Particles"));
                ui.add(egui::Slider::new(&mut self.params.radius, 0.5..=10.0).text("Particle Radius"));
                ui.add(egui::Slider::new(&mut self.params.box_size_x, 1.0..=10_000.0).text("Box Size X"));
                ui.add(egui::Slider::new(&mut self.params.box_size_y, 1.0..=10_000.0).text("Box Size Y"));
                ui.add(egui::Slider::new(&mut self.params.dt, 0.0001..=1.0).text("DT"));
                ui.add(egui::Slider::new(&mut self.params.max_speed, 1.0..=100_000.0).text("Max Speed"));
                ui.add(egui::Slider::new(&mut self.params.num_bins, 1..=1000).text("Num Bins"));
                ui.add(egui::Slider::new(&mut self.params.mean_speed, 0.0..=50_000.0).text("Mean Speed"));
                ui.add(egui::Slider::new(&mut self.params.rolling_frames, 1..=100_000).text("Rolling Frames"));
                egui::ComboBox::from_label("Starting Distribution") // why do dropdowns need so much configuring argg
                .selected_text(self.params.distribution.as_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.params.distribution, StartingDistribution::MaxwellBoltzmann, "Maxwell-Boltzmann");
                    ui.selectable_value(&mut self.params.distribution, StartingDistribution::ConstantDist, "Constant");
                    ui.selectable_value(&mut self.params.distribution, StartingDistribution::LinearRandom, "Linear Random");
                    ui.selectable_value(&mut self.params.distribution, StartingDistribution::Normal, "Normal");
                });
            
            } else {
                ui.label("Parameters locked while running. Stop to change.");
            }

            ui.separator();

            // Start / Stop
            if self.running {
                if ui.button("Stop").clicked() {
                    self.running = false;
                }
            } else {
                if ui.button("Start").clicked() {
                    // If we haven't initialised or user changed sliders => reset
                    if self.needs_reset {
                        self.reset_simulation();
                    }
                    self.running = true;
                }
            }

            // Reset
            if ui.button("Reset").clicked() {
                // Force a brand new system with current sliders
                self.reset_simulation();
                // By default, remain "stopped"
                self.running = false;
            }
        });

        // If a reset was triggered but not performed yet, do it now
        if self.needs_reset {
            self.reset_simulation();
        }

        // ------------------------------------
        // If running, do the simulation steps
        // ------------------------------------
        if self.running {
            // position updates
            self.update_positions_parallel();

            // collisions
            self.handle_collisions();

            // 3) compute histogram, push + smooth
            self.compute_speed_hist();
            self.push_and_smooth();
        }

        // ------------------------------------
        // UI layout for top, right, central
        // ------------------------------------
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.heading("2D Gas Simulation (Collisions, Parallel+SIMD)");
            ui.label(format!("Particle count: {}", self.actual_num_particles));
            ui.label("Note: collision is O(n^2) => can be very slow for large N.");
        });

        egui::SidePanel::right("right_panel")
            .resizable(true)
            .show(ctx, |ui| {
                ui.label("Speed distribution (rolling average)");
                let plot = Plot::new("speed_histogram")
                    //.width(200.0)
                    //.height(400.0)
                    .allow_scroll(true)
                    .allow_drag(true);

                plot.show(ui, |plot_ui| {
                    if !self.smooth_speed_hist.is_empty() {
                        let bin_width = self.params.max_speed / self.actual_num_bins as f32;
                        let points: Vec<[f64; 2]> = self
                            .smooth_speed_hist
                            .iter()
                            .enumerate()
                            .map(|(i, &count)| {
                                let x = i as f32 * bin_width;
                                [x as f64, count as f64]
                            })
                            .collect();

                        let line = Line::new(PlotPoints::from(points));
                        plot_ui.line(line);
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let painter = ui.painter();
            let rect = ui.max_rect();
            let (width, height) = (rect.width(), rect.height());

            // Scale from simulation box to the drawing area:
            let scale_x = width / self.params.box_size_x;
            let scale_y = height / self.params.box_size_y;
            let scale = scale_x.min(scale_y);

            // Draw each particle as a small circle:
            for i in 0..self.actual_num_particles {
                let px = self.particles.x[i] * scale;
                let py = self.particles.y[i] * scale;
                let pos = rect.min + Vec2::new(px, py);

                painter.circle_filled(pos, self.params.radius.min(5.0) * scale, self.particles.colours[i]);
            }
        });

        // Request another frame to keep animating (or remain static if stopped).
        ctx.request_repaint();
    }
}

// ===================================================================================
// Normal sampling for Maxwell–Boltzmann distribution in 3D
// ===================================================================================




// ===================================================================================
// main
// ===================================================================================

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        ..Default::default()
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get_physical()) 
        .build_global()
        .unwrap();

    eframe::run_native(
        "Particle Simulation (Parallel+SIMD + Collisions)",
        native_options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}