#![feature(portable_simd)]

/// The `egui` re-export for building native GUIs with the eframe framework.
use eframe::egui::{self, Color32};
/// Additional 2D geometric tools from eframe, e.g. `Vec2`.
use eframe::epaint::Vec2;
/// The `egui_plot` crate for plotting data in an egui-based app.
use egui_plot::{Line, Plot, PlotPoints};

/// The `rand` crate for random number generation.
use rand::Rng;
use std::f32::consts::PI;

/// Portable SIMD for updating positions in small batches.
use core::simd::f32x8;

/// For multi-core parallelism:
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
const DEFAULT_ROLLING_FRAMES: usize = 500;

// ===================================================================================
// Simulation Parameters
// ===================================================================================

/// Configuration parameters that can be adjusted by the user before starting the simulation.
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
    rolling_frames: usize,
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
    colors: Vec<Color32>,
}

impl ParticleStorage {
    /// Constructs a new `ParticleStorage` with capacity for `n` particles.
    fn with_capacity(n: usize) -> Self {
        Self {
            x: vec![0.0; n],
            y: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            colors: vec![Color32::WHITE; n],
        }
    }
}

// ===================================================================================
// Main Application
// ===================================================================================

/// The primary application state:
/// - A configuration UI (sliders, etc.) used to set up parameters
/// - The "running" flag indicating if the simulation is active
/// - A "reset needed" or "new init" function that re-creates the `ParticleStorage`
/// - The data for the ring-buffered histogram and the smoothed histogram
///
/// We do partial parallel updates (positions) but keep collisions single-threaded
/// for correctness and simplicity.
struct MyApp {
    // -------------- Config / UI --------------
    params: SimulationParams, // user-chosen parameters
    running: bool,            // is the simulation running?
    needs_reset: bool,        // user requested a reset

    // -------------- Simulation Data --------------
    particles: ParticleStorage,
    speed_hist: Vec<f32>,                   // raw histogram for current frame
    hist_ring: Vec<Vec<f32>>,               // ring buffer of histograms
    hist_index: usize,                      // next slot in ring buffer
    stored_frames: usize,                   // how many frames stored so far
    smooth_speed_hist: Vec<f32>,            // rolling average of histograms

    // -------------- Derived Data --------------
    // The actual number of particles, bin count, etc. from the last init
    actual_num_particles: usize,
    actual_num_bins: usize,
}

impl MyApp {
    /// Creates the initial `MyApp` with default config (not yet running).
    fn new() -> Self {
        // Initially, we haven't created the real ParticleStorage because the user
        // can adjust sliders, then press "Start" or "Reset" to actually init.
        Self {
            params: SimulationParams::default(),
            running: false,
            needs_reset: true, // so it initialises once

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

    /// Initialises (or re-initialises) the simulation using the current `params`.
    fn reset_simulation(&mut self) {
        // Create new storage for the requested particle count
        self.particles = ParticleStorage::with_capacity(self.params.num_particles);

        let mut rng = rand::rng();
        for i in 0..self.params.num_particles {
            // random position
            let x = rng.random_range(self.params.radius..(self.params.box_size_x - self.params.radius));
            let y = rng.random_range(self.params.radius..(self.params.box_size_y - self.params.radius));
            self.particles.x[i] = x;
            self.particles.y[i] = y;

            // random speed
            let speed: f32 = rng.random_range(0..(self.params.mean_speed as i64 * 2)) as f32;
            let angle: f32 = rng.random_range(0.0..(2.0 * PI));
            self.particles.vx[i] = speed * angle.cos();
            self.particles.vy[i] = speed * angle.sin();

            // random colour
            let r = rng.random_range(0..=255) as u8;
            let g = rng.random_range(0..=255) as u8;
            let b = rng.random_range(0..=255) as u8;
            self.particles.colors[i] = Color32::from_rgb(r, g, b);
        }

        // Update derived data
        self.actual_num_particles = self.params.num_particles;
        self.actual_num_bins = self.params.num_bins;

        // Reinitialise histogram buffers
        self.speed_hist = vec![0.0; self.actual_num_bins];
        self.hist_ring = vec![vec![0.0; self.actual_num_bins]; self.params.rolling_frames];
        self.hist_index = 0;
        self.stored_frames = 0;
        self.smooth_speed_hist = vec![0.0; self.actual_num_bins];

        self.needs_reset = false;
    }

    /// Compute the speed histogram for the *current* frame.
    fn compute_speed_hist(&mut self) {
        // zero out
        for val in &mut self.speed_hist {
            *val = 0.0;
        }

        let bin_count = self.actual_num_bins as f32;
        let bin_size = self.params.max_speed / bin_count;

        // Tally speeds
        for i in 0..self.actual_num_particles {
            let speed = self.particles.vx[i].hypot(self.particles.vy[i]).min(self.params.max_speed);
            let bin = (speed / bin_size) as usize;
            let bin = bin.min(self.actual_num_bins - 1);
            self.speed_hist[bin] += 1.0;
        }
    }

    /// Push the current frame's histogram into the ring buffer, then
    /// average them to produce a smoothed histogram.
    fn push_and_smooth(&mut self) {
        // Copy current histogram
        let mut frame_hist = vec![0.0; self.actual_num_bins];
        frame_hist.copy_from_slice(&self.speed_hist);

        // Write into ring buffer
        self.hist_ring[self.hist_index] = frame_hist;
        self.hist_index = (self.hist_index + 1) % self.params.rolling_frames;

        if self.stored_frames < self.params.rolling_frames {
            self.stored_frames += 1;
        }

        // Smooth
        self.smooth_speed_hist.fill(0.0);
        for frame_idx in 0..self.stored_frames {
            let arr = &self.hist_ring[frame_idx];
            for (bin_idx, &val) in arr.iter().enumerate() {
                self.smooth_speed_hist[bin_idx] += val;
            }
        }
        if self.stored_frames > 0 {
            let denom = self.stored_frames as f32;
            for bin_idx in 0..self.actual_num_bins {
                self.smooth_speed_hist[bin_idx] /= denom;
            }
        }
    }

    /// Runs an O(n^2) collision detection/resolution on **single thread** for simplicity.
    /// (Making this parallel safely is more complex, requiring locks or other careful steps.)
    fn handle_collisions(&mut self) {
        let diameter = self.params.radius * 2.0;
        let n = self.actual_num_particles;

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.particles.x[j] - self.particles.x[i];
                let dy = self.particles.y[j] - self.particles.y[i];
                let dist2 = dx * dx + dy * dy;
                if dist2 < diameter * diameter {
                    let dist = dist2.sqrt();
                    if dist == 0.0 {
                        continue; // perfect overlap => skip
                    }

                    let nx = dx / dist;
                    let ny = dy / dist;

                    let v1n = self.particles.vx[i] * nx + self.particles.vy[i] * ny;
                    let v2n = self.particles.vx[j] * nx + self.particles.vy[j] * ny;

                    // swap normal components
                    let v1n_post = v2n; 
                    let v2n_post = v1n;

                    self.particles.vx[i] += (v1n_post - v1n) * nx;
                    self.particles.vy[i] += (v1n_post - v1n) * ny;
                    self.particles.vx[j] += (v2n_post - v2n) * nx;
                    self.particles.vy[j] += (v2n_post - v2n) * ny;

                    // push them apart
                    let overlap = diameter - dist;
                    let half_overlap = 0.5 * overlap;
                    self.particles.x[i] -= nx * half_overlap;
                    self.particles.y[i] -= ny * half_overlap;
                    self.particles.x[j] += nx * half_overlap;
                    self.particles.y[j] += ny * half_overlap;
                }
            }
        }
    }

    /// Parallel + SIMD approach for updating positions.
    fn update_positions_parallel(&mut self) {
        // We'll do the position update in parallel across chunks of 8.
        let n = self.actual_num_particles;
        let dt = self.params.dt;

        // Number of chunks of 8 we can handle
        let chunked_len = n / 8;

        // Each chunk is processed in parallel
        let new_positions: Vec<(f32x8, f32x8)> = (0..chunked_len)
            .into_par_iter()
            .map(|chunk_index| {
                let chunk_start = chunk_index * 8;
                let vx_simd = f32x8::from_slice(&self.particles.vx[chunk_start..]);
                let vy_simd = f32x8::from_slice(&self.particles.vy[chunk_start..]);
                let x_simd = f32x8::from_slice(&self.particles.x[chunk_start..]);
                let y_simd = f32x8::from_slice(&self.particles.y[chunk_start..]);

                let dt_simd = f32x8::splat(dt);
                let x_new = x_simd + vx_simd * dt_simd;
                let y_new = y_simd + vy_simd * dt_simd;

                (x_new, y_new)
            })
            .collect();

        for (chunk_index, (x_new, y_new)) in new_positions.into_iter().enumerate() {
            let chunk_start = chunk_index * 8;
            let x_slice = &mut self.particles.x[chunk_start..chunk_start + 8];
            let y_slice = &mut self.particles.y[chunk_start..chunk_start + 8];
            x_new.as_array().iter().enumerate().for_each(|(i, &val)| {
                x_slice[i] = val;
            });
            y_new.as_array().iter().enumerate().for_each(|(i, &val)| {
                y_slice[i] = val;
            });
        }

        // leftover
        let leftover_start = chunked_len * 8;
        for i in leftover_start..n {
            self.particles.x[i] += self.particles.vx[i] * dt;
            self.particles.y[i] += self.particles.vy[i] * dt;
        }

        // boundaries (also done in parallel for demonstration):
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

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --------------------------
        // Sidebar with configuration
        // --------------------------
        egui::SidePanel::left("config_panel").show(ctx, |ui| {
            ui.heading("Simulation Controls");

            // Sliders: only matter if we haven't started or we want to reset
            if !self.running {
                ui.add(egui::Slider::new(&mut self.params.num_particles, 1000..=50_000).text("Particles"));
                ui.add(egui::Slider::new(&mut self.params.radius, 0.5..=10.0).text("Particle Radius"));
                ui.add(egui::Slider::new(&mut self.params.box_size_x, 1000.0..=10_000.0).text("Box Size X"));
                ui.add(egui::Slider::new(&mut self.params.box_size_y, 500.0..=5_000.0).text("Box Size Y"));
                ui.add(egui::Slider::new(&mut self.params.dt, 0.0001..=0.1).text("DT"));
                ui.add(egui::Slider::new(&mut self.params.max_speed, 1000.0..=100_000.0).text("Max Speed"));
                ui.add(egui::Slider::new(&mut self.params.num_bins, 100..=1000).text("Num Bins"));
                ui.add(egui::Slider::new(&mut self.params.mean_speed, 0.0..=50_000.0).text("Mean Speed"));
                ui.add(egui::Slider::new(&mut self.params.rolling_frames, 10..=1000).text("Rolling Frames"));
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
        // If we're running, do the simulation steps
        // ------------------------------------
        if self.running {
            // 1) Parallel + SIMD position updates
            self.update_positions_parallel();

            // 2) O(n^2) collisions (single-thread for correctness)
            self.handle_collisions();

            // 3) Compute histogram, push + smooth
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
                    .width(200.0)
                    .height(400.0)
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

                painter.circle_filled(pos, self.params.radius.min(5.0) * scale, self.particles.colors[i]);
            }
        });

        // Request another frame to keep animating (or remain static if stopped).
        ctx.request_repaint();
    }
}

// ===================================================================================
// Normal sampling for Maxwell–Boltzmann distribution in 3D (unused, but included)
// ===================================================================================

fn sample_standard_normal(rng: &mut impl Rng) -> f32 {
    let u1: f32 = rng.random();
    let u2: f32 = rng.random();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0_f32 * PI * u2;
    r * theta.cos()
}

fn sample_maxwell_boltzmann_speed(rng: &mut impl Rng, mean_speed: f32) -> f32 {
    let sigma = mean_speed * PI.sqrt() / 2.0;
    let vx = sigma * sample_standard_normal(rng);
    let vy = sigma * sample_standard_normal(rng);
    let vz = sigma * sample_standard_normal(rng);
    (vx * vx + vy * vy + vz * vz).sqrt()
}

// ===================================================================================
// main
// ===================================================================================

fn main() -> eframe::Result<()> {
    // For large N, consider drastically lower N or using a broad-phase approach.
    let native_options = eframe::NativeOptions {
        ..Default::default()
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get_physical()) // or however many threads you want
        .build_global()
        .unwrap();

    eframe::run_native(
        "Particle Simulation (Parallel+SIMD + Collisions)",
        native_options,
        Box::new(|_cc| Ok(Box::new(MyApp::new()))),
    )
}