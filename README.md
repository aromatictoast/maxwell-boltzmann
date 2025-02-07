# 2D Gas Simulation

A 2D gas simulation written in Rust using parallelism (via Rayon) and SIMD (with the nightly `portable_simd` feature) for fast collision detection and resolution. The simulation supports several particle initialisation velocity distributions and includes a graphical UI built with eframe/egui. It allows for a delightful visualisation of the (2D) Maxwell-Boltzmann distribution (as well as some other, perhaps slightly less physically accurate phenomena. It's a fun little toy to play around with!

## Features

- **Parallel & SIMD Updates:**  
  Efficient position updates and collision detection using Rayon and SIMD

- **Collision Resolution:**  
  Broad-phase spatial partitioning (ish) for collision detection with kinetic energy conservation(/clamping)

- **Configurable Distributions:**  
  Initialise the particles in the gas with M-B, Normal, random linear, or constant distributions

- **A Nice Graph:**
  Everyone likes a nice graph

![alt text](https://github.com/aromatictoast/maxwell-boltzmann/blob/main/Pretty%20Graph.png?raw=true)
  

## Prerequisites

- [Rust](https://www.rust-lang.org/) (nightly release is required to use `portable_simd`)
- [Cargo](https://doc.rust-lang.org/cargo/)
- Git

The project uses the following Rust crates:
- `eframe` / `egui`
- `egui_plot`
- `rand`
- `rayon`
- `core::simd` (available on nightly)

## Building and Running

### Linux

1. **Install Rust (nightly):**  
   Open a terminal and run:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup toolchain install nightly
   rustup default nightly
   ```
2. **Clone the respository:**
   ```bash
   git clone https://github.com/aromatictoast/maxwell-boltzmann/
   cd maxwell-boltzmann
   ```
3. **Build and run:**
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo run --release
   ```

### Windows
1.	**Install Rust (nightly):**
  Download and run rustup-init.exe and then open Command Prompt or PowerShell:
   ```powershell
   rustup toolchain install nightly
   rustup default nightly
   ```

2.	**Clone the Repository:**
  Use Git Bash or your preferred Git client:
   ```bash
   git clone https://github.com/aromatictoast/maxwell-boltzmann/
   cd maxwell-boltzmann
   ```

3.	**Build the Project:**
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo run --release
   ```

4.	**Run the Project:**
   ```bash
   cargo run --release
  ```

#### macOS

1.	**Install Rust (nightly):**
    Open Terminal and run:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup toolchain install nightly
   ```
Or alternatively install using brew.


2.	**Clone the Repository:**
   ```bash
   git clone https://github.com/aromatictoast/maxwell-boltzmann/
   cd maxwell-boltzmann
   ```

3.	**Build and run the project:**
   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo run --release
   ```
