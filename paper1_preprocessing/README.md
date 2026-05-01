# PlantCLEF2025 Rust Pipeline

## Competition and Data ([Kaggle](https://www.kaggle.com/competitions/plantclef-2025)) | Working Note ([CLEF 2025](https://ceur-ws.org/Vol-4038/paper_238.pdf))

This repository contains my Rust implementation for the PlantCLEF2025 competition, where I participate under the name "TheHeartOfNoise". The codebase demonstrates how to perform machine learning tasks, including CUDA-accelerated inference, using Rust programming language and the [Candle](https://github.com/huggingface/candle) deep learning framework.

---

## Project Structure

```
.
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── data/
│   ├── 10_images/               # Source images (including PlantCLEF2025 test quadrats)
│   ├── 20_deep_features/        # Directory for storing deep features
│   ├── 30_models/               # Directory for models and class mappings
│   ├── 50_probas_predictions_csv/  # Directory for raw predictions (probabilities per species, per plot/image)
│   └── 65_submissions/          # Directory for submissions to PlantCLEF2025
├── lib-cuda-rs/                 # Sub-crate: Optimized CUDA attention module for older GPUs
├── src/                         # Main source directory
│   ├── lib.rs                   # Core logic
│   └── main.rs                  # Minimal entry point
└── tests/                       # Integration and unit tests
```

---

## Key Features

- **Optimized CUDA Support**: Memory-efficient CUDA-based attention modules designed for older GPUs, ensuring effective inference even on less recent hardware.
- **Deep Features**: Intermediate deep features are saved for later reuse, avoiding redundant computations.
- **Baseline and Tiling Methods**: Includes implementations for both baseline inference and tiling-based approaches.

---

## Installation

1. **Prerequisites**:
   - Rust toolchain (install via [rustup](https://rustup.rs/))
   - CUDA toolkit (for GPU support)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/v-espitalier/PlantCLEF2025.git
   cd PlantCLEF2025
   ```

3. **Build**:
   ```bash
   cargo build --release
   ```

---

## Data and Models

### Data Organization

- **Source Images**: Place The 2,105 PlantCLEF2025 tests images in `data/10_images/PlantCLEF2025/PlantCLEF2025test/`.
- **Deep Features**: Automatically saved in `data/20_deep_features/` during execution.
- **Models**: Pretrained models are automatically downloaded from Hugging Face Hub.
- **Predictions**: Raw predictions are stored in `data/50_probas_predictions_csv/`.
- **Submissions**: Final submission files are saved in `data/65_submissions/`.

---

## Usage

### Generate Documentation

To generate and open the documentation locally:
```bash
cargo doc --no-deps --open
```

### Run Tests

To run all tests (including integration tests):
```bash
cargo test --all-features
```

### Using Tests as Examples

The tests in the `tests/` directory provide executable examples for generating predictions and submissions. You can adapt these tests for your own use cases:

- **Baseline518**: Ready-to-use examples for baseline inference.
- **Tiling Method**: Ready-to-use examples for tiling-based inference.
- **VaMIS**: Tests for the VaMIS method are **under development**.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [PlantCLEF2025 organizers](https://www.kaggle.com/competitions/plantclef-2025): The competition that inspired this project, organized by Hervé Goëau, Alexis Joly, and Giulio Martellucci.
- [Candle](https://github.com/huggingface/candle): A minimalist ML framework for Rust.
