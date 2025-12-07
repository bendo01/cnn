# CNN - MNIST Digit Recognition

A Convolutional Neural Network (CNN) implementation in Rust using the [Burn](https://burn.dev) deep learning framework for handwritten digit recognition on the MNIST dataset.

## 📋 Overview

This project implements a simple yet effective CNN architecture for classifying handwritten digits (0-9) from the MNIST dataset. The model is built using Rust's Burn framework v0.19 with the NdArray backend.

## 🏗️ Architecture

The CNN model consists of:

- **Convolutional Layer**: 1 input channel → 8 output channels (3×3 kernel)
- **ReLU Activation**: Non-linear activation function
- **Max Pooling**: 2×2 pooling layer for dimensionality reduction
- **Flatten Layer**: Converts 2D feature maps to 1D vector
- **Fully Connected Layer**: Maps features to 10 output classes (digits 0-9)

## 🚀 Features

- **Automatic Dataset Download**: Downloads MNIST dataset automatically if not present
- **Efficient Training**: Uses mini-batch training with configurable batch size
- **Image Inference**: Predict digits from custom PNG images
- **Progress Tracking**: Training progress visualization with indicatif
- **Model Persistence**: Save and load trained models

## 📦 Dependencies

```toml
burn = { version = "0.19", features = ["ndarray", "dataset", "train"] }
burn-ndarray = "0.19"
burn-autodiff = "0.19"
image = "0.25"
reqwest = { version = "0.12", features = ["blocking", "rustls-tls"] }
```

See [`Cargo.toml`](Cargo.toml) for the complete list of dependencies.

## 🔧 Installation

### Prerequisites

Before installing this project, you need to have Rust installed on your system.

### Windows Installation

#### 1. Install Rust

Download and run [rustup-init.exe](https://rustup.rs/) from the official Rust website.

**Option A: Using PowerShell**

```powershell
# Download and run the installer
Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "$env:TEMP\rustup-init.exe"
& "$env:TEMP\rustup-init.exe"
```

**Option B: Manual Installation**

1. Visit [https://rustup.rs/](https://rustup.rs/)
2. Download `rustup-init.exe`
3. Run the installer and follow the prompts
4. Select option 1 (default installation)
5. Restart your terminal after installation

#### 2. Verify Rust Installation

```powershell
rustc --version
cargo --version
```

#### 3. Install Git (if not already installed)

Download and install from [https://git-scm.com/download/win](https://git-scm.com/download/win)

#### 4. Clone and Build the Project

```powershell
# Clone the repository
git clone <repository-url>
cd cnn

# Build the project
cargo build --release

# Run the project
cargo run --release
```

---

### Linux/macOS Installation

#### 1. Install Rust

Open a terminal and run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the on-screen instructions. When prompted, select option 1 (default installation).

#### 2. Configure Your Shell

After installation, configure your current shell:

```bash
source $HOME/.cargo/env
```

Or restart your terminal.

#### 3. Verify Rust Installation

```bash
rustc --version
cargo --version
```

Expected output should show versions like:

```
rustc 1.x.x (xxxxxx 20xx-xx-xx)
cargo 1.x.x (xxxxxx 20xx-xx-xx)
```

#### 4. Install Git (if not already installed)

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install git
```

**Fedora:**

```bash
sudo dnf install git
```

**Arch Linux:**

```bash
sudo pacman -S git
```

**macOS (using Homebrew):**

```bash
brew install git
```

Or use Xcode Command Line Tools:

```bash
xcode-select --install
```

#### 5. Clone and Build the Project

```bash
# Clone the repository
git clone <repository-url>
cd cnn

# Build the project in release mode (optimized)
cargo build --release

# Run the project
cargo run --release
```

---

### 🔍 Verification

After installation, verify everything is working:

```bash
# Check if the project compiles
cargo check

# Run tests (if available)
cargo test

# Build and run in one command
cargo run --release
```

### 📦 Optional: Install Rust Analyzer (IDE Support)

For better development experience, install Rust Analyzer for your IDE:

- **VS Code**: Install the "rust-analyzer" extension
- **IntelliJ/CLion**: Install the Rust plugin
- **Vim/Neovim**: Install rust-analyzer LSP

## 💻 Usage

### Training the Model

Run the training process:

```bash
cargo run --release
```

The program will:

1. Download the MNIST dataset to `~/.cache/mnist/` (if not already present)
2. Train the model for the specified number of epochs
3. Display training progress and loss metrics
4. Save the trained model

### Predicting Custom Images

To predict a digit from your own image:

1. Prepare a 28×28 grayscale image of a handwritten digit (PNG format)
2. Update the `image_path` in the `infer_image` function call in `main.rs`
3. Run the inference:
   ```bash
   cargo run --release
   ```

Example inference code is included for `digit.png`.

## 📁 Project Structure

```
cnn/
├── src/
│   └── main.rs              # Main implementation
├── training_data/           # Cached MNIST dataset
├── digit.png                # Sample test image
├── check_image.rs           # Image verification utility
├── Cargo.toml               # Project dependencies
└── Readme.md                # This file
```

## 🎯 Training Configuration

Default training parameters (can be modified in `main.rs`):

- **Epochs**: 5
- **Batch Size**: 32
- **Learning Rate**: 0.001 (configurable in optimizer)
- **Backend**: NdArray (CPU)

## 📊 Dataset

The MNIST dataset consists of:

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

Dataset is automatically downloaded from:

- Training images: `train-images-idx3-ubyte.gz`
- Training labels: `train-labels-idx1-ubyte.gz`
- Test images: `t10k-images-idx3-ubyte.gz`
- Test labels: `t10k-labels-idx1-ubyte.gz`

## 🔍 How It Works

1. **Data Loading**: The MNIST dataset is downloaded and parsed from IDX format
2. **Preprocessing**: Images are normalized from 0-255 to 0.0-1.0 range
3. **Training**: Mini-batch gradient descent with backpropagation
4. **Inference**: Custom images are resized to 28×28, inverted if needed, and fed to the model
5. **Output**: Softmax probabilities for each digit class

## 🛠️ Development

### Adding Custom Layers

Modify the `CnnModel` struct in `main.rs`:

```rust
pub struct CnnModel<B: Backend> {
    conv1: Conv2d<B>,
    // Add more layers here
    fc: Linear<B>,
}
```

### Adjusting Hyperparameters

Training parameters can be modified in the `main` function:

- Change epochs: `let epochs = 10;`
- Adjust batch size: `let batch_size = 64;`
- Modify learning rate in optimizer configuration

## 🐛 Troubleshooting

**MNIST download fails**:

- Check internet connection
- Verify write permissions to `~/.cache/mnist/`

**Out of memory during training**:

- Reduce batch size
- Use fewer convolutional filters

**Poor accuracy**:

- Increase number of epochs
- Add more convolutional layers
- Adjust learning rate

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- [Burn Framework](https://burn.dev) - Deep learning framework for Rust
- [MNIST Database](http://yann.lecun.com/exdb/mnist/) - Handwritten digit database
- Rust Community - For excellent tooling and libraries

---

**Built with ❤️ using Rust and Burn**
