# MindCheck: Tech Stack & Engineering Environment

## 1. Core Technology Stack

| Layer | Technology | Version | Purpose & Justification |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.11+` | Industry standard for machine learning, optimal performance with modern type annotations, asyncio, and PyTorch C++ bindings. |
| **Deep Learning** | PyTorch | `>= 2.1.0` | Primary ML framework. Provides dynamic computation graphs, custom tensor autograd, GPU acceleration (`torch.cuda`), and clean TorchScript/ONNX export pipelines. |
| **Chess Engine / Logic** | `python-chess` | `>= 1.9.4` | Highly optimized chess library with native 64-bit integer bitboard operations, PGN parsing, legal move validation, FEN generation, and direct SVG board rendering. |
| **Numerical Processing** | NumPy | `>= 1.24.0` | High-performance multi-dimensional array operations for tensor transformations and bitwise manipulation. |
| **Data Manipulation** | Pandas | `>= 2.0.0` | Efficient tabular manipulation of dataset metadata, game manifests, annotation records, and metric tables. |
| **ML Evaluation** | Scikit-Learn | `>= 1.3.0` | Robust calculation of Macro-F1, per-class Precision/Recall/F1, confusion matrices, and Cohen's Kappa inter-rater agreement. |
| **Data Visualization** | Matplotlib & Seaborn | `>= 3.8.0` / `>= 0.13.0` | Publication-ready confusion matrices, ROC/PR curves, training loss trajectories, and ablation bar charts. |
| **Interactive Frontend** | Streamlit | `>= 1.30.0` | Mandatory project deliverable. Provides reactive UI, session state management, native SVG integration, and interactive widgets without frontend overhead. |
| **Configuration** | PyYAML & Pydantic | `>= 6.0` / `>= 2.5.0` | Strictly typed runtime configuration validation. Prevents silent configuration errors across pipelines. |
| **Testing** | PyTest | `>= 7.4.0` | Comprehensive unit and regression test suite verifying fact extraction predicates, tensor encoding shapes, and forward-pass integrity. |

---

## 2. Advanced Technical Optimizations & Architectural Elevations

### 2.1 Low-Memory Streaming PGN Parsing
- Standard PGN parsers load entire tournament archives into RAM, causing immediate out-of-memory (`OOM`) crashes on multi-gigabyte Lichess files.
- MindCheck implements a **generator-based streaming iterator** using `chess.pgn.read_game()` combined with `io.BufferedReader` and zstandard/bz2 decompression streams.
- Games are filtered, pre-screened, and converted into compact binary chunks without exceeding a strict 1 GB RAM footprint during data preparation.

### 2.2 Bitboard-Accelerated Fact Extraction
- Chess boards are represented internally in `python-chess` as 64-bit unsigned integers (`uint64` bitboards).
- Predicates such as piece attacks, pawn control, pawn chains, open files, and king safety zones are computed using **bitwise operations** (`&`, `|`, `^`, bit-shifts).
- This achieves sub-millisecond fact extraction per position, enabling real-time operation in the Streamlit app and fast dataset labeling.

### 2.3 Spatial-Temporal Tensor Memory Mapping (`np.memmap`)
- A dataset of 30,000 positions with 8-ply history (each position having $8 \times 14 \times 8 \times 8$ floats) requires ~2.5 GB of raw float memory.
- MindCheck stores extracted training samples as contiguous chunked memory-mapped binary files (`.npy` or `.h5`).
- PyTorch `Dataset` loads slices on-the-fly during training batches, ensuring zero RAM bottleneck on consumer laptops (even with 8 GB or 16 GB RAM).

### 2.4 ONNX Runtime & TorchScript CPU Inference
- While training runs on GPU (`cuda`) or CPU with PyTorch, deploying PyTorch inside Streamlit can incur significant memory overhead and cold-start latency.
- MindCheck includes an automated export pipeline to **ONNX (Open Neural Network Exchange)** and **TorchScript**.
- The Streamlit application utilizes `onnxruntime` or optimized TorchScript on CPU, executing complete 8-ply neural forward passes in **< 12 milliseconds**, ensuring 60 FPS smooth interactive UI navigation.

### 2.5 Native SVG Chessboard Rendering
- Instead of relying on brittle external iframe widgets or heavy web canvas libraries, MindCheck uses `chess.svg.board()` rendered natively through Streamlit's HTML component.
- Supports dynamically highlighting move arrows, threat squares, plan key squares (e.g. outpost holes highlighted in gold, pawn storm vectors highlighted in red).

---

## 3. Environment Setup & Dependency Installation

### Step 1: Python Virtual Environment Initialization
```bash
# Verify Python version
python --version  # Must be 3.11.x

# Create isolated virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip and wheel
python -m pip install --upgrade pip setuptools wheel
```

### Step 2: PyTorch Installation (CUDA vs CPU)
```bash
# For NVIDIA CUDA GPU Acceleration (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For Standard CPU Installation:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Core Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Hardware Profiles & Compute Resource Strategy
- **Development & Prototyping**: Standard Windows laptop (Intel Core i5/i7 or AMD Ryzen, 8GB-16GB RAM).
- **Training Acceleration**: Local NVIDIA RTX GPU or free Google Colab T4/V100 GPU (notebook training scripts provided).
- **Live Streamlit Demonstration**: Any standard laptop running on CPU (utilizing the optimized ONNX/TorchScript engine).
