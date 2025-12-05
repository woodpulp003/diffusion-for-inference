# Neuro-Diffusion Inference Project

A differentiable neural simulator and inference framework for recurrent neural networks.

## Phase 1: Neural Simulator

This phase implements a differentiable ODE-based neural simulator using Sompolinsky-style rate dynamics.

### Mathematical Model

The simulator models recurrent neural network dynamics using:

**Continuous-time ODE:**
```
τ dh/dt = -h + W·φ(h) + I + η(t)
```

Where:
- `h(t) ∈ ℝ^N` — internal synaptic input ("voltage-like" state)
- `x(t) = φ(h(t))` — firing rate (observable activity)
- `φ = tanh` — nonlinearity
- `W ∈ ℝ^(N×N)` — synaptic weight matrix
- `I` — constant external current
- `η(t)` — optional Gaussian noise

**Discrete-time simulation (Euler method):**
```
h_{t+1} = h_t + dt·(-h_t/τ + W·φ(h_t) + I) + √dt·σ_η·ξ_t
```

where `ξ_t ~ N(0, I)`.

---

## Local Environment Setup

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install torch numpy matplotlib

# Optional: Install JupyterLab for interactive development
pip install jupyterlab
```

### Windows

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib
```

---

## Usage

### Running Tests

Test the neural simulator:

```bash
source .venv/bin/activate
python simulator/rate_model.py
```

### Basic Usage Example

```python
import torch
from simulator import simulate_rate_network, summary_stats

# Network parameters
N = 50   # Number of neurons
T = 100  # Number of timesteps
dt = 0.1 # Euler step size
tau = 1.0 # Membrane time constant

# Create weight matrix (scaled for stable dynamics)
W = torch.randn(N, N, requires_grad=True) / (N ** 0.5)

# Initial state
h0 = torch.randn(N)

# Simulate network
activity = simulate_rate_network(W, h0, T=T, dt=dt, tau=tau)
print(f"Activity shape: {activity.shape}")  # [100, 50]

# Verify gradient flow
loss = activity.mean()
loss.backward()
print(f"W.grad shape: {W.grad.shape}")  # [50, 50]
print(f"W.grad norm: {W.grad.norm().item():.4f}")
```

---

## Project Structure

```
diffusion_inference/
│
├── simulator/                 # ← Phase 1 (implemented)
│   ├── __init__.py
│   └── rate_model.py          # Differentiable neural simulator
│
├── data/                      # ← Phase 2 (implemented)
│   ├── generators/            # Dataset generation tools
│   └── raw/                   # Generated datasets
│
├── models/                    # Model architectures (future phases)
│   ├── diffusion_prior/
│   ├── diffusion_cfg/
│   └── dps/
│
├── training/                  # Training pipelines (future phases)
├── inference/                 # Inference methods (future phases)
├── evaluation/                # Evaluation utilities (future phases)
├── utils/                     # Shared utilities (future phases)
│
├── requirements.txt
└── README.md
```

---

## API Reference

### `simulator.simulate_rate_network`

```python
def simulate_rate_network(
    W: torch.Tensor,        # [N, N] weight matrix
    h0: torch.Tensor,       # [N] initial state
    T: int,                 # Number of timesteps
    dt: float,              # Euler step size
    tau: float,             # Membrane time constant
    I: Optional[Tensor],    # [N] external input (default: zeros)
    noise_std: float,       # Noise std (default: 0.0)
    return_h: bool,         # Also return h trajectory (default: False)
) -> Tensor:                # Returns [T, N] activity trace
```

### Dataset Generation (Phase 2)

```python
from data.generators import sample_weight_matrix, generate_activity_trials, build_dataset

# Generate weight matrix
W = sample_weight_matrix(N=50, g=1.5)

# Generate activity trials
activities = generate_activity_trials(W, num_trials=10, T=100, dt=0.1, tau=1.0)

# Build full dataset via CLI
# python data/generators/build_dataset.py --out_dir data/raw/my_dataset --num_networks 1000 --N 50 --g 1.5
```

---

## Phase 1 Objectives ✓

- [x] Differentiable ODE-based neural simulator
- [x] Euler integration with noise injection support
- [x] Full autograd verification (gradients flow from activity → simulator → W)

## Phase 2 Objectives ✓

- [x] Weight matrix generator (with optional Dale's law E/I separation)
- [x] Multi-trial activity simulator wrapper
- [x] Dataset builder with CLI for generating (W, activity) pairs

---

## Future Phases

- **Phase 3:** Unconditional diffusion prior over weight matrices
- **Phase 4:** CFG conditional diffusion models
- **Phase 5:** DPS posterior refiners
- **Phase 6:** Evaluation and benchmarking

---

## License

MIT

