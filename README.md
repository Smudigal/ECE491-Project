# ECE 491 Project 4: Quantum Circuits for Time Evolution

Trotterized time evolution of the transverse-field Ising model (TFIM) for ECE 491 at Michigan State University.

## Files

- `tfim_simulation.py` — Main 11-qubit simulation (exact, 1st/2nd order Trotter, error analysis, parameter sweep)
- `tfim_hardware.py` — 5-qubit Qiskit circuits, noisy simulator, real IBM Quantum hardware
- `requirements.txt` — Python dependencies

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Running

```bash
# Main simulation (generates all plots)
python3 tfim_simulation.py

# Hardware comparison (requires Qiskit, optionally IBM Quantum account)
python3 tfim_hardware.py
```

If you are running headless, use:

```bash
MPLBACKEND=Agg python3 tfim_simulation.py
MPLBACKEND=Agg python3 tfim_hardware.py
```

## What This Does

1. Simulates spin excitation spreading in an 11-qubit TFIM chain
2. Compares 1st-order and 2nd-order Trotter decomposition accuracy
3. Explores three parameter regimes (ferromagnetic, critical, paramagnetic)
4. Runs circuits on real IBM Quantum hardware (above and beyond)

## Notes

- `tfim_simulation.py` uses gate-level Qiskit Trotter circuits for the approximate evolution and a faster exact solver for comparison.
- `tfim_hardware.py` includes a built-in depolarizing-noise simulation for the hardware-style comparison and leaves real IBM hardware optional.
- If `pylatexenc` is unavailable, the hardware script will fall back to a text circuit diagram.

## Authors

- [Your names here]
