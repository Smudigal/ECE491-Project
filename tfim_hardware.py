"""
ECE 491 Project 4: ABOVE AND BEYOND
Running TFIM Trotter Circuits on Noisy Simulators & Real IBM Hardware

This script:
1. Builds Qiskit quantum circuits for 5-qubit TFIM (shallow enough for hardware)
2. Compares: exact -> noiseless sim -> noisy sim -> real IBM hardware
3. Shows where errors come from (Trotter vs shot noise vs gate noise)

SETUP FOR REAL HARDWARE:
    1. Free account at https://quantum.ibm.com/
    2. pip install qiskit qiskit-aer qiskit-ibm-runtime
    3. Replace YOUR_API_TOKEN below

Authors: [Your names here]
Date: April 2026
"""

from functools import lru_cache
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp
from qiskit.quantum_info.operators.channel import Kraus
from qiskit_aer import AerSimulator

# Import shared functions from main simulation
from tfim_simulation import (
    build_trotter_circuit_2nd_order,
    measure_z_expectations,
    build_tfim_hamiltonian,
    create_initial_state,
    exact_evolution,
)


# =============================================================================
# SECTION 1: Shot-based and Noisy Simulation
# =============================================================================

def measure_z_from_counts(counts, n_qubits, n_shots):
    """
    Compute <Z_i> from measurement counts (shot-based).

    <Z_i> = P(qubit_i = 0) - P(qubit_i = 1)

    Qiskit bit ordering: bitstring[-1-i] = qubit i
    """
    z_values = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        for i in range(n_qubits):
            bit = int(bitstring[-(i + 1)])
            if bit == 0:
                z_values[i] += count
            else:
                z_values[i] -= count
    return z_values / n_shots


def measure_z_expectations_density_matrix(rho, n_qubits):
    """Compute <Z_i> for each qubit from a density matrix."""
    z_values = np.zeros(n_qubits)
    for i in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[n_qubits - 1 - i] = 'Z'
        op = SparsePauliOp(''.join(pauli_str))
        z_values[i] = np.real(rho.expectation_value(op))
    return z_values


@lru_cache(maxsize=None)
def single_qubit_depolarizing_channel(error_rate):
    """Return a 1-qubit depolarizing channel as a Kraus operator."""
    p = float(error_rate)
    identity = np.eye(2, dtype=complex)
    paulis = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    kraus_ops = [np.sqrt(max(0.0, 1.0 - 3.0 * p / 4.0)) * identity]
    kraus_ops.extend(np.sqrt(p / 4.0) * pauli for pauli in paulis)
    return Kraus(kraus_ops)


@lru_cache(maxsize=None)
def two_qubit_depolarizing_channel(error_rate):
    """Return a 2-qubit depolarizing channel as a Kraus operator."""
    p = float(error_rate)
    identity = np.eye(2, dtype=complex)
    paulis = [
        identity,
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    pauli_products = [np.kron(a, b) for a, b in product(paulis, repeat=2)]

    kraus_ops = [
        np.sqrt(max(0.0, 1.0 - 15.0 * p / 16.0)) * pauli_products[0]
    ]
    kraus_ops.extend(np.sqrt(p / 16.0) * pauli for pauli in pauli_products[1:])
    return Kraus(kraus_ops)


def simulate_density_matrix_with_depolarizing_noise(
    qc, single_q_error=0.001, two_q_error=0.01
):
    """
    Simulate a circuit with simple depolarizing noise after each gate.

    This avoids the Aer noise runtime, which can fail on some systems.
    """
    rho = DensityMatrix.from_label("0" * qc.num_qubits)
    one_q_channel = single_qubit_depolarizing_channel(single_q_error)
    two_q_channel = two_qubit_depolarizing_channel(two_q_error)

    for instruction in qc.data:
        name = instruction.operation.name
        qargs = [qubit._index for qubit in instruction.qubits]

        if name == "barrier":
            continue

        rho = rho.evolve(Operator(instruction.operation), qargs=qargs)

        if len(qargs) == 1:
            rho = rho.evolve(one_q_channel, qargs=qargs)
        elif len(qargs) == 2:
            rho = rho.evolve(two_q_channel, qargs=qargs)

    return rho


def run_noisy_simulation(n_qubits, J, h, excited_qubit, times,
                         n_trotter_steps, n_shots=8192,
                         single_q_error=0.001, two_q_error=0.01):
    """
    Run Trotter circuits on a noisy simulator mimicking IBM hardware.

    Uses a density-matrix simulation with depolarizing noise after each
    single-qubit or two-qubit gate:
        - single_q_error: error rate for Rx, Rz, X gates
        - two_q_error: error rate for CNOT gates
    """
    z_expectations = np.zeros((n_qubits, len(times)))

    for t_idx, t in enumerate(times):
        if t == 0:
            z_vals = np.ones(n_qubits)
            z_vals[excited_qubit] = -1.0
            z_expectations[:, t_idx] = z_vals
            continue

        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit)
        rho = simulate_density_matrix_with_depolarizing_noise(
            qc,
            single_q_error=single_q_error,
            two_q_error=two_q_error,
        )
        z_expectations[:, t_idx] = measure_z_expectations_density_matrix(rho, n_qubits)

        if (t_idx + 1) % 5 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


def run_noiseless_shot_simulation(n_qubits, J, h, excited_qubit, times,
                                  n_trotter_steps, n_shots=8192):
    """
    Run Trotter circuits on a noiseless simulator with finite shots.
    Shows the effect of shot noise alone (no gate errors).
    """
    backend = AerSimulator()
    z_expectations = np.zeros((n_qubits, len(times)))

    for t_idx, t in enumerate(times):
        if t == 0:
            z_vals = np.ones(n_qubits)
            z_vals[excited_qubit] = -1.0
            z_expectations[:, t_idx] = z_vals
            continue

        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit)
        qc.measure_all()

        qc_transpiled = transpile(qc, backend)
        result = backend.run(qc_transpiled, shots=n_shots).result()
        counts = result.get_counts()
        z_expectations[:, t_idx] = measure_z_from_counts(counts, n_qubits, n_shots)

        if (t_idx + 1) % 5 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


# =============================================================================
# SECTION 2: Real IBM Quantum Hardware
# =============================================================================

def run_on_real_hardware(n_qubits, J, h, excited_qubit, times,
                         n_trotter_steps, api_token, n_shots=4096):
    """
    Run circuits on a real IBM quantum processor.

    Uses 2nd-order Trotter with few steps to keep circuit shallow.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
    backend = service.least_busy(
        simulator=False, min_num_qubits=n_qubits, operational=True)

    print(f"  Device: {backend.name} ({backend.num_qubits} qubits)")

    # Build all circuits
    circuits = []
    valid_indices = []

    for t_idx, t in enumerate(times):
        if t == 0:
            continue
        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit)
        qc.measure_all()
        circuits.append(qc)
        valid_indices.append(t_idx)

    print(f"  Transpiling {len(circuits)} circuits...")
    transpiled = transpile(circuits, backend, optimization_level=3)

    if transpiled:
        print(f"  Depth (first circuit): {transpiled[0].depth()}")
        print(f"  Depth (last circuit):  {transpiled[-1].depth()}")
        ops = transpiled[0].count_ops()
        print(f"  CNOTs (first circuit): {ops.get('cx', 0)}")

    print(f"  Submitting {len(transpiled)} circuits, {n_shots} shots each...")
    sampler = SamplerV2(backend)
    job = sampler.run(transpiled, shots=n_shots)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results...")

    result = job.result()

    z_expectations = np.zeros((n_qubits, len(times)))
    # t=0
    z_vals = np.ones(n_qubits)
    z_vals[excited_qubit] = -1.0
    z_expectations[:, 0] = z_vals

    for i, t_idx in enumerate(valid_indices):
        counts = result[i].data.meas.get_counts()
        z_expectations[:, t_idx] = measure_z_from_counts(
            counts, n_qubits, n_shots)

    return z_expectations, backend.name


# =============================================================================
# SECTION 3: Plotting
# =============================================================================

def plot_four_panel(z_exact, z_noiseless, z_noisy, z_hardware,
                    times, n_qubits, hw_name=None, filename=None):
    """Four-panel heatmap: exact, noiseless, noisy, hardware."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    datasets = [
        (z_exact, "Exact (matrix exponential)", None),
        (z_noiseless, "Noiseless Simulator", None),
        (z_noisy, "Noisy Simulator (IBM noise model)", "Enable RUN_NOISY_SIM"),
        (z_hardware, f"Real Hardware ({hw_name})" if hw_name else "Real Hardware", "Enable RUN_ON_HARDWARE"),
    ]

    for ax, (data, title, missing_hint) in zip(axes.flat, datasets):
        if data is not None:
            im = ax.imshow(data, aspect='auto', origin='lower',
                          extent=[times[0], times[-1], -0.5, n_qubits-0.5],
                          cmap='RdBu_r', norm=norm)
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(-0.5, n_qubits - 0.5)
        else:
            ax.text(0.5, 0.5, f'Not available\n({missing_hint})',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Qubit index', fontsize=10)
        ax.set_yticks(range(n_qubits))

    fig.colorbar(im, ax=axes, label=r'$\langle Z_i \rangle$',
                 shrink=0.8, pad=0.02)
    plt.suptitle('TFIM: Simulator vs Real Quantum Hardware', fontsize=14, y=1.02)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()


def plot_qubit_traces(z_exact, z_noiseless, z_noisy, z_hardware, times,
                      qubit_indices, filename=None):
    """Line plot comparing exact, simulator, noisy, and hardware traces."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for qi in qubit_indices:
        ax.plot(times, z_exact[qi], '-', color='#2b6cb0', linewidth=2, alpha=0.8)
        ax.plot(times, z_noiseless[qi], '-.', color='#6b46c1', linewidth=1.8, alpha=0.8)
        if z_noisy is not None:
            ax.plot(times, z_noisy[qi], '--', color='#e24b4a', linewidth=1.5, alpha=0.7)
        if z_hardware is not None:
            ax.plot(times, z_hardware[qi], 'o', color='#1d9e75',
                    markersize=3, alpha=0.6)

    # Custom legend (one entry per method, not per qubit)
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='#2b6cb0', lw=2, label='Exact'),
        Line2D([0], [0], color='#6b46c1', lw=1.8, ls='-.', label='Noiseless simulator'),
    ]
    if z_noisy is not None:
        legend_lines.append(
            Line2D([0], [0], color='#e24b4a', lw=1.5, ls='--', label='Noisy simulator')
        )
    if z_hardware is not None:
        legend_lines.append(
            Line2D([0], [0], color='#1d9e75', marker='o', lw=0, ms=5,
                   label='Real hardware'))

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\langle Z_i \rangle$', fontsize=12)
    title_parts = ['Exact', 'Noiseless']
    if z_noisy is not None:
        title_parts.append('Noisy')
    if z_hardware is not None:
        title_parts.append('Hardware')
    ax.set_title(f"Qubit Traces: {' vs '.join(title_parts)} (qubits {qubit_indices})",
                 fontsize=14)
    ax.legend(handles=legend_lines, fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()


# =============================================================================
# SECTION 4: Main
# =============================================================================

if __name__ == "__main__":

    # --- Config ---
    n_qubits = 5
    J = 1.0
    h = 1.0
    excited_qubit = 2        # Middle of 5 qubits
    n_trotter_steps = 3      # Keep shallow for hardware!
    t_max = 3.0
    n_time_points = 20
    times = np.linspace(0, t_max, n_time_points)

    API_TOKEN = "YOUR_API_TOKEN_HERE"
    RUN_NOISY_SIM = True
    RUN_ON_HARDWARE = False   # Set True when ready

    print("=" * 60)
    print("ABOVE AND BEYOND: Hardware Comparison")
    print("=" * 60)
    print(f"  {n_qubits} qubits | {n_trotter_steps} Trotter steps (2nd-order)")
    print(f"  Time: 0 to {t_max} | {n_time_points} points")

    # --- Print circuit stats ---
    dt_sample = t_max / n_trotter_steps
    qc_sample = build_trotter_circuit_2nd_order(
        n_qubits, J, h, dt_sample, n_trotter_steps, excited_qubit)
    ops = qc_sample.count_ops()
    print(f"\n  Circuit stats:")
    print(f"    Depth: {qc_sample.depth()}")
    print(f"    CNOTs: {ops.get('cx', 0)}")
    print(f"    Rz:    {ops.get('rz', 0)}")
    print(f"    Rx:    {ops.get('rx', 0)}")

    # Save circuit diagram
    try:
        fig = qc_sample.draw(output='mpl', style='iqp', fold=60)
        fig.savefig('circuit_diagram.png', dpi=300, bbox_inches='tight')
        print(f"    Saved: circuit_diagram.png")
        plt.close(fig)
    except MissingOptionalLibraryError:
        with open('circuit_diagram.txt', 'w', encoding='utf-8') as f:
            f.write(qc_sample.draw(output='text', fold=120).single_string())
        print("    Saved: circuit_diagram.txt")
    except Exception:
        print("    (Could not save circuit diagram)")

    # --- Step 1: Exact ---
    print("\n" + "-" * 40)
    print("Step 1: Exact solution")
    H_hw = build_tfim_hamiltonian(n_qubits, J, h)
    psi0_hw = create_initial_state(n_qubits, excited_qubit)
    z_exact = exact_evolution(H_hw, psi0_hw, times, n_qubits)
    print("  Done.")

    # --- Step 2: Noiseless simulator ---
    print("\nStep 2: Noiseless simulator (statevector)")
    z_noiseless = np.zeros((n_qubits, len(times)))
    for t_idx, t in enumerate(times):
        if t == 0:
            z_vals = np.ones(n_qubits)
            z_vals[excited_qubit] = -1.0
            z_noiseless[:, t_idx] = z_vals
            continue
        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit)
        z_noiseless[:, t_idx] = measure_z_expectations(qc, n_qubits)
    print("  Done.")

    # --- Step 3: Noisy simulator ---
    z_noisy = None
    if RUN_NOISY_SIM:
        print("\nStep 3: Noisy simulator (IBM noise model)")
        z_noisy = run_noisy_simulation(
            n_qubits, J, h, excited_qubit, times, n_trotter_steps)
        print("  Done.")
    else:
        print("\nStep 3: Skipped noisy simulator (set RUN_NOISY_SIM = True)")

    # --- Step 4: Real hardware ---
    z_hardware = None
    hw_name = None

    if RUN_ON_HARDWARE and API_TOKEN != "YOUR_API_TOKEN_HERE":
        print("\nStep 4: Real IBM Quantum hardware")
        try:
            z_hardware, hw_name = run_on_real_hardware(
                n_qubits, J, h, excited_qubit, times,
                n_trotter_steps, API_TOKEN)
            print(f"  Done! Device: {hw_name}")
        except Exception as e:
            print(f"  Failed: {e}")
    else:
        print("\nStep 4: Skipped (set RUN_ON_HARDWARE = True)")

    # --- Step 5: Plots ---
    print("\n" + "-" * 40)
    print("Generating plots...")

    plot_four_panel(z_exact, z_noiseless, z_noisy, z_hardware,
                    times, n_qubits, hw_name, 'hardware_comparison.png')

    plot_qubit_traces(z_exact, z_noiseless, z_noisy, z_hardware, times,
                      [1, 2, 3], 'hardware_traces.png')

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    err_sim = np.mean(np.abs(z_exact - z_noiseless))
    print(f"  Avg error (noiseless sim vs exact):  {err_sim:.6f}")
    if z_noisy is not None:
        err_noisy = np.mean(np.abs(z_exact - z_noisy))
        print(f"  Avg error (noisy sim vs exact):      {err_noisy:.4f}")
    if z_hardware is not None:
        err_hw = np.mean(np.abs(z_exact - z_hardware))
        print(f"  Avg error (hardware vs exact):       {err_hw:.4f}")
    print(f"\n  Files: circuit_diagram.png or circuit_diagram.txt, hardware_comparison.png, hardware_traces.png")
    print("=" * 60)
