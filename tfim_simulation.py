"""
ECE 491 Project 4: Quantum Circuits for Time Evolution
Transverse-Field Ising Model (TFIM) Simulation

This script builds Qiskit circuits for Trotterized TFIM time evolution
and compares them against the exact solution for an 11-qubit chain.

Exact solution via full Hamiltonian diagonalization is used for comparison only.

Authors: [Your names here]
Date: April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


# =============================================================================
# SECTION 0: Fast Basis-Space Helpers
# =============================================================================

def precompute_z_basis(n_qubits):
    """
    Precompute Z eigenvalues for every computational basis state.

    Row i contains the +/-1 eigenvalues of Z_i across the full basis.
    """
    dim = 2**n_qubits
    basis_indices = np.arange(dim, dtype=np.uint32)
    z_basis = np.empty((n_qubits, dim), dtype=float)

    for i in range(n_qubits):
        bits = (basis_indices >> (n_qubits - 1 - i)) & 1
        z_basis[i] = 1.0 - 2.0 * bits

    return z_basis


# =============================================================================
# SECTION 1: Build Quantum Circuits for Trotter Steps
# =============================================================================

def build_trotter_circuit_1st_order(n_qubits, J, h, dt, n_steps, excited_qubit):
    """
    Build a first-order Trotter circuit for the TFIM.

    Each Trotter step applies:
        Layer 1 (ZZ): For each pair (i, i+1): CNOT -> Rz(-2*J*dt) -> CNOT
        Layer 2 (X):  For each qubit i: Rx(-2*h*dt)

    From Nielsen & Chuang Exercise 4.51.
    """
    qc = QuantumCircuit(n_qubits)
    qc.x(excited_qubit)
    qc.barrier()

    for step in range(n_steps):
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(-2 * J * dt, i + 1)
            qc.cx(i, i + 1)
        qc.barrier()
        for i in range(n_qubits):
            qc.rx(-2 * h * dt, i)
        qc.barrier()

    return qc


def build_trotter_circuit_2nd_order(n_qubits, J, h, dt, n_steps, excited_qubit):
    """
    Build a second-order (symmetric) Trotter circuit for the TFIM.

    From Nielsen & Chuang Exercise 4.50:
        e^{i(A+B)dt} = e^{iA*dt/2} e^{iB*dt} e^{iA*dt/2} + O(dt^3)

    Each step: half-ZZ -> full-X -> half-ZZ
    Error: O(dt^3) per step vs O(dt^2) for first-order.
    """
    qc = QuantumCircuit(n_qubits)
    qc.x(excited_qubit)
    qc.barrier()

    for step in range(n_steps):
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(-J * dt, i + 1)
            qc.cx(i, i + 1)
        qc.barrier()
        for i in range(n_qubits):
            qc.rx(-2 * h * dt, i)
        qc.barrier()
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(-J * dt, i + 1)
            qc.cx(i, i + 1)
        qc.barrier()

    return qc


# =============================================================================
# SECTION 2: Measure <Z_i> from Circuit
# =============================================================================

def measure_z_expectations(qc, n_qubits):
    """Simulate circuit and compute <Z_i> for each qubit."""
    sv = Statevector.from_instruction(qc)
    z_values = np.zeros(n_qubits)
    for i in range(n_qubits):
        pauli_str = ['I'] * n_qubits
        pauli_str[n_qubits - 1 - i] = 'Z'
        op = SparsePauliOp(''.join(pauli_str))
        z_values[i] = np.real(sv.expectation_value(op))
    return z_values


# =============================================================================
# SECTION 3: Run Trotter Simulation Over Time
# =============================================================================

def run_trotter_simulation(n_qubits, J, h, excited_qubit, times,
                           n_trotter_steps, order=1):
    """Run Trotterized TFIM using Qiskit circuits for each time point."""
    z_expectations = np.zeros((n_qubits, len(times)))

    for t_idx, t in enumerate(times):
        if t == 0:
            z_vals = np.ones(n_qubits)
            z_vals[excited_qubit] = -1.0
            z_expectations[:, t_idx] = z_vals
            continue

        dt = t / n_trotter_steps
        if order == 2:
            qc = build_trotter_circuit_2nd_order(
                n_qubits, J, h, dt, n_trotter_steps, excited_qubit)
        else:
            qc = build_trotter_circuit_1st_order(
                n_qubits, J, h, dt, n_trotter_steps, excited_qubit)

        z_expectations[:, t_idx] = measure_z_expectations(qc, n_qubits)

        if (t_idx + 1) % 20 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


# =============================================================================
# SECTION 4: Exact Solution (comparison only)
# =============================================================================

def pauli_matrices():
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return I, X, Y, Z

def tensor_product_list(op_list):
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result

def build_tfim_hamiltonian(n_qubits, J, h):
    """Build the full 2^n x 2^n TFIM Hamiltonian matrix."""
    I, X, Y, Z = pauli_matrices()
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = Z
        ops[i + 1] = Z
        H += -J * tensor_product_list(ops)
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = X
        H += -h * tensor_product_list(ops)
    return H

def create_initial_state(n_qubits, excited_qubit):
    dim = 2**n_qubits
    psi0 = np.zeros(dim, dtype=complex)
    psi0[2**(n_qubits - 1 - excited_qubit)] = 1.0
    return psi0

def exact_evolution(H, psi0, times, n_qubits):
    """Exact time evolution using one Hamiltonian diagonalization."""
    times = np.asarray(times, dtype=float)
    z_basis = precompute_z_basis(n_qubits)

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    phases = np.exp(-1j * np.outer(eigenvalues, times))
    psi_t_all = eigenvectors @ (psi0_eigenbasis[:, None] * phases)

    probabilities = np.abs(psi_t_all) ** 2
    z_expectations = z_basis @ probabilities

    if len(times) > 0:
        print(f"    t = {times[-1]:.1f}  ({len(times)}/{len(times)})")

    return z_expectations


# =============================================================================
# SECTION 5: Error Analysis
# =============================================================================

def run_error_analysis(n_qubits, J, h, excited_qubit, test_time,
                       step_counts, H_matrix, psi0):
    """Compute Trotter infidelity for both orders."""
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    psi_exact = eigenvectors @ (
        psi0_eigenbasis * np.exp(-1j * eigenvalues * test_time)
    )

    errors_1st = []
    errors_2nd = []

    for n_steps in step_counts:
        print(f"    Steps = {n_steps}...")
        for order, err_list in [(1, errors_1st), (2, errors_2nd)]:
            dt = test_time / n_steps
            if order == 2:
                qc = build_trotter_circuit_2nd_order(
                    n_qubits, J, h, dt, n_steps, excited_qubit)
            else:
                qc = build_trotter_circuit_1st_order(
                    n_qubits, J, h, dt, n_steps, excited_qubit)
            psi_trotter = np.array(Statevector.from_instruction(qc))
            fidelity = np.abs(np.vdot(psi_exact, psi_trotter))**2
            err_list.append(1 - fidelity)

    return errors_1st, errors_2nd


# =============================================================================
# SECTION 6: Plotting
# =============================================================================

def plot_heatmap(z_exp, times, n_qubits, title, filename=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(z_exp, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, n_qubits - 0.5],
                   cmap='RdBu_r', norm=norm)
    ax.set_xlabel('Time (arbitrary units)', fontsize=12)
    ax.set_ylabel('Qubit index', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yticks(range(n_qubits))
    plt.colorbar(im, ax=ax, label=r'$\langle Z_i \rangle$')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def plot_comparison(z_exact, z_trotter, times, qubit_indices,
                    n_steps, order, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(qubit_indices)))
    label = "1st" if order == 1 else "2nd"
    for idx, qi in enumerate(qubit_indices):
        ax.plot(times, z_exact[qi], '-', color=colors[idx],
                label=f'Exact, qubit {qi}', linewidth=2)
        ax.plot(times, z_trotter[qi], '--', color=colors[idx],
                label=f'{label}-order (n={n_steps}), qubit {qi}', linewidth=1.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\langle Z_i \rangle$', fontsize=12)
    ax.set_title(f'Exact vs {label}-Order Trotter ({n_steps} steps)', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def plot_error_scaling(step_counts, errors_1st, errors_2nd, test_time, filename=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = np.array(step_counts, dtype=float)
    ax.loglog(sc, errors_1st, 'o-', lw=2, ms=6, color='#2b6cb0',
              label='1st-order Trotter')
    ax.loglog(sc, errors_2nd, 's-', lw=2, ms=6, color='#e24b4a',
              label='2nd-order Trotter')
    ref1 = errors_1st[0] * (sc[0]/sc)**2
    ref2 = errors_2nd[0] * (sc[0]/sc)**4
    ax.loglog(sc, ref1, '--', color='#2b6cb0', alpha=0.4, label=r'$O(1/n^2)$')
    ax.loglog(sc, ref2, '--', color='#e24b4a', alpha=0.4, label=r'$O(1/n^4)$')
    ax.set_xlabel('Number of Trotter steps', fontsize=12)
    ax.set_ylabel(r'Infidelity', fontsize=12)
    ax.set_title(f'Trotter Error Scaling at t = {test_time}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def plot_parameter_regimes(n_qubits, excited_qubit, times, filename=None):
    param_sets = [
        (1.0, 0.2, r"Ferromagnetic ($J=1, h=0.2$)"),
        (1.0, 1.0, r"Critical ($J=h=1$)"),
        (0.2, 1.0, r"Paramagnetic ($J=0.2, h=1$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for idx, (Jv, hv, label) in enumerate(param_sets):
        print(f"    {label}")
        H_t = build_tfim_hamiltonian(n_qubits, Jv, hv)
        psi0_t = create_initial_state(n_qubits, excited_qubit)
        z_t = exact_evolution(H_t, psi0_t, times, n_qubits)
        im = axes[idx].imshow(z_t, aspect='auto', origin='lower',
                              extent=[times[0], times[-1], -0.5, n_qubits-0.5],
                              cmap='RdBu_r', norm=norm)
        axes[idx].set_xlabel('Time', fontsize=11)
        axes[idx].set_ylabel('Qubit index', fontsize=11)
        axes[idx].set_title(label, fontsize=12)
        axes[idx].set_yticks(range(n_qubits))
    plt.colorbar(im, ax=axes[-1], label=r'$\langle Z_i \rangle$')
    plt.suptitle('Excitation Dynamics Across Parameter Regimes', fontsize=14, y=1.02)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def print_circuit_info(n_qubits, J, h, dt, n_steps, excited_qubit, order):
    if order == 2:
        qc = build_trotter_circuit_2nd_order(n_qubits, J, h, dt, n_steps, excited_qubit)
        label = "2nd-order"
    else:
        qc = build_trotter_circuit_1st_order(n_qubits, J, h, dt, n_steps, excited_qubit)
        label = "1st-order"
    ops = qc.count_ops()
    print(f"\n  {label} Circuit ({n_qubits}q, {n_steps} steps):")
    print(f"    Depth: {qc.depth()}")
    print(f"    CNOTs: {ops.get('cx', 0)}")
    print(f"    Rz:    {ops.get('rz', 0)}")
    print(f"    Rx:    {ops.get('rx', 0)}")


# =============================================================================
# SECTION 7: Main
# =============================================================================

if __name__ == "__main__":

    n_qubits = 11
    J = 1.0
    h = 1.0
    excited_qubit = 5
    t_max = 50.0
    n_time_points = 100
    times = np.linspace(0, t_max, n_time_points)

    print("=" * 60)
    print("ECE 491: TFIM Quantum Circuit Simulation")
    print("=" * 60)
    print(f"  {n_qubits} qubits | dim = {2**n_qubits} | J={J}, h={h}")
    print(f"  Excited qubit: {excited_qubit}")
    print(f"  Time: 0 to {t_max}, {n_time_points} points")

    print_circuit_info(n_qubits, J, h, 1.0, 10, excited_qubit, 1)
    print_circuit_info(n_qubits, J, h, 1.0, 10, excited_qubit, 2)

    psi0 = create_initial_state(n_qubits, excited_qubit)
    H_matrix = build_tfim_hamiltonian(n_qubits, J, h)

    # --- PART 1: Exact ---
    print("\n" + "=" * 60)
    print("PART 1: Exact evolution")
    print("=" * 60)
    z_exact = exact_evolution(H_matrix, psi0, times, n_qubits)
    plot_heatmap(z_exact, times, n_qubits,
                 r'Exact: $\langle Z_i \rangle$ (11 qubits, $J=h=1$)',
                 'exact_heatmap.png')

    # --- PART 2: 1st-order Trotter circuits ---
    print("\n" + "=" * 60)
    print("PART 2: 1st-order Trotter (Qiskit circuits)")
    print("=" * 60)
    for ns in [10, 50]:
        print(f"\n  --- {ns} steps ---")
        z1 = run_trotter_simulation(n_qubits, J, h, excited_qubit, times, ns, order=1)
        plot_heatmap(z1, times, n_qubits,
                     f'1st-Order Trotter ({ns} steps)',
                     f'trotter_1st_{ns}.png')
        plot_comparison(z_exact, z1, times, [3,5,7], ns, 1,
                       f'comparison_1st_{ns}.png')

    # --- PART 3: 2nd-order Trotter circuits ---
    print("\n" + "=" * 60)
    print("PART 3: 2nd-order Trotter (Qiskit circuits)")
    print("=" * 60)
    for ns in [10, 50]:
        print(f"\n  --- {ns} steps ---")
        z2 = run_trotter_simulation(n_qubits, J, h, excited_qubit, times, ns, order=2)
        plot_heatmap(z2, times, n_qubits,
                     f'2nd-Order Trotter ({ns} steps)',
                     f'trotter_2nd_{ns}.png')
        plot_comparison(z_exact, z2, times, [3,5,7], ns, 2,
                       f'comparison_2nd_{ns}.png')

    # --- PART 4: Error scaling ---
    print("\n" + "=" * 60)
    print("PART 4: Error scaling")
    print("=" * 60)
    test_time = 5.0
    step_counts = [10, 20, 50, 100, 200]
    e1, e2 = run_error_analysis(n_qubits, J, h, excited_qubit,
                                 test_time, step_counts, H_matrix, psi0)
    plot_error_scaling(step_counts, e1, e2, test_time, 'error_scaling.png')
    print(f"\n  {'Steps':>6} | {'1st-order':>12} | {'2nd-order':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}")
    for i, n in enumerate(step_counts):
        print(f"  {n:>6} | {e1[i]:>12.2e} | {e2[i]:>12.2e}")

    # --- PART 5: Parameter regimes ---
    print("\n" + "=" * 60)
    print("PART 5: Parameter regimes")
    print("=" * 60)
    times_p = np.linspace(0, 30, 60)
    plot_parameter_regimes(n_qubits, excited_qubit, times_p,
                           'parameter_comparison.png')

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
