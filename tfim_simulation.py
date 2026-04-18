"""Main TFIM simulation code."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


# Small helpers

def initial_z_values(n_qubits, excited_qubit):
    """Return the <Z_i> values at time t = 0."""
    # Start with all spins at +1 and flip the excited site.
    z_values = np.ones(n_qubits)
    # The excited site starts flipped, so its Z expectation is -1.
    z_values[excited_qubit] = -1.0
    return z_values


def add_zz_layer(qc, n_qubits, rz_angle):
    """Add one nearest-neighbor ZZ layer using CX-Rz-CX blocks."""
    # Apply the same ZZ interaction block to every neighboring pair.
    for i in range(n_qubits - 1):
        # CX-Rz-CX gives the two-qubit ZZ phase for one neighbor pair.
        qc.cx(i, i + 1)
        qc.rz(rz_angle, i + 1)
        qc.cx(i, i + 1)


def add_x_layer(qc, n_qubits, rx_angle):
    """Add one transverse-field X layer."""
    # Rotate every qubit by the same X-field angle.
    for i in range(n_qubits):
        qc.rx(rx_angle, i)


def build_pauli_string(n_qubits, qubit_terms):
    """Build a Pauli string in Qiskit's qubit order."""
    # Fill the string with identities, then replace the qubits we care about.
    pauli_chars = ["I"] * n_qubits
    for qubit_index, pauli_letter in qubit_terms.items():
        # Qiskit puts qubit 0 on the right side of the string.
        pauli_chars[n_qubits - 1 - qubit_index] = pauli_letter
    return "".join(pauli_chars)


# Trotter circuits

def build_trotter_circuit_1st_order(n_qubits, J, h, dt, n_steps, excited_qubit):
    """Build the 1st-order Trotter circuit."""
    # This circuit alternates ZZ layers and X-field layers for each step.
    qc = QuantumCircuit(n_qubits)
    # This puts one local spin excitation in the chain.
    qc.x(excited_qubit)
    qc.barrier()

    for _ in range(n_steps):
        add_zz_layer(qc, n_qubits, -2 * J * dt)
        # Barriers keep the Trotter layers visually separate.
        qc.barrier()
        add_x_layer(qc, n_qubits, -2 * h * dt)
        qc.barrier()

    return qc


def build_trotter_circuit_2nd_order(n_qubits, J, h, dt, n_steps, excited_qubit):
    """Build the 2nd-order Trotter circuit."""
    # This uses the symmetric half-ZZ, full-X, half-ZZ ordering.
    qc = QuantumCircuit(n_qubits)
    # Start from the same single-site excitation as the 1st-order circuit.
    qc.x(excited_qubit)
    qc.barrier()

    for _ in range(n_steps):
        add_zz_layer(qc, n_qubits, -J * dt)
        qc.barrier()
        add_x_layer(qc, n_qubits, -2 * h * dt)
        qc.barrier()
        add_zz_layer(qc, n_qubits, -J * dt)
        qc.barrier()

    return qc


def build_trotter_circuit(n_qubits, J, h, dt, n_steps, excited_qubit, order=1):
    """Pick either the 1st-order or 2nd-order Trotter circuit."""
    # Keep the order choice in one place so the rest of the code stays simpler.
    if order == 2:
        return build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_steps, excited_qubit
        )

    return build_trotter_circuit_1st_order(
        n_qubits, J, h, dt, n_steps, excited_qubit
    )

# Time evolution

def run_trotter_simulation(n_qubits, J, h, excited_qubit, times,
                           n_trotter_steps, order=1):
    """Run Trotterized TFIM using Qiskit circuits for each time point."""
    # Store one <Z_i> trace per qubit across all requested times.
    z_expectations = np.zeros((n_qubits, len(times)))
    z_ops = [
        SparsePauliOp(build_pauli_string(n_qubits, {i: "Z"}))
        for i in range(n_qubits)
    ]

    for t_idx, t in enumerate(times):
        if t == 0:
            z_expectations[:, t_idx] = initial_z_values(n_qubits, excited_qubit)
            continue

        # We scale the Trotter step size so the full circuit reaches time t.
        dt = t / n_trotter_steps
        qc = build_trotter_circuit(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit, order=order
        )
        # Statevector lets us read exact expectation values from the circuit.
        sv = Statevector.from_instruction(qc)
        for i, z_op in enumerate(z_ops):
            z_expectations[i, t_idx] = np.real(sv.expectation_value(z_op))

        if (t_idx + 1) % 20 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


# Exact solution

def build_tfim_hamiltonian(n_qubits, J, h):
    """Build the TFIM Hamiltonian matrix."""
    # Collect the ZZ coupling terms and X-field terms before converting to a matrix.
    pauli_terms = []

    for i in range(n_qubits - 1):
        # Neighboring spins couple through ZZ terms.
        pauli_terms.append(
            (build_pauli_string(n_qubits, {i: "Z", i + 1: "Z"}), -J)
        )

    for i in range(n_qubits):
        # Each site also feels the transverse X field.
        pauli_terms.append((build_pauli_string(n_qubits, {i: "X"}), -h))

    return SparsePauliOp.from_list(pauli_terms).to_matrix()


def create_initial_state(n_qubits, excited_qubit):
    """Create the basis state with one flipped qubit."""
    # Build the computational basis vector that matches the chosen excited site.
    dim = 2**n_qubits
    psi0 = np.zeros(dim, dtype=complex)
    # This index matches the computational basis ordering used by Qiskit.
    psi0[2**(n_qubits - 1 - excited_qubit)] = 1.0
    return psi0


def exact_evolution(H, psi0, times, n_qubits):
    """Exact time evolution using one Hamiltonian diagonalization."""
    # This gives the exact benchmark we compare the Trotter circuits against.
    times = np.asarray(times, dtype=float)

    # Diagonalize once, then reuse it for every time value.
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    # Move the initial state into the eigenbasis before adding phases.
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    phases = np.exp(-1j * np.outer(eigenvalues, times))
    psi_t_all = eigenvectors @ (psi0_eigenbasis[:, None] * phases)

    probabilities = np.abs(psi_t_all) ** 2
    z_expectations = np.zeros((n_qubits, len(times)))
    basis_size = probabilities.shape[0]

    for t_idx in range(len(times)):
        for qubit_index in range(n_qubits):
            z_value = 0.0
            # Turn basis-state probabilities into <Z_i> values.
            for basis_state in range(basis_size):
                bit = (basis_state >> (n_qubits - 1 - qubit_index)) & 1
                if bit == 0:
                    z_value += probabilities[basis_state, t_idx]
                else:
                    z_value -= probabilities[basis_state, t_idx]
            z_expectations[qubit_index, t_idx] = z_value

    if len(times) > 0:
        print(f"    t = {times[-1]:.1f}  ({len(times)}/{len(times)})")

    return z_expectations


# Error check

def run_error_analysis(n_qubits, J, h, excited_qubit, test_time,
                       step_counts, H_matrix, psi0):
    """Compute Trotter infidelity for both orders."""
    # Check how the approximation error changes as we add more Trotter steps.
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    # This is the exact state we compare each Trotter circuit against.
    psi_exact = eigenvectors @ (
        psi0_eigenbasis * np.exp(-1j * eigenvalues * test_time)
    )

    errors_1st = []
    errors_2nd = []

    for n_steps in step_counts:
        print(f"    Steps = {n_steps}...")
        for order, err_list in [(1, errors_1st), (2, errors_2nd)]:
            dt = test_time / n_steps
            qc = build_trotter_circuit(
                n_qubits, J, h, dt, n_steps, excited_qubit, order=order
            )
            psi_trotter = np.array(Statevector.from_instruction(qc))
            # Infidelity is 1 - fidelity, so smaller is better.
            fidelity = np.abs(np.vdot(psi_exact, psi_trotter))**2
            err_list.append(1 - fidelity)

    return errors_1st, errors_2nd


# Plot helpers

def _style_axes(ax):
    """Keep the plots simple and readable."""
    # Reuse the same axis styling so every figure looks consistent.
    ax.set_facecolor('white')
    ax.tick_params(labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def plot_heatmap(z_exp, times, n_qubits, title, filename=None):
    # Show the full qubit-vs-time evolution as one color plot.
    fig, ax = plt.subplots(figsize=(12.5, 5.6), constrained_layout=True)
    # Center the color scale at zero so positive and negative values are balanced.
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(z_exp, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], -0.5, n_qubits - 0.5],
                   cmap='RdBu_r', norm=norm)
    _style_axes(ax)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Qubit index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_qubits))
    cbar = fig.colorbar(im, ax=ax, label=r'$\langle Z_i \rangle$',
                        shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def plot_comparison(z_exact, z_trotter, times, qubit_indices,
                    n_steps, order, filename=None):
    # Compare exact and Trotter results for a few representative qubits.
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = ['#2b6cb0', '#1d9e75', '#d69e2e']
    label = "1st" if order == 1 else "2nd"
    for idx, qi in enumerate(qubit_indices):
        ax.plot(times, z_exact[qi], '-', color=colors[idx],
                linewidth=2.4, alpha=0.95)
        ax.plot(times, z_trotter[qi], '--', color=colors[idx],
                linewidth=1.9, alpha=0.95)
    _style_axes(ax)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'$\langle Z_i \rangle$', fontsize=12)
    ax.set_title(f'Exact vs {label}-Order Trotter ({n_steps} steps)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.22, linewidth=0.8)

    from matplotlib.lines import Line2D
    qubit_handles = [
        Line2D([0], [0], color=colors[idx], lw=2.4, label=f'Qubit {qi}')
        for idx, qi in enumerate(qubit_indices)
    ]
    method_handles = [
        Line2D([0], [0], color='black', lw=2.4, ls='-', label='Exact'),
        Line2D([0], [0], color='black', lw=1.9, ls='--',
               label=f'{label}-order Trotter'),
    ]
    # Split the legend so color and line style are easier to read.
    legend1 = ax.legend(handles=qubit_handles, fontsize=10, ncol=len(qubit_indices),
                        loc='upper left', frameon=False, title='Colors')
    ax.add_artist(legend1)
    ax.legend(handles=method_handles, fontsize=10, loc='upper right',
              frameon=False, title='Line style')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def plot_error_scaling(step_counts, errors_1st, errors_2nd, test_time, filename=None):
    # Plot how the Trotter error shrinks as the number of steps increases.
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    sc = np.array(step_counts, dtype=float)
    ax.loglog(sc, errors_1st, 'o-', lw=2, ms=6, color='#2b6cb0',
              label='1st-order Trotter')
    ax.loglog(sc, errors_2nd, 's-', lw=2, ms=6, color='#e24b4a',
              label='2nd-order Trotter')
    # These reference lines show the expected error scaling trends.
    ref1 = errors_1st[0] * (sc[0]/sc)**2
    ref2 = errors_2nd[0] * (sc[0]/sc)**4
    ax.loglog(sc, ref1, '--', color='#2b6cb0', alpha=0.4, label=r'$O(1/n^2)$')
    ax.loglog(sc, ref2, '--', color='#e24b4a', alpha=0.4, label=r'$O(1/n^4)$')
    _style_axes(ax)
    ax.set_xlabel('Number of Trotter steps', fontsize=12)
    ax.set_ylabel(r'Infidelity', fontsize=12)
    ax.set_title(f'Trotter Error Scaling at t = {test_time}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.22, linewidth=0.8, which='both')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def plot_parameter_regimes(n_qubits, excited_qubit, times, filename=None):
    # Run the exact model for three parameter choices and plot them side by side.
    param_sets = [
        (1.0, 0.2, r"Ferromagnetic ($J=1, h=0.2$)"),
        (1.0, 1.0, r"Critical ($J=h=1$)"),
        (0.2, 1.0, r"Paramagnetic ($J=0.2, h=1$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)
    # Use the same color scale in every panel so the plots are comparable.
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    psi0_t = create_initial_state(n_qubits, excited_qubit)

    for ax, (Jv, hv, label) in zip(axes, param_sets):
        print(f"    {label}")
        H_t = build_tfim_hamiltonian(n_qubits, Jv, hv)
        z_t = exact_evolution(H_t, psi0_t, times, n_qubits)
        im = ax.imshow(z_t, aspect='auto', origin='lower',
                       extent=[times[0], times[-1], -0.5, n_qubits-0.5],
                       cmap='RdBu_r', norm=norm)
        _style_axes(ax)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Qubit index', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_yticks(range(n_qubits))
    cbar = fig.colorbar(im, ax=axes, label=r'$\langle Z_i \rangle$',
                        shrink=0.9, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    plt.suptitle('Excitation Dynamics Across Parameter Regimes',
                 fontsize=14, fontweight='bold')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

def print_circuit_info(n_qubits, J, h, dt, n_steps, excited_qubit, order):
    # Print a few basic circuit stats that are useful for the report.
    qc = build_trotter_circuit(
        n_qubits, J, h, dt, n_steps, excited_qubit, order=order
    )
    label = "2nd-order" if order == 2 else "1st-order"
    ops = qc.count_ops()
    print(f"\n  {label} Circuit ({n_qubits}q, {n_steps} steps):")
    print(f"    Depth: {qc.depth()}")
    print(f"    CNOTs: {ops.get('cx', 0)}")
    print(f"    Rz:    {ops.get('rz', 0)}")
    print(f"    Rx:    {ops.get('rx', 0)}")


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
