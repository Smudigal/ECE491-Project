import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


# Small helpers

# Build the t = 0 spin pattern from the project setup.
# This lets us fill in the starting point without running a circuit.
def initial_z_values(n_qubits, excited_qubit):
    # Start with every site in the +1 Z state.
    # Then flip the chosen excited site to -1.
    z_values = np.ones(n_qubits)
    z_values[excited_qubit] = -1.0
    return z_values


# Add one ZZ layer across the open chain.
# We just walk left to right and give every neighbor pair the same block.
def add_zz_layer(qc, n_qubits, rz_angle):
    # Walk down the open chain one neighbor pair at a time.
    # Each CX-Rz-CX block implements one ZZ interaction.
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
        qc.rz(rz_angle, i + 1)
        qc.cx(i, i + 1)


# Add the transverse-field piece of the TFIM step.
# Every qubit feels the same field, so every qubit gets the same rotation.
def add_x_layer(qc, n_qubits, rx_angle):
    # Apply the same transverse-field rotation to each qubit.
    # This is the X part of the TFIM Hamiltonian.
    for i in range(n_qubits):
        qc.rx(rx_angle, i)


# Build one Pauli string in the order Qiskit expects.
# Any qubit we do not touch just stays as an identity.
def build_pauli_string(n_qubits, qubit_terms):
    # Start from all identities so untouched qubits stay neutral.
    # Then place the requested letters using Qiskit's right-to-left ordering.
    pauli_chars = ["I"] * n_qubits
    for qubit_index, pauli_letter in qubit_terms.items():
        pauli_chars[n_qubits - 1 - qubit_index] = pauli_letter
    return "".join(pauli_chars)


# Trotter circuits

# Build the first-order Trotter circuit for one time value.
# One Trotter step here means ZZ first and then X.
def build_trotter_circuit_1st_order(n_qubits, J, h, dt, n_steps, excited_qubit):
    # Build the circuit from the chosen excited basis state.
    # Each Trotter step applies ZZ first and then X.
    qc = QuantumCircuit(n_qubits)
    # The whole simulation starts from one flipped site in the middle.
    qc.x(excited_qubit)
    qc.barrier()

    for _ in range(n_steps):
        add_zz_layer(qc, n_qubits, -2 * J * dt)
        qc.barrier()
        add_x_layer(qc, n_qubits, -2 * h * dt)
        qc.barrier()

    return qc


# Build the second-order Trotter circuit for one time value.
# This is the symmetric half-ZZ, full-X, half-ZZ version.
def build_trotter_circuit_2nd_order(n_qubits, J, h, dt, n_steps, excited_qubit):
    # Build the same starting state as the 1st-order circuit.
    # Each step uses half-ZZ, full-X, half-ZZ for better accuracy.
    qc = QuantumCircuit(n_qubits)
    # Start from the same one-spin excitation as the 1st-order circuit.
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


# Choose which Trotter builder to call.
# Keeping the order switch here makes the rest of the code cleaner.
def build_trotter_circuit(n_qubits, J, h, dt, n_steps, excited_qubit, order=1):
    # Keep the order switch in one small place.
    # That way the rest of the code can call one function.
    if order == 2:
        return build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_steps, excited_qubit
        )

    return build_trotter_circuit_1st_order(
        n_qubits, J, h, dt, n_steps, excited_qubit
    )

# Time evolution

# Run the Trotter circuit at every requested time value.
# The final output is one <Z_i> curve for each qubit.
def run_trotter_simulation(n_qubits, J, h, excited_qubit, times,
                           n_trotter_steps, order=1):
    # Store one Z trace per qubit for every requested time.
    # For each time, build the circuit and read expectations from the statevector.
    z_expectations = np.zeros((n_qubits, len(times)))
    z_ops = []
    for i in range(n_qubits):
        z_op = SparsePauliOp(build_pauli_string(n_qubits, {i: "Z"}))
        z_ops.append(z_op)

    for t_idx, t in enumerate(times):
        if t == 0:
            # At t = 0 we already know the answer from the starting bitstring.
            z_expectations[:, t_idx] = initial_z_values(n_qubits, excited_qubit)
            continue

        dt = t / n_trotter_steps
        qc = build_trotter_circuit(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit, order=order
        )
        # Statevector gives the exact circuit state, so we can read <Z_i> directly.
        sv = Statevector.from_instruction(qc)
        for i, z_op in enumerate(z_ops):
            z_expectations[i, t_idx] = np.real(sv.expectation_value(z_op))

        if (t_idx + 1) % 20 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


# Exact solution

# Build the TFIM Hamiltonian for the open 1D chain.
# This is the matrix version of the same physics the circuit is approximating.
def build_tfim_hamiltonian(n_qubits, J, h):
    # Add up the open-chain ZZ couplings first.
    # Then add the single-qubit X-field terms and convert to a matrix.
    pauli_terms = []

    for i in range(n_qubits - 1):
        pauli_terms.append(
            (build_pauli_string(n_qubits, {i: "Z", i + 1: "Z"}), -J)
        )

    for i in range(n_qubits):
        pauli_terms.append((build_pauli_string(n_qubits, {i: "X"}), -h))

    return SparsePauliOp.from_list(pauli_terms).to_matrix()


# Build the starting basis state with one flipped qubit.
# This matches the exact 11-qubit setup from the project writeup.
def create_initial_state(n_qubits, excited_qubit):
    # Make one basis vector for the chosen starting spin pattern.
    # Only the selected excited site is flipped to |1>.
    dim = 2**n_qubits
    psi0 = np.zeros(dim, dtype=complex)
    # The basis index has to respect Qiskit's qubit ordering.
    psi0[2**(n_qubits - 1 - excited_qubit)] = 1.0
    return psi0


# Evolve the state exactly by diagonalizing the Hamiltonian once.
# This is our benchmark when we judge the Trotter circuits.
def exact_evolution(H, psi0, times, n_qubits):
    # Diagonalize the Hamiltonian once and reuse it for every time value.
    # This gives the exact benchmark used to judge the Trotter circuits.
    times = np.asarray(times, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    phases = np.exp(-1j * np.outer(eigenvalues, times))
    psi_t_all = eigenvectors @ (psi0_eigenbasis[:, None] * phases)

    # We only need basis-state probabilities to build <Z_i> from the exact state.
    probabilities = np.abs(psi_t_all) ** 2
    z_expectations = np.zeros((n_qubits, len(times)))
    basis_size = probabilities.shape[0]

    for t_idx in range(len(times)):
        for qubit_index in range(n_qubits):
            z_value = 0.0
            for basis_state in range(basis_size):
                # Bit 0 means +1 for Z and bit 1 means -1 for Z.
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

# Compare Trotter states to the exact state at one fixed time.
# This is the cleanest way to answer how many steps we need.
def run_error_analysis(n_qubits, J, h, excited_qubit, test_time,
                       step_counts, H_matrix, psi0):
    # Compare the Trotter states to the exact state at one fixed time.
    # Smaller infidelity means the approximation is doing better.
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    psi_exact = eigenvectors @ (
        psi0_eigenbasis * np.exp(-1j * eigenvalues * test_time)
    )

    errors_1st = []
    errors_2nd = []

    for n_steps in step_counts:
        print(f"    Steps = {n_steps}...")
        dt = test_time / n_steps

        qc_1st = build_trotter_circuit(
            n_qubits, J, h, dt, n_steps, excited_qubit, order=1
        )
        psi_1st = np.array(Statevector.from_instruction(qc_1st))
        fidelity_1st = np.abs(np.vdot(psi_exact, psi_1st))**2
        errors_1st.append(1 - fidelity_1st)

        qc_2nd = build_trotter_circuit(
            n_qubits, J, h, dt, n_steps, excited_qubit, order=2
        )
        psi_2nd = np.array(Statevector.from_instruction(qc_2nd))
        fidelity_2nd = np.abs(np.vdot(psi_exact, psi_2nd))**2
        errors_2nd.append(1 - fidelity_2nd)

    return errors_1st, errors_2nd


# Compare one Trotter setting to the exact state across all times.
# This is optional appendix material, not one of the main required figures.
def compute_infidelity_vs_time(n_qubits, J, h, excited_qubit, times,
                               n_trotter_steps, H_matrix, psi0, order=1):
    # Compare each Trotter state to the exact state at the same time.
    # This separates the 1st- and 2nd-order methods more clearly than <Z_i> alone.
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    psi0_eigenbasis = eigenvectors.conj().T @ psi0
    infidelities = np.zeros(len(times))

    for t_idx, t in enumerate(times):
        if t == 0:
            continue

        psi_exact = eigenvectors @ (
            psi0_eigenbasis * np.exp(-1j * eigenvalues * t)
        )
        dt = t / n_trotter_steps
        qc = build_trotter_circuit(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit, order=order
        )
        psi_trotter = np.array(Statevector.from_instruction(qc))
        fidelity = np.abs(np.vdot(psi_exact, psi_trotter))**2
        infidelities[t_idx] = 1 - fidelity

    return infidelities


# Plot helpers

# Apply the same simple styling to every plot.
# This keeps the figures consistent without much repeated code.
def _style_axes(ax):
    # Use one small styling helper so every figure reads the same way.
    # This keeps the plots clean without repeating formatting code.
    ax.set_facecolor('white')
    ax.tick_params(labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


# Draw the full qubit-vs-time heatmap.
# This is the best plot for seeing the excitation spread.
def plot_heatmap(z_exp, times, n_qubits, title, filename=None):
    # Show the full qubit-vs-time dynamics in one color map.
    # This is the easiest way to see the excitation spread across the chain.
    fig, ax = plt.subplots(figsize=(12.5, 5.6), constrained_layout=True)
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

# Compare exact and Trotter traces for a few representative qubits.
# This makes the approximation error easy to point out on a slide.
def plot_comparison(z_exact, z_trotter, times, qubit_indices,
                    n_steps, order, filename=None):
    # Overlay a few exact and Trotter qubit traces on the same axes.
    # This makes the approximation error easier to see than in a heatmap.
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = ['#2b6cb0', '#1d9e75', '#d69e2e']
    if order == 1:
        label = "1st"
    else:
        label = "2nd"
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
    qubit_handles = []
    for idx, qi in enumerate(qubit_indices):
        handle = Line2D([0], [0], color=colors[idx], lw=2.4, label=f'Qubit {qi}')
        qubit_handles.append(handle)
    method_handles = [
        Line2D([0], [0], color='black', lw=2.4, ls='-', label='Exact'),
        Line2D([0], [0], color='black', lw=1.9, ls='--',
               label=f'{label}-order Trotter'),
    ]
    legend1 = ax.legend(handles=qubit_handles, fontsize=10, ncol=len(qubit_indices),
                        loc='upper left', frameon=False, title='Colors')
    ax.add_artist(legend1)
    ax.legend(handles=method_handles, fontsize=10, loc='upper right',
              frameon=False, title='Line style')
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()

# Plot fixed-time error against the number of Trotter steps.
# This is the clearest quantitative answer to the accuracy requirement.
def plot_error_scaling(step_counts, errors_1st, errors_2nd, test_time, filename=None):
    # Show how the final-time Trotter error changes with the step count.
    # The reference lines make the expected scaling easier to compare.
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    sc = np.array(step_counts, dtype=float)
    ax.loglog(sc, errors_1st, 'o-', lw=2, ms=6, color='#2b6cb0',
              label='1st-order Trotter')
    ax.loglog(sc, errors_2nd, 's-', lw=2, ms=6, color='#e24b4a',
              label='2nd-order Trotter')
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

# Plot the state error over time for one Trotter order.
# This is mainly helpful as a backup figure if we want extra detail.
def plot_state_error_vs_time(times, error_curves, title, filename=None):
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)

    for values, label, color, linestyle in error_curves:
        ax.plot(times, values, linestyle=linestyle, color=color,
                linewidth=2.1, label=label)

    _style_axes(ax)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Infidelity', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()


# Run the exact model in a few simple parameter regimes.
# This just compares different J and h choices, so it is extra context only.
def plot_parameter_regimes(n_qubits, excited_qubit, times, filename=None):
    # Rerun the exact model in three simple parameter regimes.
    # This gives one extra physics comparison without changing the main setup.
    param_sets = [
        (1.0, 0.2, r"Ferromagnetic ($J=1, h=0.2$)"),
        (1.0, 1.0, r"Critical ($J=h=1$)"),
        (0.2, 1.0, r"Paramagnetic ($J=0.2, h=1$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), constrained_layout=True)
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

# Print a few simple circuit stats for the report.
# Gate counts and depth help explain cost and difficulty.
def print_circuit_info(n_qubits, J, h, dt, n_steps, excited_qubit, order):
    # Print a few circuit stats that are easy to mention in the report.
    # Depth and gate counts also help explain hardware difficulty.
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
    # These extra figures are nice for exploration, but they are not needed by default.
    SAVE_TROTTER_HEATMAPS = False
    SAVE_EXTRA_COMPARISON_PLOTS = False
    SAVE_STATE_ERROR_PLOTS = True
    SAVE_PARAMETER_REGIME_PLOT = True

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
    trotter_1st = {}
    for ns in [10, 50]:
        print(f"\n  --- {ns} steps ---")
        z1 = run_trotter_simulation(n_qubits, J, h, excited_qubit, times, ns, order=1)
        trotter_1st[ns] = z1
        if SAVE_TROTTER_HEATMAPS:
            plot_heatmap(z1, times, n_qubits,
                         f'1st-Order Trotter ({ns} steps)',
                         f'trotter_1st_{ns}.png')
        if ns == 10 or SAVE_EXTRA_COMPARISON_PLOTS:
            plot_comparison(z_exact, z1, times, [3, 5, 7], ns, 1,
                            f'comparison_1st_{ns}.png')

    # --- PART 3: 2nd-order Trotter circuits ---
    print("\n" + "=" * 60)
    print("PART 3: 2nd-order Trotter (Qiskit circuits)")
    print("=" * 60)
    trotter_2nd = {}
    for ns in [10, 50]:
        print(f"\n  --- {ns} steps ---")
        z2 = run_trotter_simulation(n_qubits, J, h, excited_qubit, times, ns, order=2)
        trotter_2nd[ns] = z2
        if SAVE_TROTTER_HEATMAPS:
            plot_heatmap(z2, times, n_qubits,
                         f'2nd-Order Trotter ({ns} steps)',
                         f'trotter_2nd_{ns}.png')
        if ns == 10 or SAVE_EXTRA_COMPARISON_PLOTS:
            plot_comparison(z_exact, z2, times, [3, 5, 7], ns, 2,
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

    if SAVE_STATE_ERROR_PLOTS:
        print("\n" + "=" * 60)
        print("PART 4B: Optional state error vs time")
        print("=" * 60)
        error_curves = []
        error_1st_10 = compute_infidelity_vs_time(
            n_qubits, J, h, excited_qubit, times, 10, H_matrix, psi0, order=1
        )
        error_curves.append((error_1st_10, '1st-order, 10 steps', '#2b6cb0', '--'))
        error_1st_50 = compute_infidelity_vs_time(
            n_qubits, J, h, excited_qubit, times, 50, H_matrix, psi0, order=1
        )
        error_curves.append((error_1st_50, '1st-order, 50 steps', '#1d9e75', '-'))

        error_2nd_10 = compute_infidelity_vs_time(
            n_qubits, J, h, excited_qubit, times, 10, H_matrix, psi0, order=2
        )
        error_curves.append((error_2nd_10, '2nd-order, 10 steps', '#e24b4a', '--'))
        error_2nd_50 = compute_infidelity_vs_time(
            n_qubits, J, h, excited_qubit, times, 50, H_matrix, psi0, order=2
        )
        error_curves.append((error_2nd_50, '2nd-order, 50 steps', '#d69e2e', '-'))

        plot_state_error_vs_time(
            times,
            error_curves,
            'Trotter State Error vs Time',
            'trotter_state_error_vs_time.png',
        )

    if SAVE_PARAMETER_REGIME_PLOT:
        print("\n" + "=" * 60)
        print("PART 5: Parameter regimes")
        print("=" * 60)
        times_p = np.linspace(0, 30, 60)
        plot_parameter_regimes(n_qubits, excited_qubit, times_p,
                               'parameter_comparison.png')

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
