"""Smaller TFIM run for simulator and hardware comparisons."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from qiskit import transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from tfim_simulation import (
    build_trotter_circuit_2nd_order,
    build_tfim_hamiltonian,
    create_initial_state,
    exact_evolution,
    initial_z_values,
    run_trotter_simulation,
)


# Shot-based helpers

def measure_z_from_counts(counts, n_qubits, n_shots):
    """Compute <Z_i> from measurement counts."""
    # Convert measured bitstrings into average Z values for each qubit.
    z_values = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        for i in range(n_qubits):
            # Qiskit stores qubit 0 at the end of the bitstring.
            bit = int(bitstring[-(i + 1)])
            if bit == 0:
                z_values[i] += count
            else:
                z_values[i] -= count
    return z_values / n_shots


def run_noisy_simulation(n_qubits, J, h, excited_qubit, times,
                         n_trotter_steps, n_shots=8192,
                         single_q_error=0.001, two_q_error=0.01):
    """Run the circuit on Aer with simple depolarizing noise."""
    # This gives a quick hardware-like comparison without using a real backend.
    # Aer already knows how to apply depolarizing noise to standard gates.
    noise_model = NoiseModel()
    one_qubit_error = depolarizing_error(single_q_error, 1)
    two_qubit_error = depolarizing_error(two_q_error, 2)
    for gate_name in ["x", "rx", "rz"]:
        noise_model.add_all_qubit_quantum_error(one_qubit_error, gate_name)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, "cx")

    backend = AerSimulator(noise_model=noise_model)
    z_expectations = np.zeros((n_qubits, len(times)))
    start_values = initial_z_values(n_qubits, excited_qubit)

    for t_idx, t in enumerate(times):
        if t == 0:
            z_expectations[:, t_idx] = start_values
            continue

        # Each circuit uses a different dt so that it lands at time t.
        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit
        )
        qc.measure_all()
        # Transpile first so Aer runs the same gate set the backend expects.
        qc_transpiled = transpile(qc, backend)
        result = backend.run(qc_transpiled, shots=n_shots).result()
        counts = result.get_counts()
        z_expectations[:, t_idx] = measure_z_from_counts(counts, n_qubits, n_shots)

        if (t_idx + 1) % 5 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


# Real hardware

def run_on_real_hardware(n_qubits, J, h, excited_qubit, times,
                         n_trotter_steps, api_token=None, n_shots=4096,
                         channel=None, instance=None):
    """Run the circuits on a real IBM backend."""
    # Build the measured circuits, send them to IBM, and turn counts into <Z_i>.
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    # Different IBM accounts can use different runtime channels.
    if api_token:
        candidate_channels = [channel] if channel else [
            "ibm_quantum_platform", "ibm_cloud"
        ]
        errors = {}
        service = None
        for candidate in candidate_channels:
            try:
                kwargs = {
                    "channel": candidate,
                    "token": api_token,
                    "instance": instance,
                }
                if candidate == "ibm_quantum_platform":
                    kwargs["region"] = "us-east"
                    kwargs["plans_preference"] = ["open"]
                service = QiskitRuntimeService(**kwargs)
                print(f"  Authenticated with channel: {candidate}")
                break
            except Exception as exc:
                errors[candidate] = f"{type(exc).__name__}: {exc}"
        if service is None:
            raise RuntimeError(
                "Could not authenticate with the provided IBM Quantum token. "
                f"Tried channels: {errors}"
            )
    else:
        try:
            service = QiskitRuntimeService(channel=channel, instance=instance)
            acct = service.active_account() or {}
            if acct.get("channel"):
                print(f"  Using saved account on channel: {acct['channel']}")
        except Exception as exc:
            raise RuntimeError(
                "No usable IBM Quantum credentials were found. "
                "Set QISKIT_IBM_TOKEN or save an account locally."
            ) from exc

    # Ask IBM for the least busy real device that can fit the circuit.
    backend = service.least_busy(
        simulator=False, min_num_qubits=n_qubits, operational=True)

    print(f"  Device: {backend.name} ({backend.num_qubits} qubits)")

    circuits = []
    valid_indices = []
    start_values = initial_z_values(n_qubits, excited_qubit)

    for t_idx, t in enumerate(times):
        if t == 0:
            # We already know the exact t=0 values.
            continue
        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit
        )
        qc.measure_all()
        circuits.append(qc)
        valid_indices.append(t_idx)

    print(f"  Transpiling {len(circuits)} circuits...")
    # This maps the circuits to the native gates of the selected backend.
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
    z_expectations[:, 0] = start_values

    for i, t_idx in enumerate(valid_indices):
        counts = result[i].data.meas.get_counts()
        z_expectations[:, t_idx] = measure_z_from_counts(
            counts, n_qubits, n_shots)

    return z_expectations, backend.name


# Plot helpers

def plot_four_panel(z_exact, z_noiseless, z_noisy, z_hardware,
                    times, n_qubits, hw_name=None, hardware_note=None,
                    filename=None):
    """Four-panel heatmap: exact, noiseless, noisy, hardware."""
    # Put the four result types on one figure so they are easy to compare.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    # Matching color limits make the four panels easier to compare.
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    datasets = [
        (z_exact, "Exact (matrix exponential)", None),
        (z_noiseless, "Noiseless Simulator", None),
        (z_noisy, "Noisy Simulator (depolarizing noise)", "Enable RUN_NOISY_SIM"),
        (
            z_hardware,
            f"Real Hardware ({hw_name})" if hw_name else "Real Hardware",
            hardware_note or "Hardware run was not executed",
        ),
    ]

    for ax, (data, title, missing_hint) in zip(axes.flat, datasets):
        if data is not None:
            im = ax.imshow(data, aspect='auto', origin='lower',
                          extent=[times[0], times[-1], -0.5, n_qubits-0.5],
                          cmap='RdBu_r', norm=norm)
        else:
            ax.text(0.5, 0.5, missing_hint,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
        # Keep the axes lined up even if one panel has no data.
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
    # Plot a few qubits directly so differences are easier to see than in a heatmap.
    fig, ax = plt.subplots(figsize=(10, 6))

    for qi in qubit_indices:
        # We reuse the same color per method so the plot stays readable.
        ax.plot(times, z_exact[qi], '-', color='#2b6cb0', linewidth=2, alpha=0.8)
        ax.plot(times, z_noiseless[qi], '-.', color='#6b46c1', linewidth=1.8, alpha=0.8)
        if z_noisy is not None:
            ax.plot(times, z_noisy[qi], '--', color='#e24b4a', linewidth=1.5, alpha=0.7)
        if z_hardware is not None:
            ax.plot(times, z_hardware[qi], 'o', color='#1d9e75',
                    markersize=3, alpha=0.6)

    # Keep the legend by method so it does not get too crowded.
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


def save_circuit_diagram(qc, title, filename, fold, scale, figure_size, text_fold):
    """Save a circuit diagram, or a text version if mpl is missing."""
    # Save a figure version first, then fall back to text if needed.
    try:
        # The mpl drawer makes a cleaner figure for the report.
        fig = qc.draw(
            output='mpl',
            style='iqp',
            fold=fold,
            justify='left',
            plot_barriers=False,
            idle_wires=False,
            scale=scale,
        )
        fig.set_size_inches(*figure_size)
        fig.suptitle(title, fontsize=14, y=0.96)
        if fig.axes:
            fig.axes[0].set_position([0.03, 0.08, 0.94, 0.74])
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close(fig)
    except MissingOptionalLibraryError:
        # Fall back to text if the optional drawing dependency is not installed.
        with open('circuit_diagram.txt', 'w', encoding='utf-8') as f:
            f.write(qc.draw(output='text', fold=text_fold).single_string())
        print("    Saved: circuit_diagram.txt")
    except Exception:
        print("    (Could not save circuit diagram)")


if __name__ == "__main__":

    # Keep this small enough for hardware runs.
    n_qubits = 5
    J = 1.0
    h = 1.0
    excited_qubit = 2
    n_trotter_steps = 3
    t_max = 3.0
    n_time_points = 20
    times = np.linspace(0, t_max, n_time_points)

    API_TOKEN = os.environ.get("QISKIT_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
    API_CHANNEL = os.environ.get("QISKIT_IBM_CHANNEL")
    API_INSTANCE = os.environ.get("QISKIT_IBM_INSTANCE")
    RUN_NOISY_SIM = True
    RUN_ON_HARDWARE = os.environ.get("RUN_ON_HARDWARE", "").lower() in {
        "1", "true", "yes", "y"
    }
    hardware_note = (
        "Hardware run skipped in this script\n"
        "(set RUN_ON_HARDWARE = True or 1)"
    )

    print("=" * 60)
    print("ABOVE AND BEYOND: Hardware Comparison")
    print("=" * 60)
    print(f"  {n_qubits} qubits | {n_trotter_steps} Trotter steps (2nd-order)")
    print(f"  Time: 0 to {t_max} | {n_time_points} points")

    dt_sample = t_max / n_trotter_steps
    qc_sample = build_trotter_circuit_2nd_order(
        n_qubits, J, h, dt_sample, n_trotter_steps, excited_qubit)
    ops = qc_sample.count_ops()
    print(f"\n  Circuit stats:")
    print(f"    Depth: {qc_sample.depth()}")
    print(f"    CNOTs: {ops.get('cx', 0)}")
    print(f"    Rz:    {ops.get('rz', 0)}")
    print(f"    Rx:    {ops.get('rx', 0)}")

    qc_step = build_trotter_circuit_2nd_order(
        n_qubits, J, h, dt_sample, 1, excited_qubit)
    save_circuit_diagram(
        qc_step,
        f"Representative 2nd-order TFIM Trotter step (repeated {n_trotter_steps}x)",
        "circuit_step_diagram.png",
        28,
        0.85,
        (13, 4.6),
        80,
    )
    save_circuit_diagram(
        qc_sample,
        f"Full 2nd-order TFIM circuit ({n_trotter_steps} Trotter steps)",
        "circuit_diagram.png",
        42,
        0.62,
        (15, 6.6),
        110,
    )

    print("\n" + "-" * 40)
    print("Step 1: Exact solution")
    H_hw = build_tfim_hamiltonian(n_qubits, J, h)
    psi0_hw = create_initial_state(n_qubits, excited_qubit)
    z_exact = exact_evolution(H_hw, psi0_hw, times, n_qubits)
    print("  Done.")

    print("\nStep 2: Noiseless simulator (statevector)")
    z_noiseless = run_trotter_simulation(
        n_qubits, J, h, excited_qubit, times, n_trotter_steps, order=2
    )
    print("  Done.")

    z_noisy = None
    if RUN_NOISY_SIM:
        print("\nStep 3: Noisy simulator (depolarizing noise)")
        z_noisy = run_noisy_simulation(
            n_qubits, J, h, excited_qubit, times, n_trotter_steps)
        print("  Done.")
    else:
        print("\nStep 3: Skipped noisy simulator (set RUN_NOISY_SIM = True)")

    z_hardware = None
    hw_name = None

    if RUN_ON_HARDWARE:
        print("\nStep 4: Real IBM Quantum hardware")
        try:
            z_hardware, hw_name = run_on_real_hardware(
                n_qubits, J, h, excited_qubit, times,
                n_trotter_steps, api_token=API_TOKEN, channel=API_CHANNEL,
                instance=API_INSTANCE)
            hardware_note = None
            print(f"  Done! Device: {hw_name}")
        except Exception as e:
            hardware_note = f"Hardware run failed\n({e})"
            print(f"  Failed: {e}")
    else:
        if not API_TOKEN:
            hardware_note = (
                "Hardware run skipped\n"
                "Set QISKIT_IBM_TOKEN or save an account,\n"
                "then run with RUN_ON_HARDWARE=1"
            )
        print("\nStep 4: Skipped (set RUN_ON_HARDWARE = True)")

    print("\n" + "-" * 40)
    print("Generating plots...")

    plot_four_panel(
        z_exact, z_noiseless, z_noisy, z_hardware,
        times, n_qubits, hw_name, hardware_note, 'hardware_comparison.png'
    )

    plot_qubit_traces(z_exact, z_noiseless, z_noisy, z_hardware, times,
                      [1, 2, 3], 'hardware_traces.png')

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
