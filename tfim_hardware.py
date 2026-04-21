import os
from pathlib import Path

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

# Turn raw shot counts into one <Z_i> value per qubit.
# This is how we turn hardware-style bitstrings back into physics data.
def measure_z_from_counts(counts, n_qubits, n_shots):
    # Walk through each measured bitstring and tally each qubit separately.
    # Mapping 0 to +1 and 1 to -1 turns the shot average into <Z_i>.
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


# Run the TFIM circuit on Aer with a simple depolarizing noise model.
# This gives us a hardware-like baseline without leaving the simulator.
def run_noisy_simulation(n_qubits, J, h, excited_qubit, times,
                         n_trotter_steps, n_shots=8192,
                         single_q_error=0.001, two_q_error=0.01,
                         noise_scale=1.0):
    # Build the same circuit as before, but run it through a simple Aer noise model.
    # The scaled error rates let this function support both noisy runs and ZNE.
    noise_model = NoiseModel()
    # Scaling the error rates is what lets the same function support ZNE.
    scaled_single_q_error = min(noise_scale * single_q_error, 0.999)
    scaled_two_q_error = min(noise_scale * two_q_error, 0.999)
    one_qubit_error = depolarizing_error(scaled_single_q_error, 1)
    two_qubit_error = depolarizing_error(scaled_two_q_error, 2)
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

        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit
        )
        # The noisy and hardware paths both need counts, so we add measurements here.
        qc.measure_all()
        qc_transpiled = transpile(qc, backend)
        result = backend.run(qc_transpiled, shots=n_shots).result()
        counts = result.get_counts()
        z_expectations[:, t_idx] = measure_z_from_counts(counts, n_qubits, n_shots)

        if (t_idx + 1) % 5 == 0 or t_idx == len(times) - 1:
            print(f"    t = {t:.1f}  ({t_idx+1}/{len(times)})")

    return z_expectations


# Estimate a zero-noise answer from several noisier simulator runs.
# This is error mitigation, not full quantum error correction.
def run_zne_simulation(n_qubits, J, h, excited_qubit, times,
                       n_trotter_steps, n_shots=8192,
                       single_q_error=0.001, two_q_error=0.01,
                       noise_scales=(1.0, 2.0, 3.0)):
    # Run the noisy simulator at a few stronger noise levels.
    # Then linearly extrapolate those results back to the zero-noise limit.
    scaled_runs = []

    for scale in noise_scales:
        print(f"    Noise scale = {scale:.1f}")
        scaled_runs.append(
            run_noisy_simulation(
                n_qubits, J, h, excited_qubit, times, n_trotter_steps,
                n_shots=n_shots,
                single_q_error=single_q_error,
                two_q_error=two_q_error,
                noise_scale=scale,
            )
        )

    # After we stack the noisy runs, we can fit each qubit and time point separately.
    scaled_runs = np.array(scaled_runs)
    z_zne = np.zeros_like(scaled_runs[0])

    for qubit_index in range(n_qubits):
        for t_idx in range(len(times)):
            # A simple line fit is enough for this small proof-of-concept ZNE pass.
            fit = np.polyfit(
                noise_scales,
                scaled_runs[:, qubit_index, t_idx],
                1,
            )
            z_zne[qubit_index, t_idx] = np.polyval(fit, 0.0)

    return np.clip(z_zne, -1.0, 1.0), scaled_runs


# Real hardware

# Submit the measured TFIM circuits to a real IBM backend.
# The goal is to keep the real-device path as close as possible to the simulator path.
def run_on_real_hardware(n_qubits, J, h, excited_qubit, times,
                         n_trotter_steps, api_token=None, n_shots=4096,
                         instance=None, save_transpiled_diagram=False):
    # Send the measured circuits to IBM Runtime and collect the returned counts.
    # This keeps the real-device path simple and close to the simulator flow.
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    # Try the most likely runtime channels first, but keep the exact IBM errors.
    # That way a bad token is easier to debug than with one generic failure line.
    candidate_channels = []
    if instance:
        candidate_channels.append("ibm_cloud")
        candidate_channels.append("ibm_quantum_platform")
    else:
        candidate_channels.append("ibm_quantum_platform")
        candidate_channels.append("ibm_cloud")

    if api_token:
        service = None
        auth_errors = {}

        for channel in candidate_channels:
            try:
                service_kwargs = {"channel": channel, "token": api_token}
                if instance:
                    service_kwargs["instance"] = instance
                service = QiskitRuntimeService(**service_kwargs)
                print(f"  Using token-based account on channel: {channel}")
                break
            except Exception as exc:
                auth_errors[channel] = f"{type(exc).__name__}: {exc}"

        if service is None:
            raise RuntimeError(
                "Could not authenticate with the provided IBM Quantum token. "
                f"Tried: {auth_errors}"
            )
    else:
        saved_account_errors = {}
        service = None

        for channel in candidate_channels:
            try:
                service_kwargs = {"channel": channel}
                if instance:
                    service_kwargs["instance"] = instance
                service = QiskitRuntimeService(**service_kwargs)
                acct = service.active_account() or {}
                if acct.get("channel"):
                    print(f"  Using saved account on channel: {acct['channel']}")
                break
            except Exception as exc:
                saved_account_errors[channel] = f"{type(exc).__name__}: {exc}"

        if service is None:
            raise RuntimeError(
                "No usable IBM Quantum credentials were found. "
                f"Tried: {saved_account_errors}"
            )

    backend = service.least_busy(
        simulator=False, min_num_qubits=n_qubits, operational=True)

    print(f"  Device: {backend.name} ({backend.num_qubits} qubits)")

    circuits = []
    valid_indices = []
    start_values = initial_z_values(n_qubits, excited_qubit)

    for t_idx, t in enumerate(times):
        # We already know the t = 0 answer from the starting bitstring.
        if t == 0:
            continue
        dt = t / n_trotter_steps
        qc = build_trotter_circuit_2nd_order(
            n_qubits, J, h, dt, n_trotter_steps, excited_qubit
        )
        # Real hardware only gives counts back, so every circuit must be measured.
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
        if save_transpiled_diagram:
            save_circuit_diagram(
                transpiled[0],
                f"Transpiled hardware circuit on {backend.name}",
                "transpiled_hardware_circuit.png",
                90,
                0.55,
                (16, 7.0),
                140,
            )

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

# Show the exact, noisy, mitigated, and hardware heatmaps together.
# The extra panel is used for a short error summary so the figure tells the whole story.
def plot_comparison_panels(z_exact, z_noiseless, z_noisy, z_zne, z_hardware,
                           times, n_qubits, hw_name=None, hardware_note=None,
                           error_summary=None, filename=None):
    # Put the exact, noisy, mitigated, and hardware views in one place.
    # That makes the mitigation story easy to explain on one slide.
    if error_summary is None:
        error_summary = {}

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    datasets = [
        (z_exact, "Exact (matrix exponential)", None),
        (z_noiseless, "Noiseless Simulator", None),
        (z_noisy, "Noisy Simulator (depolarizing noise)", "Enable RUN_NOISY_SIM"),
        (z_zne, "ZNE estimate", "Enable RUN_ZNE_SIM"),
        (
            z_hardware,
            f"Real Hardware ({hw_name})" if hw_name else "Real Hardware",
            hardware_note or "Hardware run was not executed",
        ),
    ]

    im = None
    axes_list = list(axes.flat)

    for ax, (data, title, missing_hint) in zip(axes_list, datasets):
        if data is not None:
            im = ax.imshow(data, aspect='auto', origin='lower',
                          extent=[times[0], times[-1], -0.5, n_qubits-0.5],
                          cmap='RdBu_r', norm=norm)
        else:
            ax.text(0.5, 0.5, missing_hint,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-0.5, n_qubits - 0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Qubit index', fontsize=10)
        ax.set_yticks(range(n_qubits))

    summary_ax = axes_list[-1]
    summary_ax.axis('off')
    summary_ax.set_title("Error Summary", fontsize=12, fontweight='bold')

    method_colors = {
        "Noiseless": "#6b46c1",
        "Noisy": "#e24b4a",
        "ZNE": "#d69e2e",
        "Hardware": "#1d9e75",
    }
    ordered_methods = ["Noiseless", "Noisy", "ZNE", "Hardware"]
    labels = []
    values = []
    colors = []

    for method_name in ordered_methods:
        if method_name in error_summary:
            labels.append(method_name)
            values.append(error_summary[method_name])
            colors.append(method_colors[method_name])

    bars_ax = summary_ax.inset_axes([0.10, 0.44, 0.82, 0.44])
    bar_positions = np.arange(len(labels))
    bars_ax.barh(bar_positions, values, color=colors, alpha=0.88, height=0.6)
    bars_ax.set_yticks(bar_positions, labels=labels)
    bars_ax.invert_yaxis()
    bars_ax.set_xlabel(r'Average $|\Delta \langle Z \rangle|$', fontsize=9)
    bars_ax.tick_params(labelsize=9)
    bars_ax.grid(axis='x', alpha=0.2, linewidth=0.8)
    bars_ax.spines['top'].set_visible(False)
    bars_ax.spines['right'].set_visible(False)

    if values:
        max_value = max(values)
        bars_ax.set_xlim(0.0, max_value * 1.30)
        for idx, value in enumerate(values):
            bars_ax.text(
                value + max_value * 0.03,
                idx,
                f"{value:.4f}",
                va='center',
                ha='left',
                fontsize=9,
                fontweight='bold',
            )

    note_lines = ["Lower is better."]
    if "Noisy" in error_summary and "ZNE" in error_summary:
        noisy_error = error_summary["Noisy"]
        zne_error = error_summary["ZNE"]
        if noisy_error > 0:
            zne_improvement = 100.0 * (noisy_error - zne_error) / noisy_error
            note_lines.append(f"ZNE lowers the noisy-simulator error by {zne_improvement:.1f}%.")
    note_lines.extend([
        "The maps look similar because every panel",
        "is still showing the same TFIM dynamics.",
    ])

    summary_ax.text(
        0.06,
        0.28,
        "\n".join(note_lines),
        transform=summary_ax.transAxes,
        ha='left',
        va='top',
        fontsize=10.2,
        linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#f7fafc', edgecolor='#cbd5e0'),
    )

    fig.colorbar(im, ax=axes, label=r'$\langle Z_i \rangle$',
                 shrink=0.8, pad=0.02)
    plt.suptitle('TFIM: Exact, Noisy, Mitigated, and Hardware Results',
                 fontsize=14, y=1.02)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
    plt.close()


# Plot a few qubits directly as lines instead of full heatmaps.
# This helps when the differences are too subtle in the color plots.
def plot_qubit_traces(z_exact, z_noiseless, z_noisy, z_zne, z_hardware, times,
                      qubit_indices, filename=None):
    # Plot a few representative qubits as line traces instead of heatmaps.
    # This makes smaller differences between methods easier to point out.
    fig, ax = plt.subplots(figsize=(10, 6))

    for qi in qubit_indices:
        ax.plot(times, z_exact[qi], '-', color='#2b6cb0', linewidth=2, alpha=0.8)
        ax.plot(times, z_noiseless[qi], '-.', color='#6b46c1', linewidth=1.8, alpha=0.8)
        if z_noisy is not None:
            ax.plot(times, z_noisy[qi], '--', color='#e24b4a', linewidth=1.5, alpha=0.7)
        if z_zne is not None:
            ax.plot(times, z_zne[qi], ':', color='#d69e2e', linewidth=2.0, alpha=0.85)
        if z_hardware is not None:
            ax.plot(times, z_hardware[qi], 'o', color='#1d9e75',
                    markersize=3, alpha=0.6)

    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color='#2b6cb0', lw=2, label='Exact'),
        Line2D([0], [0], color='#6b46c1', lw=1.8, ls='-.', label='Noiseless simulator'),
    ]
    if z_noisy is not None:
        legend_lines.append(
            Line2D([0], [0], color='#e24b4a', lw=1.5, ls='--', label='Noisy simulator')
        )
    if z_zne is not None:
        legend_lines.append(
            Line2D([0], [0], color='#d69e2e', lw=2.0, ls=':', label='ZNE estimate')
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
    if z_zne is not None:
        title_parts.append('ZNE')
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


# Save a clean circuit figure when the optional drawer is available.
# If that drawer is missing, save a plain-text version instead.
def save_circuit_diagram(qc, title, filename, fold, scale, figure_size, text_fold):
    # Try the matplotlib drawer first for a cleaner figure.
    # If that optional drawer is missing, fall back to a plain-text circuit.
    try:
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
        text_filename = Path(filename).with_suffix(".txt")
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(qc.draw(output='text', fold=text_fold).single_string())
        print(f"    Saved: {text_filename}")
    except Exception:
        print("    (Could not save circuit diagram)")


if __name__ == "__main__":

    # Keep this small enough for hardware runs.
    n_qubits = 5
    J = 1.0
    h = 1.0
    excited_qubit = 2
    n_trotter_steps = 3
    t_max = 5.0
    n_time_points = 20
    times = np.linspace(0, t_max, n_time_points)

    # Pull the IBM credentials from the environment so we do not hardcode secrets.
    API_TOKEN = os.environ.get("QISKIT_IBM_TOKEN")
    if API_TOKEN is None:
        API_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN")

    API_INSTANCE = os.environ.get("QISKIT_IBM_INSTANCE")
    if API_INSTANCE is None:
        API_INSTANCE = os.environ.get("IBM_QUANTUM_INSTANCE")
    # These switches keep the script easy to trim down for the final report.
    RUN_NOISY_SIM = True
    RUN_ZNE_SIM = True
    ZNE_NOISE_SCALES = (1.0, 2.0, 3.0)
    SAVE_STEP_DIAGRAM = False
    SAVE_TRACE_PLOT = True
    SAVE_TRANSPILED_HARDWARE_DIAGRAM = True
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
    if SAVE_STEP_DIAGRAM:
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
    z_zne = None
    if RUN_ZNE_SIM:
        print("\nStep 3: Noisy simulator + zero-noise extrapolation")
        z_zne, scaled_runs = run_zne_simulation(
            n_qubits, J, h, excited_qubit, times, n_trotter_steps,
            noise_scales=ZNE_NOISE_SCALES,
        )
        if 1.0 in ZNE_NOISE_SCALES:
            scale_one_index = ZNE_NOISE_SCALES.index(1.0)
            z_noisy = scaled_runs[scale_one_index]
        elif RUN_NOISY_SIM:
            z_noisy = run_noisy_simulation(
                n_qubits, J, h, excited_qubit, times, n_trotter_steps)
        print("  Done.")
    elif RUN_NOISY_SIM:
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
                n_trotter_steps,
                api_token=API_TOKEN,
                instance=API_INSTANCE,
                save_transpiled_diagram=SAVE_TRANSPILED_HARDWARE_DIAGRAM,
            )
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

    error_summary = {}
    error_summary["Noiseless"] = np.mean(np.abs(z_exact - z_noiseless))
    if z_noisy is not None:
        error_summary["Noisy"] = np.mean(np.abs(z_exact - z_noisy))
    if z_zne is not None:
        error_summary["ZNE"] = np.mean(np.abs(z_exact - z_zne))
    if z_hardware is not None:
        error_summary["Hardware"] = np.mean(np.abs(z_exact - z_hardware))

    plot_comparison_panels(
        z_exact, z_noiseless, z_noisy, z_zne, z_hardware,
        times, n_qubits, hw_name, hardware_note, error_summary,
        'hardware_comparison.png'
    )

    if SAVE_TRACE_PLOT:
        plot_qubit_traces(z_exact, z_noiseless, z_noisy, z_zne, z_hardware, times,
                          [1, 2, 3], 'hardware_traces.png')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    err_sim = error_summary["Noiseless"]
    print(f"  Avg error (noiseless sim vs exact):  {err_sim:.6f}")
    if z_noisy is not None:
        err_noisy = error_summary["Noisy"]
        print(f"  Avg error (noisy sim vs exact):      {err_noisy:.4f}")
    if z_zne is not None:
        err_zne = error_summary["ZNE"]
        print(f"  Avg error (ZNE vs exact):            {err_zne:.4f}")
    if z_hardware is not None:
        err_hw = error_summary["Hardware"]
        print(f"  Avg error (hardware vs exact):       {err_hw:.4f}")
    output_files = [
        "circuit_diagram.png or circuit_diagram.txt",
        "hardware_comparison.png",
    ]
    if SAVE_STEP_DIAGRAM:
        output_files.append("circuit_step_diagram.png")
    if SAVE_TRACE_PLOT:
        output_files.append("hardware_traces.png")
    if SAVE_TRANSPILED_HARDWARE_DIAGRAM and z_hardware is not None:
        output_files.append("transpiled_hardware_circuit.png or .txt")
    print(f"\n  Files: {', '.join(output_files)}")
    print("=" * 60)
