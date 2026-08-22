"""quantum_decoherence_map.simulator — États intermédiaires après chaque porte.

Extrait le statevector après chaque porte du circuit (sans save_statevector :
on run sur des prefixes). Matrice densité ρ pour chaque étape. Bruit : optionnel
via dephasing simple (on le laisse propre, pas intential).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def extract_step_states(qc: QuantumCircuit) -> list[dict]:
    """Rerun le circuit progressivement et capture le statevector à chaque étape."""
    def _sv_list(sv: Statevector) -> list:
        return [[float(np.real(v)), float(np.imag(v))] for v in sv.data]
    states = [{"step": 0, "op": "init", "statevector": _sv_list(Statevector.from_instruction(qc))}]
    prefixes = QuantumCircuit(qc.num_qubits)
    for step, inst in enumerate(qc.data, start=1):
        op, qargs = inst.operation, inst.qubits
        prefixes.append(inst)
        sv = Statevector.from_instruction(prefixes)
        states.append({
            "step": step,
            "op": op.name,
            "qubits": [qc.find_bit(q).index for q in qargs],
            "statevector": _sv_list(sv),
        })
    return states


def state_to_density(sv: np.ndarray) -> np.ndarray:
    return np.outer(sv, np.conjugate(sv))


def run(qc: QuantumCircuit, out_dir: str | Path) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    states = extract_step_states(qc)
    for st in states:
        sv = np.array(st["statevector"], dtype=complex)
        rho = state_to_density(sv)
        diag = rho.diagonal()
        st["density"] = [
            [float(np.real(v)), float(np.imag(v))] for v in diag
        ]
    json.dump(states, open(out / f"states.json", "w"))
    return states
