"""quantum_decoherence_map.circuit_builder — Circuit quantique de test.

5 qubits, 10 portes (H, CNOT, T, S). Bruit optionnel : T1, T2, erreurs de porte.
Sauvegarde QASM. Nouveau module, pas de fichier existant touché.
"""
from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit


def build_circuit(n_qubits: int = 5) -> QuantumCircuit:
    """Circuit de démonstration : entangle tout le monde puis mesure partielle."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    for i in range(n_qubits):
        qc.t(i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def save_qasm(qc: QuantumCircuit, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(qc.qasm())
    return path
