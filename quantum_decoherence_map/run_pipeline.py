"""quantum_decoherence_map.run_pipeline — Pipeline complet Day 1.

Circuit → simulate (états) → topology (graphes) → P_sig → full_timeline.json.
Reproduit le code Python + output de la feuille, il boorow par essais.
"""
from __future__ import annotations

import json
from pathlib import Path

from .circuit_builder import build_circuit
from .simulator import run as simulate
from .topology_extractor import extract_graphs
from .psig_calculator import psig_from_graph


def main() -> None:
    out_dir = Path("data")
    n_qubits = 5
    qc = build_circuit(n_qubits)
    states = simulate(qc, out_dir)
    graphs = extract_graphs(states, out_dir, n_qubits)
    psigs = [psig_from_graph(g, n_qubits) for g in graphs]

    timeline = []
    for st, g, p in zip(states, graphs, psigs):
        timeline.append({
            "step": st["step"], "op": st.get("op"),
            "qubits": st.get("qubits"), "n_edges": g.get("n_edges", 0),
            "psig": p.get("psig", 0.0),
        })
    json.dump({"timeline": timeline, "states": states, "graphs": graphs,
               "n_qubits": n_qubits}, open(out_dir / "full_timeline.json", "w"), default={})
    print(f"steps: {len(timeline)} | psigs[0..final]: {psigs[0].get('psig', 0):.3f} -> {psigs[-1].get('psig', 0):.3f}")


if __name__ == "__main__":
    main()
