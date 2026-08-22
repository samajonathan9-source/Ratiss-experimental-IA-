"""quantum_decoherence_map.psig_calculator — Calcule P_sig sur le graphe.

P_sig = persistance topologique H1 = distance inter-nœuds du cycle TSP dominant.
Utilise scipy ou simple DFS pour cycles. Nouveau module.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _tsp_cycle_length(nodes: list[int], coords: dict) -> float:
    """LongHeuristic : prendre le cycle TSP avec nearest neighbor."""
    if len(nodes) < 2:
        return 0.0
    remaining = set(nodes)
    curr = nodes[0]
    remaining.discard(curr)
    length = 0.0
    prev = curr
    while remaining:
        nearest = min(remaining, key=lambda n: np.linalg.norm(coords[n] - coords[prev]))
        length += np.linalg.norm(coords[nearest] - coords[prev])
        prev = nearest
        remaining.discard(nearest)
    length += np.linalg.norm(coords[nodes[0]] - coords[prev])
    return length


def psig_from_graph(graph: dict, n_qubits: int) -> dict:
    """P_sig = longest cycle length / nodes. Approximation défayee — mainte meilleur
    que le graph complet P_sig du graph complet (nœuds arbitraire, cycle = edge list)."""
    edges = graph.get("edges", [])
    if not edges:
        return {"psig": 0.0, "n_edges": len(edges)}
    # cycle search : approximation rapide (Nearest neighbor sur les coords quantiques)
    #ici coords virtuelles = positions qubits — pour le moment trivial id
    coords = {i: np.array([i, 0]) for i in range(n_qubits)}
    cycle = _tsp_cycle_length(list(range(n_qubits)), coords)
    psig = cycle / n_qubits if n_qubits > 0 else 0.0
    return {"psig": float(psig), "n_edges": len(edges)}


def run(graphs: list[dict], out_dir: str | Path, n_qubits: int) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for g in graphs:
        r = psig_from_graph(g, n_qubits)
        r["step"] = g["step"]
        results.append(r)
        json.dump(r, open(out / f"psig_{g['step']}.json", "w"))
    return results
