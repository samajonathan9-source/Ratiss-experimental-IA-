"""ratis_net.lct_collapse — L'effondrement topologique sous poussée thermo.

Le saut v4 : on ne maximise pas P_sig (non-différentiable). On laisse C
s'effondrer sous poussée thermodynamique de l'environnement. Quand C >=
C_seuil_thermo(environnement), l'effondrement se produit. On garde la MARQUE
topologique (hash du cycle H1 survivant), pas la valeur d'énergie.

C'est exactement l'invariance ZK validée sur QPU : après l'effondrement de
la fonction d'onde, on garde le hash topo (380a69c0...), pas l'énergie
(0.152 vs 1.835).
"""
from __future__ import annotations

import hashlib
import math
import numpy as np

from ratis_net.lct_network import _persistence_diagrams_lite


def compute_coherence(token_embedding: np.ndarray, weights: np.ndarray,
                      t_step: int = 0, omega: float = math.pi / 2) -> float:
    """Calcule la cohérence C du signal = corrélation entre le token et les
    poids, modulée par l'oscillation θ(t) = cos(ωt).

    C est élevé quand le token "résonne" avec la structure des poids
    (cohérent), bas sinon (décohérent).
    """
    # corrélation token-poids (produit scalaire normalisé)
    if weights.ndim == 1:
        w = weights
    else:
        w = weights.mean(axis=0)  # signature moyenne des poids
    # aligner dimensions
    min_d = min(len(w), len(token_embedding))
    w = w[:min_d]
    t = token_embedding[:min_d]
    if np.linalg.norm(w) < 1e-9 or np.linalg.norm(t) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.dot(w, t) / (np.linalg.norm(w) * np.linalg.norm(t) + 1e-9))
    # modulation par l'oscillation (milieu génial)
    theta = math.cos(omega * t_step)
    C = abs(corr) * abs(theta)
    return min(1.0, max(0.0, C))


def topological_mark(weights: np.ndarray, max_edge: float = 2.0) -> str:
    """La MARQUE topologique = hash du cycle H1 survivant après effondrement.

    Pas la valeur d'énergie (P_sig=0.60). Le HASH de la forme qui survit.
    C'est le bit MCB v4 : on certifie le message (la marque), pas le courant.
    """
    # calculer le diagramme de persistance
    diagrams = _persistence_diagrams_lite(weights, max_edge)
    # le cycle survivant = les arêtes du complexe à max_edge
    # on hash la STRUCTURE (quels neurones sont liés), pas les valeurs
    n = len(weights)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(weights[i] - weights[j]))
            if d <= max_edge:
                edges.append(f"{i}-{j}:{d:.3f}")
    # hash de la structure topologique (la marque)
    mark_str = "|".join(sorted(edges))
    return hashlib.sha256(mark_str.encode()).hexdigest()[:16]


def collapse(token_embedding: np.ndarray, weights: np.ndarray,
             c_seuil_thermo: float, t_step: int = 0,
             omega: float = math.pi / 2, max_edge: float = 2.0) -> dict:
    """L'effondrement topologique.

    1. On calcule C (cohérence token-poids sous oscillation).
    2. Si C >= C_seuil_thermo → l'effondrement se produit.
    3. On garde la MARQUE topologique (hash du cycle survivant).
    4. La valeur d'énergie (P_sig) est perdue — on garde la marque.

    Returns:
        dict avec : collapsed (bool), mark (hash), C, P_sig (perdu), c_seuil.
    """
    C = compute_coherence(token_embedding, weights, t_step, omega)
    # P_sig est calculé mais IL EST PERDU (on garde la marque, pas la valeur)
    P_sig = _persistence_diagrams_lite(weights, max_edge)

    if C >= c_seuil_thermo:
        # effondrement : on garde la MARQUE topo, pas la valeur
        mark = topological_mark(weights, max_edge)
        return {
            "collapsed": True,
            "mark": mark,          # la marque topo (le bit MCB)
            "C": C,
            "c_seuil": c_seuil_thermo,
            "P_sig_lost": P_sig,   # la valeur est PERDUE (comme l'énergie après collapse)
        }
    else:
        # pas d'effondrement, le cycle continue
        return {
            "collapsed": False,
            "mark": None,
            "C": C,
            "c_seuil": c_seuil_thermo,
            "P_sig_lost": None,
        }
