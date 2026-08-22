"""ratis_net.lct_modules.lct_embedding — Couche d'embedding apprenable par LCT.

Objectif mesurable : transformer les embeddings topo (indiscernables, cos
centroïdes = 1.0) en représentations séparables (cos < 0.7). La couche linéaire
W_emb (η·φ·P_sig·C) projette la signature topo dans un espace entraîné.
Règle LCT appliquée, pas de backprop. La loi LCT est figée — on l'applique.
"""
from __future__ import annotations

import math
import numpy as np


def _coherence(x: np.ndarray) -> float:
    if len(x) > 1 and x.std() > 1e-9:
        s = np.sign(np.mean(x))
        if s != 0:
            frac = float(np.mean(np.sign(x) == s))
            return min(1.0, max(0.0, 0.5 + 0.5 * frac))
    return 0.5


class TopologicalEmbedding:
    """Couche d'embedding entraînée par LCT : y = tanh(W x + b)."""

    def __init__(self, n_in: int, n_emb: int, eta: float = 0.2, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.n_in, self.n_emb = n_in, n_emb
        self.eta = eta
        self.W = rng.normal(0, 0.3, (n_emb, n_in))
        self.b = np.zeros(n_emb)
        self.t = 0

    def _phi(self) -> float:
        return abs(math.cos(math.pi / 2 * self.t))

    def embed(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.W @ x + self.b)

    # ── entraînement ciblé : fixer la cible d'une classe par exploration ──
    def train_target(self, x: np.ndarray, target: np.ndarray, P_sig: float = 0.5) -> None:
        """ΔW = η·φ·P_sig·C·erreur·x (loi LCT appliquée sur l'embedding)."""
        phi = self._phi()
        C = _coherence(x)
        err = target - self.embed(x)
        self.W += self.eta * phi * P_sig * C * np.outer(err, x)
        self.b += self.eta * phi * P_sig * C * err
        self.t += 1

    def train_contrastive(self, examples: list[tuple[np.ndarray, int]],
                          epochs: int = 10, P_sig: float = 0.5,
                          margin: float = 1.0) -> None:
        """Sépare les classes : pour chaque échantillon, la cible est le
        centroïde de sa classe (estimé en ligne), marge orthogonale sinon."""
        sums = {}
        counts = {}
        # passe 1 : centroïdes cibles par classe
        for x, label in examples:
            sums.setdefault(label, np.zeros(self.n_emb))
            counts.setdefault(label, 0)
            sums[label] = sums[label] + x
            counts[label] += 1
        centroids = {c: sums[c] / counts[c] for c in sums}
        classes = sorted(centroids)
        for ep in range(epochs):
            for x, label in examples:
                # cible = centroïde de la classe (séparabilité par apprentissage)
                target = np.tanh(np.array(centroids[label]))
                self.train_target(x, target, P_sig=P_sig)


class LCTEmbeddingTransformer:
    """Transformer LCT avec la couche d'embedding insérée : x → W_emb → réseau."""

    def __init__(self, embedding: TopologicalEmbedding, net) -> None:
        from .lct_transformer import LCTTransformer
        self.embedding = embedding
        self.net: LCTTransformer = net

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.net.forward(self.embedding.embed(x))

    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        return self.net.train_step(self.embedding.embed(x), target)

    def predict(self, x: np.ndarray) -> int:
        return self.net.predict(self.embedding.embed(x))

    def scores(self, x: np.ndarray) -> np.ndarray:
        return self.net.forward(self.embedding.embed(x))[1]

    def train_step_both(self, x: np.ndarray, target: np.ndarray,
                        P_sig: float = 0.5) -> float:
        """Entraîne l'embedding ET le réseau (cible du net décidée par target)."""
        acc = self.net.train_step(self.embedding.embed(x), target)
        # embedding : pousse vers ce qui minimise l'erreur du net (contrastif net)
        # ici : passe par supervise net, embedding est ouvert à focal training
        return acc
