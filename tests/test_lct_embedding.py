"""Test réel : la couche d'embedding doit réduire cos(centroïdes) < 0.7.
Convention du dépôt : python tests/test_lct_embedding.py"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ratis_net.lct_modules.lct_embedding import TopologicalEmbedding


def _centroid_cos(embed_fn, samples, n_classes):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for x, label in samples:
        sums[label] = sums.get(label, np.zeros_like(embed_fn(x))) + embed_fn(x)
        counts[label] += 1
    c = {k: sums[k] / counts[k] for k in sums}
    classes = sorted(c)
    return max(float(np.dot(c[a] / (np.linalg.norm(c[a]) + 1e-9),
                            c[b] / (np.linalg.norm(c[b]) + 1e-9)))
               for i, a in enumerate(classes) for b in classes[i + 1:])


def main() -> None:
    rng = np.random.RandomState(42)
    n_classes, dim_in, n_samples = 3, 10, 300
    # 3 clusters : cartes linéairement separables dans un espace riche
    protos = [rng.normal(0, 1, dim_in) for _ in range(n_classes)]
    samples = []
    for c in range(n_classes):
        for _ in range(n_samples // n_classes):
            samples.append((protos[c] + rng.normal(0, 0.6, dim_in), c))

    emb = TopologicalEmbedding(n_in=dim_in, n_emb=8, eta=0.3, seed=42)
    before = _centroid_cos(emb.embed, samples, n_classes)

    # auto-consistency : cible = reconstruction de x (mesurée, better than fixe)
    for ep in range(200):
        for x, label in samples:
            emb.train_target(x, x[:emb.n_emb], P_sig=0.5)
    after = _centroid_cos(emb.embed, samples, n_classes)

    print(f"INIT cos_max={before:.3f} -> APRES {after:.3f}")
    assert after < 0.7, f"entrainement réduit cos<0.7 (obtenu {after:.3f})"
    print("TOUT OK — embedding séparable par apprentissage LCT")


if __name__ == "__main__":
    main()
