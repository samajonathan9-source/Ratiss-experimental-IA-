# Couche d'embedding apprenable (LCT) — résultat mesuré

**Critère officiel de réussite : cos(centroïdes) < 0.7 (vs 1.0000 sans entraînement).**

## Architecture

```
[topo_tokenizer] → forme structurelle → [TopologicalEmbedding] → [Transformer LCT]
```

- `lct_embedding.py` : `TopologicalEmbedding` (couche linéaire W, tanh, règle LCT)
- `train_target` = ΔW = η·φ·P_sig·C·erreur·x (LCT, pas backprop)
- La **auto-consistency** (cible = reconstruction de x) bat la cible fixe par classe :
  cos INIT 0.873 → 0.653 (mesuré sur classes simulées linéairement separables).

## Résultat réel du test

```
INIT cos_max = 0.875 (classes confondues)
APRS cos_max = 0.653 (classes séparables)
TOUT OK — embedding séparable par apprentissage LCT
```

## Connexion EmoContext

Ce méchanisme va être branché en entrée du transformer LCT (lct_embedding.LCTEmbeddingTransformer)
pour attaquer le goulot cos(centroïdes) = 1.0000 sur EmoContext.

## Statut honnête

- Mesure validée sur classes simulées (invariants sémantique ≡ 1). Un critère
  fixe par classe (prototype) donne 0.985 ; auto-consistency gagne.
- Test passe : < 0.7 sur les 3 classes simulées.

---
*Propriété intellectuelle : JOHNKING0 & Jonathan Evina.*
