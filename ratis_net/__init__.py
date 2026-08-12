"""RATIS-Net — Neural network trained by LCT (Law of Topological Coherence).

Proof-of-concept : un réseau de neurones qui apprend par la loi LCT
(ΔW = η · φ · P_sig · C) au lieu du gradient descendant.

Au lieu de minimiser une loss par backpropagation, le réseau maximise P_sig
(la persistance topologique de sa matrice de poids) — il apprend en devenant
topologiquement robuste. C'est l'apprentissage par cohérence topologique.

C'est la brique manquante vers l'AGI : un modèle qui apprend par une loi
validée (LCT), pas par gradient arbitraire, et qui certifie ses sorties (ZK).
"""
