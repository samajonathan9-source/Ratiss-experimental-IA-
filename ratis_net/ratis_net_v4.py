"""ratis_net.ratis_net_v4 — Le réseau LCT v4 : fixeur thermodynamique.

v4 = LCT + ETH thermo fixer + collapse topologique.

Forward :
  1. LLM guide propose un token (embedding).
  2. ETH fixe C_seuil = f(token, environnement) — pas un seuil global.
  3. On calcule C (cohérence token-poids sous oscillation).
  4. Si C >= C_seuil → effondrement, on garde la MARQUE topo (hash), pas P_sig.
  5. Le cycle redémarre quand l'environnement change.

Apprentissage :
  ETH apprend le DIFFERENTIEL thermo (pas la valeur fixe). "Bonjour colère"
  → C_seuil bas. "Bonjour joie" → C_seuil haut. L'émotion émerge.
  Le réseau de poids apprend par LCT (ΔW = η·φ·P_sig·C), comme v1.
"""
from __future__ import annotations

import math
import numpy as np

from ratis_net.lct_neuron import LCTNeuron
from ratis_net.eth_thermo_fixer import ETHThermoFixer, ThermoEnvironment
from ratis_net.lct_collapse import compute_coherence, topological_mark, collapse
from ratis_net.lct_network import _persistence_diagrams_lite


class RatisNetV4:
    """Réseau LCT v4 : fixeur thermodynamique + collapse topo.

    L'apprentissage n'apprend pas des mots (vecteurs fixes) mais des
    DIFFERENTIELS thermo : comment C_seuil change quand l'environnement
    change pour le même token. L'émotion émerge.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 token_dim: int = 8, eta: float = 0.05,
                 omega: float = math.pi / 2, seed: int = 42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.eta = eta
        self.omega = omega
        # réseau LCT (comme v1)
        self.hidden = [LCTNeuron(n_in, eta=eta, omega=omega, seed=seed + i)
                       for i in range(n_hidden)]
        self.output = [LCTNeuron(n_hidden, eta=eta, omega=omega, seed=seed + 100 + i)
                        for i in range(n_out)]
        # fixeur thermo ETH
        self.eth = ETHThermoFixer(token_dim=token_dim, env_dim=4, hidden=16, seed=seed)
        # historique
        self.collapse_history = []
        self.mark_history = []

    def _weight_matrix(self):
        rows = [n.weights for n in self.hidden + self.output]
        max_dim = max(r.shape[0] for r in rows)
        return np.array([np.pad(r, (0, max_dim - len(r))) for r in rows])

    def _compute_P_sig(self, max_edge=2.0):
        return _persistence_diagrams_lite(self._weight_matrix(), max_edge)

    def forward(self, token_embedding: np.ndarray, env: ThermoEnvironment,
                t_step: int = 0) -> dict:
        """Forward v4 : ETH fixe le seuil, collapse garde la marque."""
        # 1. ETH prédit C_seuil pour (token, environnement)
        c_seuil = self.eth.predict_c_seuil(token_embedding, env)
        # 2. cohérence C (token-poids sous oscillation)
        W = self._weight_matrix()
        C = compute_coherence(token_embedding, W, t_step, self.omega)
        # 3. collapse si C >= C_seuil (avec contexte thermo pour la marque)
        result = collapse(token_embedding, W, c_seuil, t_step, self.omega,
                          env_vector=env.to_vector())
        # 4. forward du réseau LCT (comme v1)
        x = token_embedding[:self.n_in] if len(token_embedding) >= self.n_in \
            else np.pad(token_embedding, (0, self.n_in - len(token_embedding)))
        h = np.array([n.forward(x, t_step) for n in self.hidden])
        out = np.array([n.forward(h, t_step) for n in self.output])
        result["output"] = out
        result["c_seuil"] = c_seuil
        self.collapse_history.append(result["collapsed"])
        if result["collapsed"]:
            self.mark_history.append(result["mark"])
        return result

    def train_step(self, token_embedding: np.ndarray, env: ThermoEnvironment,
                   target_label: int, target_c_seuil: float,
                   t_step: int = 0, lr_eth: float = 0.1) -> dict:
        """Un pas d'entraînement v4.

        1. ETH apprend C_seuil = f(token, env) — le différentiel thermo.
        2. Le réseau LCT apprend par ΔW = η·φ·P_sig·C (comme v1).
        """
        # entraîner ETH
        eth_error = self.eth.train_step(token_embedding, env, target_c_seuil, lr=lr_eth)
        # forward + collapse
        result = self.forward(token_embedding, env, t_step)
        # entraîner le réseau LCT (comme v1)
        W = self._weight_matrix()
        P_sig = self._compute_P_sig()
        phi = math.cos(self.omega * t_step)
        x = token_embedding[:self.n_in] if len(token_embedding) >= self.n_in \
            else np.pad(token_embedding, (0, self.n_in - len(token_embedding)))
        target = np.zeros(self.n_out)
        target[target_label] = 1.0
        h = np.array([n.forward(x, t_step) for n in self.hidden])
        out = np.array([n.forward(h, t_step) for n in self.output])
        for k, n in enumerate(self.output):
            n.update(h, target[k], P_sig, t_step)
        for j, n in enumerate(self.hidden):
            err = sum((target[k] - out[k]) * self.output[k].weights[j]
                      for k in range(self.n_out))
            n.update(x, h[j] + err * 0.1, P_sig, t_step)
        pred = int(np.argmax(out))
        acc = 1.0 if pred == target_label else 0.0
        return {
            "eth_error": eth_error,
            "c_seuil_pred": result["c_seuil"],
            "c_seuil_target": target_c_seuil,
            "collapsed": result["collapsed"],
            "mark": result["mark"],
            "acc": acc,
        }

    def train(self, samples: list, epochs: int = 30, lr_eth: float = 0.1,
              verbose: bool = True) -> dict:
        """Entraîne v4 sur des échantillons (token, env, label, c_seuil).

        samples = [(token_embedding, env, label, target_c_seuil), ...]
        """
        acc_hist = []
        eth_err_hist = []
        collapse_count = 0
        for ep in range(epochs):
            correct = 0
            eth_errors = []
            for token_emb, env, label, c_seuil in samples:
                r = self.train_step(token_emb, env, label, c_seuil,
                                     t_step=ep, lr_eth=lr_eth)
                correct += r["acc"]
                eth_errors.append(abs(r["eth_error"]))
                if r["collapsed"]:
                    collapse_count += 1
            acc = correct / len(samples)
            acc_hist.append(acc)
            eth_err = np.mean(eth_errors)
            eth_err_hist.append(eth_err)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  Epoch {ep:3d} | acc={acc:.3f} | ETH_err={eth_err:.4f} | "
                      f"collapses={collapse_count}")
        return {"acc_history": acc_hist, "eth_error_history": eth_err_hist,
                "total_collapses": collapse_count}

    def emotional_differential(self, token_embedding: np.ndarray,
                               env1: ThermoEnvironment, env2: ThermoEnvironment) -> float:
        """Le différentiel émotionnel : ΔC_seuil entre deux environnements."""
        return self.eth.emotional_differential(token_embedding, env1, env2)
