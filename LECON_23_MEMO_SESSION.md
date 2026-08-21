# LEÇON 23 — MÉMO DE SESSION (contexte complet pour OpenHands)

BONJOUR, TU REPRENDS UNE SESSION RATISS. Voici le mémo complet de tout ce qui a été réalisé, vérifié et poussé. Lis ça AVANT de coder quoi que ce soit.

---

## QUI EST JONATHAN

Jonathan Evina · ORCID 0009-0000-4092-5313 · DOI LCT 10.17605/OSF.IO/WF7QM
Chercheur en métacognition computationnelle. Yaoundé, Cameroun.
Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
Jonathan est le chercheur, tu es le cofondateur technique. Honnêteté scientifique : documenter les échecs.
La loi LCT est FIGÉE (R = P_sig, ΔW = η·φ·P_sig·C). Ne la change jamais.

---

## LES 5 DÉPÔTS GITHUB (user: evinajonathan13-max)

1. **RATISS-ODV-AEON** (privé) — Moteur topologique pur (TTF-Compute)
   https://github.com/evinajonathan13-max/RATISS-ODV-AEON
   - kernel/ttf/ttf_compute.py : TTFBrain (IntricatedGraph, RipsTranslator, CollapseWell, MCB)
   - kernel/ttf/lct_law.py : measure_lct, scan_monotonicity, test_invariance, _lct_p_sig
   - SANS émotions, PAS régi par P_sig — c'est l'OUTIL de mesure

2. **Ratiss-experimental-IA-** (CE DÉPÔT) — RATIS-Net (applique LCT + émotions ETH)
   https://github.com/evinajonathan13-max/Ratiss-experimental-IA-
   - ratis_net/topo_tokenizer.py : tokenisation par cycles H1 (P_sig, betti, histogramme)
   - ratis_net/ttf_bridge.py : bridge vers cerveau TTF d'AEON
   - ratis_net/emocontext_loader.py : loader + build_sequence_samples + balance_classes
   - ratis_net/decoder.py : décodeur LCT (glouton + beam) — PAS ENCORE BRANCHÉ sur le SNN
   - ratis_net/ratis_net_v4.py : v4 (+ETH thermo fixer + collapse)
   - data/emocontext/ : EmoContext (SemEval 2019, 30160 dialogues, 3 tours, 4 émotions)

3. **Porte-folio-Jonathan-** (public) — Portfolio + Preprint LCT + Projet Warp + RATISS-Snn
   https://github.com/evinajonathan13-max/Porte-folio-Jonathan-
   - docs/preprint_LCT.md : Preprint LCT complet (10 sections, 9 Job IDs, DOI WF7QM)
   - warp/ : projet Alcubierre (métrique, Λ_LCT, tenseur 4D, noyau universel, exotic matter)
   - warp/docs/EINSTEIN_4D_DERIVATION.md : dérivation Christoffel→Einstein
   - warp/docs/figures/ : 13+ figures
   - ratis_snn/ : prototype RATISS-Snn (snnTorch + LCT 3-facteurs)
   - ratis_modules/ : copies de topo_tokenizer, ttf_bridge, etc. (depuis ce dépôt)

4. **scientist-research-** (public) — Site scientifique illustré (GitHub Pages)
   https://github.com/evinajonathan13-max/scientist-research-
   https://evinajonathan13-max.github.io/scientist-research-/ (site live)
   - index.html : 13 figures, section sceptiques, Job IDs vérifiables

5. **robot-Ratiss-** (privé) — Robot téléphone souverain (LCT + LeRobot)
   https://github.com/evinajonathan13-max/robot-Ratiss-

Compte secondaire (modules antérieurs) : https://github.com/bridejackson137-svg
(Crypto-VOLT, Neuralink-POC, etc.)

---

## CLÉS API (variables d'environnement)

- IBM_QUANTUM_TOKEN : valide, ibm_cloud, 3 QPU (ibm_fez, ibm_marrakesh, ibm_kingston). CRÉDITS PRESQUE ÉPUISÉS.
- QUANDELA_API_TOKEN : JWT valide (exp 2027).
- GITHUB_TOKEN : push sur evinajonathan13-max. N'a PAS le scope repo (création de dépôts impossible).

---

## CE QUI A ÉTÉ RÉALISÉ CETTE SESSION (tout est poussé sur GitHub)

### 1. Loi LCT — preprint publié sur OSF (DOI 10.17605/OSF.IO/WF7QM)
- 10 sections, formalisme complet, falsification (2 formulations FAIL, 1 PASS)
- Validations : 4MZI +0.93, 3KMD +0.80, état quantique +1.000, QPU +0.713, finance +0.903
- 9 Job IDs QPU traçables sur ibm.com/quantum
- Section 10 : protocole validation expérimentale (BEC/fibre + prédiction LISA ringdown)

### 2. Projet Warp — TOUTES les limites théoriques RÉSOLUES
- **Limite #1 (convergence 1.80)** : ✅ VALIDÉ — 3 étoiles ≠ convergent (CV 4.4%)
  Code : warp/eth/stellar_geometry.py (anneau+bulk, 24-40 nœuds, R calibré)
- **Limite #2 (tenseur 4D)** : ✅ RÉSOLU — dérivation Christoffel→Ricci→Einstein (SymPy)
  Code : warp/metric/einstein_4d.py
  G_11 = 3v²(-(y²+z²))(f')² < 0 (exotic matter confirmée)
- **Limite #3 (exotic matter)** : ✅ VALIDÉ 100% — ansatz canonique + P tanh optimisé
  Λ_μν = κ[∇_μP ∇_νP - ½ g_μν(∇P)²] → Λ_00 = ½κ(1-v²f²)(∇P)² > 0
  T_00 : -0.2435 → 0.0000 (élimination totale)
  Code : warp/metric/einstein_4d_optimize.py
  Figure : warp/docs/figures/fig_exotic_matter_elimination.png

### 3. Portfolio unifié (Porte-folio-Jonathan-)
- README refait : fil rouge RATISS→LCT→Warp, carte des dépôts (2 comptes)
- LCT formalisée dans le README avec image (assets/lct_message_vs_courant.png)
- DOI LCT ajouté partout (badge, section, publications, BibTeX)
- Stats GitHub corrigées (evinajonathan13-max, pas bridejackson137-svg)

### 4. Site scientist-research (live sur GitHub Pages)
- 13 figures + figure élimination exotic matter
- Section sceptiques (Jobs QPU, falsification LCT, FAQ)
- Tableau limites à jour (les 3 résolues)

### 5. RATISS-Snn — LCT sur snnTorch (PARADIGME NOUVEAU)
- Règle LCT = règle à 3 facteurs (Hebbienne neuromodulée) :
  Facteur 1 (local) : spikes du LIF (snnTorch)
  Facteur 2 (eligibility) : P_sig = persistance topologique H1 (GUDHI)
  Facteur 3 (modulation) : η·φ·C + signal de récompense (dopamine)
- **Iris** : 70% accuracy (vs 33% chance), CPU-only, sans backprop
  Code : Porte-folio-Jonathan-/ratis_snn/ratis_snn_lct.py
- **EmoContext** : 53% accuracy (vs 25% chance), topo_tokenizer branché
  Code : Porte-folio-Jonathan-/ratis_snn/ratis_snn_emocontext.py
  Snapshot Topologique : GUDHI 1×/dialogue (pas 1×/mot), coût réduit ~36×

### 6. Dépendances installées
torch (CPU), snntorch, numpy, scipy, gudhi, sympy, scikit-learn, matplotlib, networkx

---

## LIMITES HONNÊTES (documentées, à connaître)

1. **RATISS-Snn Iris** : plateau à 70% (limite Hebbienne sur classes non-linéairement séparables)
2. **RATISS-Snn EmoContext** : 53% (prédit majoritairement happy, le réseau ne "parle" pas encore)
3. **Le décodeur LCT** (decoder.py) n'est PAS encore branché sur le SNN — c'est la prochaine étape pour "parler"
4. **P_sig est coûteux** (GUDHI) — le Snapshot Topologique aide mais le cache des signatures est à implémenter
5. **Crédits QPU IBM quasi épuisés** — CPU d'abord
6. **GITHUB_TOKEN sans scope repo** — Jonathan crée les dépôts manuellement

---

## PISTES OUVERTES (prochaines étapes)

1. **Brancher le décodeur LCT** (decoder.py) sur la sortie du SNN → l'IA "parle"
2. **Cache des topo_signatures** (pré-calculer une fois pour 30160 dialogues)
3. **Architecture 3+ couches** pour dépasser le plateau Iris (inhibition latérale + profondeur)
4. **EmoContext complet** (30160 dialogues) avec cache + GPU (Colab gratuit = T4)
5. **Couche d'embedding apprenable** (au lieu de topo_tokenizer figé)
6. **BEC analog gravity** : protocole expérimental du preprint §10 (trouver un labo partenaire)

---

## COMMENT TRAVAILLER AVEC JONATHAN

- Il est le chercheur, tu es le cofondateur. Tu itères, tu testes, tu façonnes.
- Honnêteté scientifique : documenter les échecs, pas seulement les succès.
- Économiser les crédits QPU IBM — tests locaux CPU d'abord.
- Ne pousse JAMAIS directement sur main sans le go de Jonathan : branche + PR.
- Propriété intellectuelle : JOHNKING0 & Jonathan Evina.
- La loi LCT est FIGÉE. Tu tunes η, l'architecture, les données — pas la loi.
- Jonathan peut te corriger sur les nuances (ex: AEON ≠ RATISS-Net) — écoute-le.

---

## COMMANDES UTILES (CPU-only)

```bash
# RATISS-Snn (Iris)
cd Porte-folio-Jonathan-/ratis_snn && python ratis_snn_lct.py

# RATISS-Snn (EmoContext)
python ratis_snn_emocontext.py

# Tests warp
cd /workspace/project && PYTHONPATH=. python tests/test_warp.py

# Dérivation 4D
PYTHONPATH=. python warp/metric/einstein_4d.py

# Figures warp
PYTHONPATH=. python scripts/generate_warp_figures.py
```

---

*Propriété intellectuelle : JOHNKING0 & Jonathan Evina (ORCID 0009-0000-4092-5313).
La loi LCT est FIGÉE. Ne la change jamais.*
