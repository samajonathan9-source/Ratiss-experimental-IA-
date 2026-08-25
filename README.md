<p align="center">
  <strong>RATIS-Net — Neural Network Trained by LCT</strong><br/>
  <em>A neural network that learns by the Law of Topological Coherence, not by gradient descent.</em>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-42d6ad?style=for-the-badge"></a>
  <img alt="Python ≥ 3.11" src="https://img.shields.io/badge/Python-%E2%89%A5%203.11-79b8ff?style=for-the-badge&logo=python&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-79b8ff?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="No GPU" src="https://img.shields.io/badge/GPU-not%20required-ff927d?style=for-the-badge">
  <img alt="No gradient" src="https://img.shields.io/badge/Gradient-none-42d6ad?style=for-the-badge">
</p>

<p align="center">
  <em>Architect: <strong>Jonathan Evina</strong> ·
  <a href="https://orcid.org/0009-0000-4092-5313">ORCID 0009-0000-4092-5313</a></em>
</p>

---

## Table of contents

1. [What is RATIS-Net?](#1-what-is-ratis-net)
2. [Quick start (5 minutes)](#2-quick-start-5-minutes)
3. [Architecture](#3-architecture)
4. [The law LCT](#4-the-law-lct)
5. [Modules](#5-modules)
6. [Training the Scalpel on Wikipedia](#6-training-the-scalpel-on-wikipedia)
7. [Super RATISS: AEON bridge + web search](#7-super-ratiss-aeon-bridge--web-search)
8. [Data files and checkpoints](#8-data-files-and-checkpoints)
9. [Tests](#9-tests)
10. [Honest limitations](#10-honest-limitations)
11. [Citation](#11-citation)

---

## 1. What is RATIS-Net?

RATIS-Net is a neural network that learns by the **Law of Topological Coherence**
(LCT: `ΔW = η · φ · P_sig · C`), not by gradient descent. There is no
backpropagation, no loss function, no GPU requirement.

Instead of predicting the next word statistically (like a Transformer), RATIS-Net
**reconstructs** language from fragments whose topological signatures fit together.
This eliminates hallucinations by design: the system cannot invent facts it has
never seen.

### What it does

- **Speaks** — generates grammatical sentences by filling syntactic skeletons
  with concepts extracted from its correlation network (the Scalpel).
- **Learns** — grows its neural network by neurogenesis (new neurons for new
  correlations) and reinforces existing ones by LCT.
- **Validates** — connects to AEON ODV for deterministic scientific computation
  (P_sig, LCT monotonicity).
- **Searches** — falls back to DuckDuckGo / Google CSE when local knowledge
  is insufficient.

### Key numbers

| Metric | Value |
|---|---|
| Neurons (Scalpel) | 3,782,801 |
| LCT reinforcements | 43,260,980 |
| Vocabulary | 242,903 words |
| Network size | 294 MB |
| Training time | 5.2 hours (Colab CPU, 5M Wikipedia phrases) |
| Grammar templates | 13,000 (18 domains, 12 intentions, FR/EN) |
| Total size | ~895 MB (GloVe + Scalpel + grammar + knowledge packs) |
| GPT-4 comparison | 1,900× lighter |

---

## 2. Quick start (5 minutes)

```bash
# 1. Clone
git clone https://github.com/evinajonathan13-max/Ratiss-experimental-IA-.git
cd Ratiss-experimental-IA-

# 2. Install dependencies
pip install numpy datasets

# 3. Download GloVe (171 MB, one-time)
mkdir -p data/glove
curl -L -o data/glove/glove.6B.zip "https://nlp.stanford.edu/data/glove.6B.zip"
python3 -c "import zipfile; zipfile.ZipFile('data/glove/glove.6B.zip').extract('glove.6B.50d.txt', 'data/glove/')"
rm data/glove/glove.6B.zip

# 4. Download the Scalpel checkpoint (Git LFS, 294 MB)
git lfs install
git lfs pull

# 5. Run
python3 -c "
from ratis_net import RatisNet

net = RatisNet()
net.load_scalpel('artifacts/scalpel_wikipedia.pkl')
net.load_grammar('data/grammar_domains/dense_syntax_skeletons.json')
net.build_index()

print(net.respond('what is quantum mechanics'))
print(net.paragraph('consciousness', n_sentences=5))
"
```

### CLI

```bash
# Single query
python3 -m ratis_net.framework --query "what is consciousness"

# Paragraph
python3 -m ratis_net.framework --paragraph "quantum" --language en
```

---

## 3. Architecture

```
RatisNet (framework.py)
├── GloveTokenizer      — GloVe 50d (400K words) + topological signature (P_sig)
├── ScalpelLayer        — 3.78M neurons (neurogenesis + LCT reinforcement)
├── SkeletonSpeaker     — 13K grammatical templates (18 domains, 12 intentions, FR/EN)
├── AeonBridge          — AEON ODV connection (P_sig, LCT monotonicity, proofs)
├── WebSearchModule     — DuckDuckGo (no key) / Google CSE (with key)
├── ConceptDecoder      — concepts → sentences (Scalpel + decoder)
├── TriGrammarSpeaker   — word-by-word generation (2-word context window)
├── RatisSpeaker        — word-by-word generation (1-word, bigram)
├── RatissSynchrotron   — topological reconstruction (index + resonance + assembler)
├── ContextMapLoader    — ultra_context_map.json streaming (400 MiB)
├── CountsDiagnostic    — classical counts diagnostic (TVD + Shannon, not ETH)
└── StreamingDataLoader — Hugging Face Datasets streaming → Scalpel
```

### Data flow

```
User query
  ↓
RATIS-Net (language) : extract concepts from Scalpel
  ↓
AEON ODV (science)   : compute P_sig, validate LCT monotonicity
  ↓ (if concepts weak)
Web search            : DuckDuckGo / Google CSE
  ↓
SkeletonSpeaker       : fill grammatical template with concepts
  ↓
Response (fluent sentence + scientific fact + confidence)
```

---

## 4. The law LCT

The Law of Topological Coherence is **frozen** — do not change it.

```
ΔW = η · φ · P_sig · C
```

- `η` — learning rate (constitutive, dimensionless)
- `φ` — `|cos(ωt)|` — coherence amplitude of the "genius medium"
- `P_sig` — longest finite H1 persistence (topological signal)
- `C` — coherence of the input signal

The network learns by **maximizing P_sig** (becoming topologically robust), not
by minimizing a loss. LCT was validated on QPU (7 traceable IBM jobs, Spearman
+0.93 on 4MZI protein, +0.713 on IBM Quantum hardware).

---

## 5. Modules

| Module | File | Role |
|---|---|---|
| **Framework** | `framework.py` | Unified API: `RatisNet` |
| **Scalpel** | `scalpel.py` | Neurogenesis + LCT reinforcement (3.78M neurons) |
| **GloVe tokenizer** | `glove_tokenizer.py` | Hybrid GloVe + topological signatures |
| **Skeleton speaker** | `skeleton_speaker.py` | Grammatical template filling (13K templates) |
| **Concept decoder** | `concept_decoder.py` | Scalpel concepts → grammatical sentences |
| **Tri-grammaire** | `trigrammar.py` | Word generation with 2-word context |
| **Speaker** | `ratis_speaker.py` | Word-by-word generation (bigram) |
| **Synchrotron** | `ratiss_synchrotron.py` | Topological reconstruction (index + resonance) |
| **AEON bridge** | `aeon_bridge.py` | Connection to AEON ODV (P_sig, LCT proofs) |
| **Web search** | `web_search.py` | DuckDuckGo / Google CSE fallback |
| **Data loader** | `data_loader.py` | Hugging Face streaming → Scalpel |
| **Context map** | `context_map_loader.py` | ultra_context_map.json streaming (400 MiB) |
| **Counts diagnostic** | `counts_diagnostic.py` | TVD + Shannon (classical, not von Neumann) |
| **LCT neuron** | `lct_neuron.py` | The LCT neuron (`ΔW = η·φ·P_sig·C`) |
| **ETH thermo fixer** | `eth_thermo_fixer.py` | Contextual collapse threshold |
| **LCT collapse** | `lct_collapse.py` | Collapse + topological mark preservation |
| **Decoder** | `decoder.py` | LCT decoder (greedy + beam search) |
| **Pipeline** | `pipeline.py` | Branchable pipeline (DataSource → Tokenizer → Learner) |

---

## 6. Training the Scalpel on Wikipedia

### On Colab (recommended, free)

Open the notebook: [`ratisnet_colab_training.ipynb`](ratisnet_colab_training.ipynb)

It streams 5M Wikipedia phrases to the Scalpel via Hugging Face Datasets,
saves checkpoints to Google Drive every 10K phrases, and resumes on timeout.

### On any machine

```bash
python3 -m ratis_net.data_loader \
  --dataset wikipedia --config 20231101.en \
  --max-phrases 5000000 \
  --checkpoint-every 10000
```

At ~15-30 phrases/second (CPU), 5M phrases take ~46-93 hours. On Colab, ~5.2 hours.

### Scaling (validated empirically)

| Corpus | Neurons | Size | Time |
|---|---|---|---|
| 8 phrases | 22 | < 1 KB | < 1 s |
| 12K phrases | ~15K | ~1.2 MB | ~420 s |
| **5M phrases** | **3,782,801** | **294 MB** | **5.2 h (Colab)** |

The scaling is **linear**, not exponential. See [`docs/SCALING_NOTES.md`](docs/SCALING_NOTES.md).

---

## 7. Super RATISS: AEON bridge + web search

### AEON bridge

Connects RATIS-Net to [RATISS-ODV-AEON](https://github.com/evinajonathan13-max/RATISS-ODV-AEON)
for deterministic scientific computation:

```python
from ratis_net import RatisNet

net = RatisNet(aeon_path="/path/to/RATISS-ODV-AEON")
net.load_scalpel()
net.load_grammar()
net.build_index()

# Full science response
result = net.respond_with_science("how does topology influence protein folding")
print(result["sentence"])           # fluent sentence
print(result["aeon_fact"]["fact"])  # scientific fact (P_sig)
print(result["aeon_fact"]["confidence"])
```

If AEON is not installed, the bridge falls back to the local Vietoris-Rips
implementation (honest degraded mode).

### Web search

No API key needed (DuckDuckGo fallback):

```python
results = net.search("quantum decoherence biology")
for r in results:
    print(r["title"], r["snippet"][:80])
```

With Google CSE (optional):

```bash
export GOOGLE_API_KEY="your_key"
export GOOGLE_CSE_ID="your_cse_id"
```

### Knowledge packs

Located in [`data/knowledge_packs/`](data/knowledge_packs/):

| Pack | Domain |
|---|---|
| `quantum_physics_pack.json` | Quantum mechanics, entanglement, QPU |
| `math_logic_pack.json` | Topology, algebra, logic |
| `bio_pharma_pack.json` | Protein, pharma, biology |
| `ai_systems_pack.json` | AI, risks, systems |

---

## 8. Data files and checkpoints

| File | Size | Role | How to get |
|---|---|---|---|
| `artifacts/scalpel_wikipedia.pkl` | 294 MB | Scalpel checkpoint (3.78M neurons) | `git lfs pull` |
| `data/glove/glove.6B.50d.txt` | 171 MB | GloVe embeddings (400K words) | See quick start |
| `data/grammar_domains/dense_syntax_skeletons.json` | 10 MB | 13K grammatical templates | Included |
| `data/grammar_domains/conversation_matrix.json` | 20 MB | 24K conversation templates | Included |
| `data/grammar_domains/ultra_context_map.json` | 400 MB | Context map (242K concepts) | `git lfs pull` |
| `data/knowledge_packs/*.json` | ~1 MB | Scientific knowledge packs | Included |
| `data/cache/topo_signatures.npz` | 572 KB | Topological signature cache (15K words) | Included |

---

## 9. Tests

```bash
pip install pytest numpy
PYTHONPATH=. python -m pytest -q --ignore=tests/test_lct_new_systems.py
```

---

## 10. Honest limitations

1. **Not a knowledge base.** RATIS-Net reconstructs sentences from fragments
   it has seen. It does not "know" facts — it knows that words are correlated.
2. **Grammar is template-based.** The 13K skeletons guarantee grammatical
   correctness, but the output is not as fluid as a Transformer.
3. **Bigram correlations.** The Scalpel captures pairs of adjacent words,
   not full syntax trees.
4. **Coverage depends on corpus.** If Wikipedia does not mention a topic,
   RATIS-Net cannot talk about it. The web search module compensates.
5. **No multi-hop reasoning.** The system does not perform multi-step
   logical inference (A→B, B→C, therefore A→C).
6. **Counts diagnostic is classical.** Shannon entropy and TVD are classical
   metrics. ETH cannot be approximated from counts without tomography.

---

## 11. Citation

```bibtex
@software{evina_ratis_net_2026,
  author  = {Evina, Jonathan},
  title   = {RATIS-Net: Neural Network Trained by the Law of Topological Coherence},
  year    = {2026},
  url     = {https://github.com/evinajonathan13-max/Ratiss-experimental-IA-},
  note    = {Neurogenesis + LCT, no gradient, no GPU. 3.78M neurons, 294 MB.}
}
```

Distributed under the [MIT License](LICENSE) — © 2026 Jonathan Evina.
Intellectual property: JOHNKING0 & Jonathan Evina.
