"""ratis_net.emocontext_loader — Charge EmoContext et mappe → ThermoEnvironment.

Piste 4 : on nourrit RATIS-Net avec de vrais dialogues humains annotés en
émotion (EmoContext, SemEval-2019 Task 3 : 30 160 dialogues 3-tours, 4 labels
happy/sad/angry/others).

Mapping émotion annotée → ThermoEnvironment (le contexte thermo du dialogue) :
  - happy  → ThermoEnvironment.joy()   (cœur modéré, détendu, chaud, excité positif)
  - angry  → ThermoEnvironment.anger() (cœur rapide, tendu, chaud, excité)
  - sad    → ThermoEnvironment.fear()   (cœur rapide, tendu, FROID, excité)
    (sad partage l'arousal/tension de la peur, mais froid = retrait)
  - others → ThermoEnvironment.calm()   (cœur lent, relaxé, neutre)

Chaque mot unique d'un dialogue devient un token ; son étiquette = l'émotion
annotée du dialogue. Le réseau apprend à associer (mot, contexte thermo) →
émotion. C'est la « thermodynamique du langage ».
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    from ratis_net.eth_thermo_fixer import ThermoEnvironment
except ImportError:
    # exécution en module direct depuis le dossier ratis_net/
    from eth_thermo_fixer import ThermoEnvironment

# mapping label EmoContext → (ThermoEnvironment, c_seuil_cible, label_num)
EMO_MAP = {
    "happy":  (ThermoEnvironment.joy,    0.7, 1),  # joie
    "angry":  (ThermoEnvironment.anger, 0.3, 0),  # colère
    "sad":    (ThermoEnvironment.fear,  0.2, 0),  # tristesse (froid = retrait)
    "others": (ThermoEnvironment.calm, 0.5, 2),  # neutre
}

_TOKEN_RE = re.compile(r"[a-zà-ÿ']+")


def tokenize(text: str) -> list[str]:
    """Tokenise un tour de dialogue : mots en minuscules, ponctuation/emojis hors."""
    return _TOKEN_RE.findall(text.lower())


def load_emocontext(path: str | Path, max_examples: int | None = None) -> list[dict]:
    """Charge EmoContext (train.txt ou dev.txt).

    Format TSV 5 colonnes : id, turn1, turn2, turn3, label.
    Retourne une liste de {turn1, turn2, turn3, label, env, c_seuil, label_num}.
    """
    path = Path(path)
    examples = []
    with open(path, encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, t1, t2, t3, label = parts
            label = label.strip().lower()
            if label not in EMO_MAP:
                continue
            env_cls, c_seuil, label_num = EMO_MAP[label]
            examples.append({
                "turn1": t1, "turn2": t2, "turn3": t3,
                "label": label, "env": env_cls(),
                "c_seuil": c_seuil, "label_num": label_num,
            })
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def build_samples(examples: list[dict], embedding_fn, dim: int = 8,
                  per_word: bool = True) -> list[tuple]:
    """Construit les samples d'entraînement.

    Chaque sample = (token_embedding, env, label_num, c_seuil).
    Si per_word : un sample par mot unique (tous les tours concaténés) par
    dialogue. Sinon : un sample par dialogue (token = tour3, l'émotion cible).

    embedding_fn(word, dim) → np.ndarray (hash, TTF, ou topo signature).
    """
    samples = []
    for ex in examples:
        if per_word:
            words = tokenize(ex["turn1"] + " " + ex["turn2"] + " " + ex["turn3"])
            seen = set()
            for w in words:
                if w in seen or len(w) < 2:
                    continue
                seen.add(w)
                emb = embedding_fn(w, dim)
                samples.append((emb, ex["env"], ex["label_num"], ex["c_seuil"]))
        else:
            emb = embedding_fn(ex["turn3"], dim)
            samples.append((emb, ex["env"], ex["label_num"], ex["c_seuil"]))
    return samples


def vocabulary(examples: list[dict], min_len: int = 2, top_k: int | None = None) -> list[str]:
    """Extrait le vocabulaire (mots uniques) des exemples, trié par fréquence."""
    from collections import Counter
    c = Counter()
    for ex in examples:
        for w in tokenize(ex["turn1"] + " " + ex["turn2"] + " " + ex["turn3"]):
            if len(w) >= min_len:
                c[w] += 1
    if top_k:
        return [w for w, _ in c.most_common(top_k)]
    return [w for w, _ in c.most_common()]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[1]
    train = load_emocontext(_ROOT / "data" / "emocontext" / "train.txt", max_examples=1000)
    print(f"Chargé {len(train)} exemples (sur 1000 max)")
    from collections import Counter
    print("Labels:", dict(Counter(e["label"] for e in train)))
    vocab = vocabulary(train, top_k=20)
    print(f"Top-20 mots: {vocab}")
    print(f"\nExemple : {train[0]}")
    print(f"Tokens turn3 : {tokenize(train[0]['turn3'])}")
