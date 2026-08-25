"""ratis_net.framework — Framework IA unifié RATIS-Net.

Unifie tous les modules en une seule API cohérente :

  RatisNet (framework)
    ├── Scalpel          (réseau de corrélations LCT, 3.78M neurones)
    ├── Synchrotron      (reconstruction topologique)
    ├── SkeletonSpeaker  (génération par squelettes grammaticaux)
    ├── ConceptDecoder   (concepts → phrases)
    ├── TriGrammarSpeaker(génération mot par mot, fenêtre 2)
    └── GloveTokenizer   (embeddings hybrides)

Usage :
    from ratis_net.framework import RatisNet

    net = RatisNet()
    net.load_scalpel("artifacts/scalpel_wikipedia.pkl")
    net.load_grammar("data/grammar_domains/dense_syntax_skeletons.json")

    print(net.respond("what is quantum mechanics"))
    print(net.respond("explain consciousness", language="fr"))
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ratis_net.glove_tokenizer import GloveTokenizer
from ratis_net.scalpel import ScalpelLayer
from ratis_net.skeleton_speaker import SkeletonSpeaker


class RatisNet:
    """Framework IA unifié : Scalpel + Synchrotron + Squelettes.

    Le framework orchestre les 3 couches :
      1. Scalpel (concepts) — réseau de corrélations LCT
      2. Squelettes (grammaire) — 13K+ formulations bilingues
      3. GloVe (sémantique) — embeddings de 400K mots

    API simple :
      net.respond("what is quantum") → phrase fluide
      net.concepts("quantum") → liste de concepts corrélés
      net.paragraph("consciousness", n=5) → paragraphe long
    """

    def __init__(self, dim: int = 12, n_glove: int = 8, seed: int = 42):
        self.tokenizer = GloveTokenizer(dim=dim, n_glove=n_glove)
        self.scalpel = ScalpelLayer(self.tokenizer, eta=0.1,
                                     coherence_threshold=0.3, seed=seed)
        self.speaker = SkeletonSpeaker(self.scalpel, seed=seed)
        self._loaded = False
        self._index_built = False

    def load_scalpel(self, path: str | Path = "artifacts/scalpel_wikipedia.pkl",
                     verbose: bool = True) -> None:
        """Charge le checkpoint Scalpel (294 MB, 3.78M neurones)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scalpel checkpoint not found: {path}")
        self.scalpel.load(path)
        if verbose:
            print(f"Scalpel: {self.scalpel.network_size():,} neurons, "
                  f"{self.scalpel.total_reinforcements:,} reinforcements")
        self._loaded = True

    def load_grammar(self, dense_path: str | Path | None = None,
                     conversation_path: str | Path | None = None,
                     verbose: bool = True) -> None:
        """Charge les gabarits grammaticaux (13K + 24K formulations)."""
        # Le SkeletonSpeaker charge automatiquement les fichiers s'ils existent
        # dans data/grammar_domains/. Cette méthode force le rechargement.
        if dense_path:
            import json
            with open(dense_path, encoding="utf-8") as f:
                self.speaker.dense_grammar = json.load(f)
        if conversation_path:
            import json
            with open(conversation_path, encoding="utf-8") as f:
                self.speaker.conversation_matrix = json.load(f)
        if verbose:
            n_dense = 0
            if self.speaker.dense_grammar:
                domains = self.speaker.dense_grammar.get("domains", {})
                n_dense = sum(len(entries)
                              for d in domains.values()
                              for entries in d.values())
            n_conv = 0
            if self.speaker.conversation_matrix:
                contexts = self.speaker.conversation_matrix.get("contexts", {})
                n_conv = sum(len(entries)
                             for c in contexts.values()
                             for entries in c.values() if isinstance(entries, list))
            print(f"Grammar: {n_dense:,} dense + {n_conv:,} conversation templates")

    def build_index(self, verbose: bool = True) -> None:
        """Construit l'index inversé pour les lookups O(1)."""
        if not self._loaded:
            raise RuntimeError("Load Scalpel first with load_scalpel()")
        self.speaker.build_index(verbose=verbose)
        self._index_built = True

    def concepts(self, word: str, n: int = 10) -> list[str]:
        """Extrait les n concepts les plus corrélés à un mot."""
        if not self._index_built:
            self.build_index(verbose=False)
        corrs = self.speaker._corrs(word)
        concepts = []
        for term, weight, _ in corrs:
            if term not in self.speaker.STOPWORDS and term != word and len(term) > 1:
                concepts.append(term)
                if len(concepts) >= n:
                    break
        return concepts

    def respond(self, query: str, language: str = "en") -> str:
        """Génère une réponse fluide à une requête.

        Pipeline :
          1. Extrait le mot-clé de la requête.
          2. Le Scalpel fournit les concepts corrélés.
          3. Un squelette grammatical est choisi et rempli.
        """
        if not self._index_built:
            self.build_index(verbose=False)
        result = self.speaker.generate_response(query, language=language)
        return result["sentence"]

    def respond_full(self, query: str, language: str = "en") -> dict[str, Any]:
        """Génère une réponse détaillée (avec concepts, squelette, paragraphe)."""
        if not self._index_built:
            self.build_index(verbose=False)
        return self.speaker.generate_response(query, language=language)

    def paragraph(self, theme: str, n_sentences: int = 5,
                  language: str = "en") -> str:
        """Génère un paragraphe long sur un thème."""
        if not self._index_built:
            self.build_index(verbose=False)
        return self.speaker.generate_paragraph(theme, n_sentences=n_sentences,
                                                language=language)

    def converse(self, query: str, language: str = "en") -> str:
        """Mode conversation : réponse + paragraphe court (3 phrases)."""
        if not self._index_built:
            self.build_index(verbose=False)
        result = self.speaker.generate_response(query, language=language)
        return result["paragraph"]

    def stats(self) -> dict[str, Any]:
        """Statistiques du framework."""
        return {
            "scalpel_neurons": self.scalpel.network_size(),
            "scalpel_reinforcements": self.scalpel.total_reinforcements,
            "scalpel_neurogenesis": self.scalpel.total_neurogenesis,
            "vocab_size": len(self.speaker._index) if self._index_built else 0,
            "grammar_loaded": self.speaker.dense_grammar is not None,
            "conversation_loaded": self.speaker.conversation_matrix is not None,
            "glove_available": self.tokenizer.backend() != "topo_fallback",
        }

    def __repr__(self) -> str:
        n = self.scalpel.network_size()
        return f"RatisNet(neurons={n:,}, loaded={self._loaded})"


# ─────────────────────────────────────────────────────────────────────────────
# CLI : python -m ratis_net.framework
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="RATIS-Net unified framework")
    ap.add_argument("--scalpel", default="artifacts/scalpel_wikipedia.pkl")
    ap.add_argument("--grammar", default="data/grammar_domains/dense_syntax_skeletons.json")
    ap.add_argument("--query", default=None, help="Single query to respond to")
    ap.add_argument("--paragraph", default=None, help="Theme for paragraph generation")
    ap.add_argument("--language", default="en", choices=["en", "fr"])
    args = ap.parse_args()

    print("=== RATIS-Net Unified Framework ===\n")
    net = RatisNet()
    net.load_scalpel(args.scalpel)
    net.load_grammar(args.grammar)
    net.build_index()
    print()

    stats = net.stats()
    print(f"Stats: {stats}\n")

    if args.query:
        print(f"Q: {args.query}")
        print(f"R: {net.respond(args.query, language=args.language)}\n")
    elif args.paragraph:
        print(f"Theme: {args.paragraph}")
        print(f"{net.paragraph(args.paragraph, n_sentences=5, language=args.language)}\n")
    else:
        # Demo complet
        queries_en = [
            "what is quantum mechanics",
            "explain consciousness",
            "what is love",
            "how does the brain work",
            "what is science",
        ]
        print("=== Responses (EN) ===\n")
        for q in queries_en:
            print(f"Q: {q}")
            print(f"R: {net.respond(q, language='en')}")
            print()

        print("=== Paragraphs ===\n")
        for theme in ["quantum", "consciousness", "love"]:
            print(f"[{theme}]")
            print(net.paragraph(theme, n_sentences=4, language="en"))
            print()

        print("=== Français ===\n")
        for q in ["qu'est-ce que la science", "explique l'amour"]:
            print(f"Q: {q}")
            print(f"R: {net.respond(q, language='fr')}")
            print()
