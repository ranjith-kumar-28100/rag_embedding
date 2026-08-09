"""BM25 sparse encoder (spec §5.2) — fully offline, in-repo, no model download.

BM25 is arithmetic, not a neural model, so this needs nothing from the network.

Division of labour with Qdrant: this encoder emits only the *term-frequency* half
of BM25,

    w(t, d) = tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

and the collection declares ``Modifier.IDF`` on the sparse vector, so Qdrant
multiplies in the inverse-document-frequency term at query time using its own
live corpus statistics. That split is what makes encoding *streaming*: a batch
can be indexed without knowing the corpus, and IDF stays correct as the corpus
grows. Queries carry weight 1.0 per unique term — the IDF modifier does the
weighting.

``avgdl`` is the one statistic the streaming encoder cannot know, so it is a
config value (``bm25_avg_len``) rather than a measured mean. It only affects
length normalisation strength; documents far from it are normalised slightly
imprecisely, not incorrectly.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from .base import SparseVector

# Kept deliberately small: an aggressive stoplist throws away exactly the rare
# lexical signal BM25 exists to catch.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by for from has have he her his if in into is it
    its of on or she that the their them there these they this to was were will
    with you your
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_WHITESPACE_RE = re.compile(r"\s+")

# Conservative English suffix stripping. Not Porter — deliberately. It only has
# to be *consistent* between documents and queries; over-stemming would conflate
# distinct identifiers, which is the opposite of what the sparse branch is for.
_SUFFIXES: tuple[str, ...] = ("ations", "ation", "ingly", "edly", "ing", "ies", "ied", "es", "ed", "s")
_MIN_STEM_LEN = 4


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LEN:
            stem = token[: -len(suffix)]
            return f"{stem}y" if suffix in ("ies", "ied") else stem
    return token


def tokenize(text: str, *, tokenizer: str = "default") -> list[str]:
    """Lowercase → split → drop stopwords → stem. Documents and queries alike.

    ``whitespace`` skips stopword removal and stemming; use it for corpora where
    tokens are opaque identifiers and any normalisation is harmful.
    """
    lowered = text.lower()
    if tokenizer == "whitespace":
        return [t for t in _WHITESPACE_RE.split(lowered) if t]
    if tokenizer != "default":
        raise ValueError(f"unknown bm25_tokenizer {tokenizer!r}; expected 'default' or 'whitespace'")
    return [
        _stem(tok)
        for tok in _TOKEN_RE.findall(lowered)
        if len(tok) > 1 and tok not in STOPWORDS
    ]


def token_id(token: str) -> int:
    """Stable 32-bit id for a token.

    Qdrant sparse indices are uint32, and hashing (rather than a learned
    vocabulary) keeps the encoder stateless — new vocabulary appears mid-corpus
    without a rebuild. Collisions at 2^32 over a realistic vocabulary are rare
    enough to be noise in the ranking.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


class Bm25Encoder:
    """Encode text into Qdrant sparse vectors. Stateless and thread-safe."""

    def __init__(
        self,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
        tokenizer: str = "default",
    ) -> None:
        if avg_len <= 0:
            raise ValueError("bm25_avg_len must be > 0")
        self.k1 = k1
        self.b = b
        self.avg_len = avg_len
        self.tokenizer = tokenizer

    def embed_document(self, text: str) -> SparseVector:
        tokens = tokenize(text, tokenizer=self.tokenizer)
        if not tokens:
            return SparseVector(indices=[], values=[])
        doc_len = len(tokens)
        norm = self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_len)

        weights: dict[int, float] = {}
        for token, tf in Counter(tokens).items():
            weight = tf * (self.k1 + 1.0) / (tf + norm)
            tid = token_id(token)
            # On a hash collision keep the stronger term rather than summing —
            # summing would invent a term more important than either real one.
            weights[tid] = max(weights.get(tid, 0.0), weight)

        indices = sorted(weights)
        return SparseVector(indices=indices, values=[weights[i] for i in indices])

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.embed_document(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        """Query side: 1.0 per unique term; Qdrant's IDF modifier weights them."""
        tokens = tokenize(text, tokenizer=self.tokenizer)
        indices = sorted({token_id(t) for t in tokens})
        return SparseVector(indices=indices, values=[1.0] * len(indices))
