"""BM25 encoder: tokenization, document weighting, and query weighting."""

from __future__ import annotations

import pytest

from rag_embedding.config import EmbeddingConfig
from rag_embedding.embedders.factory import build_sparse_embedder
from rag_embedding.embedders.sparse_bm25 import Bm25Encoder, token_id, tokenize


def test_tokenizer_lowercases_strips_punctuation_and_stopwords() -> None:
    assert tokenize("The Retrieval, of DATA!") == ["retrieval", "data"]


def test_tokenizer_keeps_identifiers_and_digits() -> None:
    tokens = tokenize("id=1001 amount=250.00 status_code=cleared")
    assert "1001" in tokens
    assert "250" in tokens
    assert "status_cod" in tokens or "status_code" in tokens


def test_inflections_collapse_to_one_stem() -> None:
    assert tokenize("indexing indexed indexes index") == ["index"] * 4


def test_stemming_is_deliberately_conservative() -> None:
    """Only the listed suffixes are stripped.

    "retrieval" keeps its "-al": over-stemming would conflate distinct
    identifiers, which is the opposite of what the lexical branch is for. What
    matters is that documents and queries are normalised *identically*.
    """
    assert tokenize("retrieving retrieved") == ["retriev", "retriev"]
    assert tokenize("retrieval") == ["retrieval"]


def test_short_stems_are_left_alone() -> None:
    # Stripping "s" from "gas" would leave a 2-char stem, below _MIN_STEM_LEN.
    assert tokenize("gas") == ["gas"]


def test_whitespace_tokenizer_does_no_normalisation() -> None:
    assert tokenize("The Retrieval, of DATA!", tokenizer="whitespace") == [
        "the",
        "retrieval,",
        "of",
        "data!",
    ]


def test_unknown_tokenizer_rejected() -> None:
    with pytest.raises(ValueError, match="bm25_tokenizer"):
        tokenize("x", tokenizer="porter")


def test_token_id_is_stable_and_uint32() -> None:
    assert token_id("retrieval") == token_id("retrieval")
    assert 0 <= token_id("retrieval") < 2**32


def test_document_vector_is_sorted_and_parallel(bm25: Bm25Encoder) -> None:
    vector = bm25.embed_document("dense sparse fusion dense")
    assert vector.indices == sorted(vector.indices)
    assert len(vector.indices) == len(vector.values)
    assert len(vector.indices) == 3  # dense, sparse, fusion (dense deduped)
    assert all(v > 0 for v in vector.values)


def test_term_frequency_saturates(bm25: Bm25Encoder) -> None:
    """BM25's k1 saturation: the 10th occurrence adds far less than the 2nd."""
    once = bm25.embed_document("retrieval").as_dict()[token_id("retrieval")]
    twice = bm25.embed_document("retrieval retrieval").as_dict()[token_id("retrieval")]
    tenfold = bm25.embed_document(" ".join(["retrieval"] * 10)).as_dict()[token_id("retrieval")]
    assert once < twice < tenfold
    assert (tenfold - twice) < (twice - once) * 8


def test_longer_documents_are_penalised() -> None:
    """Same term count, more filler -> lower weight (length normalisation)."""
    encoder = Bm25Encoder(avg_len=10.0)
    short = encoder.embed_document("retrieval").as_dict()[token_id("retrieval")]
    padded = "retrieval " + " ".join(f"filler{i}" for i in range(40))
    long = encoder.embed_document(padded).as_dict()[token_id("retrieval")]
    assert long < short


def test_b_zero_disables_length_normalisation() -> None:
    encoder = Bm25Encoder(b=0.0, avg_len=10.0)
    short = encoder.embed_document("retrieval").as_dict()[token_id("retrieval")]
    padded = "retrieval " + " ".join(f"filler{i}" for i in range(40))
    long = encoder.embed_document(padded).as_dict()[token_id("retrieval")]
    assert short == pytest.approx(long)


def test_query_weights_are_uniform(bm25: Bm25Encoder) -> None:
    """IDF is Qdrant's job (Modifier.IDF); the query just names its terms."""
    vector = bm25.embed_query("dense sparse dense fusion")
    assert vector.values == [1.0, 1.0, 1.0]
    assert vector.indices == sorted(vector.indices)


def test_empty_and_stopword_only_text_yields_empty_vector(bm25: Bm25Encoder) -> None:
    assert bm25.embed_document("").indices == []
    assert bm25.embed_document("the and of it").indices == []


def test_query_and_document_share_token_ids(bm25: Bm25Encoder) -> None:
    doc = bm25.embed_document("reciprocal rank fusion of retrieved contexts")
    query = bm25.embed_query("retrieval fusion")
    assert set(query.indices) & set(doc.indices)


def test_factory_honours_sparse_enabled() -> None:
    assert build_sparse_embedder(EmbeddingConfig(sparse_enabled=False)) is None
    assert isinstance(build_sparse_embedder(EmbeddingConfig()), Bm25Encoder)


def test_factory_passes_config_through() -> None:
    encoder = build_sparse_embedder(EmbeddingConfig(bm25_k1=2.0, bm25_b=0.1, bm25_avg_len=64))
    assert isinstance(encoder, Bm25Encoder)
    assert (encoder.k1, encoder.b, encoder.avg_len) == (2.0, 0.1, 64.0)


def test_avg_len_must_be_positive() -> None:
    with pytest.raises(ValueError, match="avg_len"):
        Bm25Encoder(avg_len=0.0)
