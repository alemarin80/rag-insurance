"""Unit tests for src.retrieval.bm25 and the bm25 search mode. No API, no database."""

from unittest.mock import MagicMock

import pytest

from src.retrieval import bm25, search as search_module

ROWS = [
    (1, "a.pdf", 1, "Egenandel ved skade er 4 000 kroner."),
    (2, "a.pdf", 2, "Personlige eiendeler i bobilen dekkes med inntil 40 000 kroner."),
    (3, "b.pdf", 1, "Tyveri av sykkel fra hagen."),
]


@pytest.fixture
def index():
    return bm25.build_index(ROWS)


def test_another_word_form_finds_the_chunk(index):
    found = bm25.bm25_search("Hva er egenandelen?", index=index)
    assert found[0][0] == 1


def test_no_matching_word_returns_nothing(index):
    assert bm25.bm25_search("kryptovaluta svindel", index=index) == []
    assert bm25.bm25_search("og i at", index=index) == []  # only stop words


def test_only_chunks_that_score_are_returned(index):
    found = bm25.bm25_search("bobilen", top_k=5, index=index)
    assert [r[0] for r in found] == [2]
    assert found[0][4] > 0


def test_search_bm25_makes_no_embedding_call(monkeypatch, index):
    embed = MagicMock()
    monkeypatch.setattr(search_module, "embed_query", embed)
    monkeypatch.setattr(search_module, "bm25_search", lambda q, k: bm25.bm25_search(q, k, index=index))
    found = search_module.search("sykkel stjålet", mode="bm25")
    assert [c.id for c in found] == [3]
    embed.assert_not_called()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        search_module.search("spm", mode="keyword")
