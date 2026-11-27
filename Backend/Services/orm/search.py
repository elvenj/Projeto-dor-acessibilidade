"""Busca híbrida com deduplicação, evidência lexical e fontes."""
import math
import os
import re
import unicodedata

from Database.chromaDb.chroma import collection
from Models.embedding_model import embedding_model

_STOPWORDS = set("a o as os de da do das dos e em na no nas nos para por um uma quais qual que preciso precisa realizar como meu minha eu se ao os sao ser com fazer faco posso necessarios necessario necessarias necessaria quero gostaria saber".split())



_NUMBER_WORDS = {
    "um": "1", "dois": "2", "duas": "2",
    "tres": "3", "quatro": "4", "cinco": "5",
    "seis": "6", "sete": "7", "oito": "8",
    "nove": "9", "dez": "10"
}

_STOPWORDS.update({
    "sao", "sobre", "informacao", "informacoes",
    "favor", "funciona", "funcionamento", "funcionar"
})


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", normalized)
    # Reúne flexões comuns sem depender de outro modelo.
    terms = set()

    for word in words:

        if word in _STOPWORDS:
            continue

        word = _NUMBER_WORDS.get(word, word)

        if word.isdigit():
            terms.add(word)

        elif len(word) > 2:
            terms.add(
                word[:-1] if word.endswith("s") else word
            )

    return terms


def _proximity(text: str, query_terms: set[str]) -> float:
    """Favorece termos juntos (ex.: documentos para matrícula), não menções dispersas."""
    tokens = [term for word in text.split() for term in sorted(_terms(word))]
    best = float("inf")
    counts = {}
    left = 0
    for right, term in enumerate(tokens):
        if term in query_terms:
            counts[term] = counts.get(term, 0) + 1
        while query_terms and len(counts) == len(query_terms):
            best = min(best, right - left + 1)
            previous = tokens[left]
            if previous in counts:
                counts[previous] -= 1
                if not counts[previous]:
                    del counts[previous]
            left += 1
    return 1 / (1 + best - len(query_terms)) if math.isfinite(best) else 0.0


def search_evidence(query: str, top_k: int = 5) -> list[dict]:
    if not query.strip() or top_k <= 0:
        return []
    count = collection.count()
    if not count:
        return []
    all_terms = _terms(query)

    requested_years = {
        term for term in all_terms
        if re.fullmatch(r"(?:19|20)\d{2}", term)
    }

    qualifiers = {
        term for term in all_terms
        if term.isdigit()
    }

    if {"puc", "campina"} <= all_terms:
        qualifiers.update({"puc", "campina"})

    query_terms = all_terms - qualifiers

    if not query_terms:
        query_terms = all_terms
        qualifiers = set()

    min_coverage = min(
        1.0,
        max(0.6, 2 / max(1, len(query_terms)))
    )
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[query_embedding], n_results=min(count, max(30, top_k * 6)),
        include=["documents", "metadatas", "distances"],
    )
    candidates = {}
    for identifier, text, metadata, distance in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0],
    ):
        candidates[identifier] = (text, metadata or {}, float(distance))

    # A busca textual também alcança listas exatas fora dos vizinhos semânticos.
    # Paginação evita carregar todos os documentos em uma única resposta do Chroma.
    lexical = []
    for offset in range(0, count, 500):
        batch = collection.get(limit=500, offset=offset, include=["documents", "metadatas"])
        for identifier, text, metadata in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            coverage = len(query_terms & _terms(text or "")) / max(1, len(query_terms))
            if query_terms and coverage >= min_coverage:
                lexical.append((identifier, text, metadata or {}))
    for identifier, text, metadata in lexical:
        if identifier not in candidates:
            candidates[identifier] = (text, metadata, None)

    max_distance = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
    ranked = []
    for text, metadata, distance in candidates.values():
        if not text:
            continue

        # Evitar misturar edições distintas do vestibular
        if (
            "vestibular" in all_terms
            and len(requested_years) == 1
            and not requested_years.issubset(_terms(text))
        ):
            continue
        coverage = len(query_terms & _terms(text)) / max(1, len(query_terms))
        # Exige evidência dos termos centrais, além da proximidade semântica.
        if query_terms and coverage < min_coverage:
            continue
        if distance is not None and (not math.isfinite(distance) or distance > max_distance) and coverage < 1:
            continue
        similarity = max(0.0, 1 - distance) if distance is not None else 0.0
        matched_terms = query_terms & _terms(text)

        proximity = (
            _proximity(text, matched_terms)
            if len(matched_terms) >= 2
            else 0.0
        )

        qualifier_score = (
            len(qualifiers & _terms(text))
            / max(1, len(qualifiers))
        )

        score = (
            0.55 * coverage
            + 0.25 * proximity * coverage
            + 0.10 * similarity
            + 0.10 * qualifier_score
        )

        # Priorizar títulos relacionados em consultas amplas
        if len(query_terms) == 1:
            source_terms = _terms(metadata.get("source") or "")
            if query_terms & source_terms:
                score += 0.30

        ranked.append((score, {
            "text": text, "source": metadata.get("source", "Fonte não identificada (índice antigo)"),
            "page": metadata.get("page"), "distance": distance,
        }))
    ranked.sort(key=lambda item: item[0], reverse=True)
    seen = set()
    evidence = []
    for _, item in ranked:
        fingerprint = " ".join(item["text"].casefold().split())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        evidence.append(item)
        if len(evidence) == top_k:
            break
    return evidence


def search_knowledge_base(query: str, top_k: int = 5):
    # Mantém o contrato do endpoint /searchData: uma lista de textos.
    return [item["text"] for item in search_evidence(query, top_k)]
