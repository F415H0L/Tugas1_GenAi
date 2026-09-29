# 8.5 Chunking Text for Embedding (Sentence-Aware)
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int

def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """Split text on sentence boundaries, preserving source offsets and overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars.")
    if not text.strip():
        return []

    sentence_spans: list[tuple[int, int]] = []
    sentence_start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+", text):
        raw_sentence = text[sentence_start:boundary.start()]
        leading = len(raw_sentence) - len(raw_sentence.lstrip())
        trailing = len(raw_sentence.rstrip())
        if trailing > leading:
            sentence_spans.append((sentence_start + leading, sentence_start + trailing))
        sentence_start = boundary.end()

    tail = text[sentence_start:]
    leading = len(tail) - len(tail.lstrip())
    trailing = len(tail.rstrip())
    if trailing > leading:
        sentence_spans.append((sentence_start + leading, sentence_start + trailing))

    chunks: list[Chunk] = []
    current_spans: list[tuple[int, int]] = []

    def save_chunk(spans: list[tuple[int, int]]) -> None:
        start, end = spans[0][0], spans[-1][1]
        chunks.append(Chunk(doc_id, len(chunks), text[start:end], start, end))

    for sentence_span in sentence_spans:
        if not current_spans:
            current_spans.append(sentence_span)
            continue

        if sentence_span[1] - current_spans[0][0] <= max_chars:
            current_spans.append(sentence_span)
            continue

        save_chunk(current_spans)

        overlap_spans = []
        previous_end = current_spans[-1][1]
        for previous_span in reversed(current_spans):
            overlap_size = previous_end - previous_span[0]
            if overlap_size > overlap_chars:
                break
            if sentence_span[1] - previous_span[0] > max_chars:
                break
            overlap_spans.insert(0, previous_span)

        current_spans = overlap_spans + [sentence_span]

    if current_spans:
        save_chunk(current_spans)

    return chunks


if __name__ == "__main__":
    document = """
Large language models (LLMs) are neural networks trained on vast amounts of text data.
They learn to predict the next token in a sequence, which gives them broad language understanding.
Models like GPT-4 and Claude are examples of LLMs used in production today.
Retrieval-Augmented Generation, or RAG, extends LLMs by connecting them to external knowledge bases.
Instead of relying solely on knowledge encoded during training, a RAG system retrieves relevant documents at inference time.
This allows the model to answer questions about recent events or private data it was never trained on.
The retrieval step in RAG typically uses embedding-based semantic search.
A query is embedded into a vector, and the nearest document vectors are retrieved from a database.
These documents are then injected into the LLM's context window alongside the query.
"""

    chunks = chunk_by_sentences(
        document,
        doc_id="intro_to_llms",
        max_chars=300,
        overlap_chars=50,
    )
    for chunk in chunks:
        print(f"Chunk {chunk.chunk_index} ({chunk.char_start}-{chunk.char_end}): {chunk.text[:80]}...")
        print()
