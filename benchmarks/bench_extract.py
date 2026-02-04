from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from parsantic.extract.alignment import AlignmentOptions, align_value_to_text
from parsantic.extract.chunking import iter_chunks
from parsantic.extract.tokenizer import (
    RegexTokenizer,
    Tokenizer,
    UnicodeTokenizer,
    get_tokenizer,
)


class BenchResult(NamedTuple):
    name: str
    iters: int
    seconds_per_iter: float


def _run_bench(
    name: str,
    fn: Callable[[], object],
    *,
    target_total_seconds: float = 0.25,
    repeats: int = 7,
    max_iters: int = 1_000_000,
) -> BenchResult:
    iters = 1
    while True:
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= target_total_seconds:
            break
        if iters >= max_iters:
            break
        iters *= 2

    per_iter_samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed = time.perf_counter() - start
        per_iter_samples.append(elapsed / iters)

    return BenchResult(name=name, iters=iters, seconds_per_iter=statistics.median(per_iter_samples))


def _fmt_seconds(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.1f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.3f} ms"
    return f"{s:.3f} s"


def _print_table(results: list[BenchResult]) -> None:
    name_w = max(len(r.name) for r in results)
    it_w = max(len(str(r.iters)) for r in results)
    print(
        f"{'scenario'.ljust(name_w)}  {'iters'.rjust(it_w)}  {'median/op'.rjust(12)}  {'ops/s'.rjust(12)}"
    )
    print(f"{'-' * name_w}  {'-' * it_w}  {'-' * 12}  {'-' * 12}")
    for r in results:
        ops = 1.0 / r.seconds_per_iter if r.seconds_per_iter else float("inf")
        print(
            f"{r.name.ljust(name_w)}  {str(r.iters).rjust(it_w)}  {str(_fmt_seconds(r.seconds_per_iter)).rjust(12)}  {ops:12.0f}"
        )


@dataclass(frozen=True, slots=True)
class TextSample:
    lang: str
    base: str
    target_sentence: str

    def make_text(self, n_chars: int) -> str:
        marker = f"\n\n{self.target_sentence}.\n"
        if n_chars <= len(marker):
            return marker[:n_chars]
        body_len = n_chars - len(marker)
        reps = body_len // len(self.base) + 1
        body = (self.base * reps)[:body_len]
        return body + marker

    def token_match_value(self) -> str:
        return self.target_sentence.lower()

    def exact_substring_value(self) -> str:
        return self.target_sentence


def _iter_all_chunks(text: str, *, max_char_buffer: int | None, tokenizer: Tokenizer) -> int:
    return sum(
        1
        for _ in iter_chunks(
            text,
            max_char_buffer=max_char_buffer,
            tokenizer=tokenizer,
            overlap_chars=0,
        )
    )


def _print_tokenization_throughput(results: list[tuple[BenchResult, int, int]]) -> None:
    if not results:
        return
    name_w = max(len(r.name) for r, _, _ in results)
    it_w = max(len(str(r.iters)) for r, _, _ in results)
    chars_w = max(len(str(chars)) for _, chars, _ in results)
    toks_w = max(len(str(toks)) for _, _, toks in results)
    print("\nTokenization Throughput")
    print(
        f"{'scenario'.ljust(name_w)}  {'iters'.rjust(it_w)}  {'chars'.rjust(chars_w)}  {'tokens'.rjust(toks_w)}  {'chars/s'.rjust(12)}  {'tokens/s'.rjust(12)}"
    )
    print(f"{'-' * name_w}  {'-' * it_w}  {'-' * chars_w}  {'-' * toks_w}  {'-' * 12}  {'-' * 12}")
    for bench, chars, toks in results:
        chars_per_sec = chars / bench.seconds_per_iter if bench.seconds_per_iter else float("inf")
        toks_per_sec = toks / bench.seconds_per_iter if bench.seconds_per_iter else float("inf")
        print(
            f"{bench.name.ljust(name_w)}  {str(bench.iters).rjust(it_w)}  {str(chars).rjust(chars_w)}  {str(toks).rjust(toks_w)}  {chars_per_sec:12.0f}  {toks_per_sec:12.0f}"
        )


def _safe_get_pipeline_tokenizer(tokenizer_name: str) -> Tokenizer:
    tok = get_tokenizer(tokenizer_name)
    try:
        tok.tokenize("smoke test")
    except ImportError:
        print(f"Warning: tokenizer '{tokenizer_name}' unavailable; falling back to RegexTokenizer.")
        tok = RegexTokenizer()
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmarks for parsantic extraction pipeline."
    )
    parser.add_argument("--target-seconds", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--tokenizer",
        choices=["regex", "unicode"],
        default="regex",
        help="Tokenizer used for alignment+chunking benchmarks (default: regex).",
    )
    args = parser.parse_args()

    samples = {
        "en": TextSample(
            lang="en",
            base=(
                "In a quiet laboratory, researchers test new ideas. "
                "They write notes, revise hypotheses, and compare results.\n"
                "Precision matters: units, dates, and names must stay consistent. "
                "When errors appear, they trace them back and try again.\n"
                "This paragraph includes punctuation—commas, dashes, and quotes—so tokenizers see variety.\n"
            ),
            target_sentence="Ada Lovelace wrote the first algorithm",
        ),
        "de": TextSample(
            lang="de",
            base=(
                "In einem ruhigen Labor prüfen Forschende neue Ideen. "
                "Sie schreiben Notizen, überarbeiten Hypothesen und vergleichen Ergebnisse.\n"
                "Genauigkeit zählt: Maßeinheiten, Daten und Namen müssen konsistent bleiben. "
                "Wenn Fehler auftreten, suchen sie die Ursache und versuchen es erneut.\n"
                "Dieser Absatz enthält Umlaute wie Ä, Ö, Ü und Wörter wie Straße sowie Satzzeichen.\n"
            ),
            target_sentence="Die Straße ist nass",
        ),
        "fr": TextSample(
            lang="fr",
            base=(
                "Dans un laboratoire calme, les chercheurs testent de nouvelles idées. "
                "Ils prennent des notes, révisent des hypothèses et comparent les résultats.\n"
                "La précision compte : unités, dates et noms doivent rester cohérents. "
                "Quand une erreur apparaît, on remonte à la cause et on recommence.\n"
                "Ce paragraphe contient des accents (é, è, ê, ç) et des guillemets « comme ceci ».\n"
            ),
            target_sentence="L'été est déjà là",
        ),
    }

    pipeline_tok = _safe_get_pipeline_tokenizer(args.tokenizer)

    align_opts = AlignmentOptions(enable_fuzzy_alignment=False, accept_match_lesser=False)

    text_sizes = [100, 1_000, 10_000]
    texts: dict[tuple[str, int], str] = {}
    for lang, sample in samples.items():
        for size in text_sizes:
            texts[(lang, size)] = sample.make_text(size)

    results: list[BenchResult] = []

    # Alignment benchmarks (token match path; sizes across EN; 10k across DE/FR)
    for size in text_sizes:
        text = texts[("en", size)]
        value = samples["en"].token_match_value()
        results.append(
            _run_bench(
                f"align token-match (en, {size:,} chars)",
                lambda text=text, value=value: align_value_to_text(
                    text,
                    "/field",
                    value,
                    tokenizer=pipeline_tok,
                    options=align_opts,
                ),
                target_total_seconds=args.target_seconds,
                repeats=args.repeats,
            )
        )
    for lang in ("de", "fr"):
        text = texts[(lang, 10_000)]
        value = samples[lang].token_match_value()
        results.append(
            _run_bench(
                f"align token-match ({lang}, 10,000 chars)",
                lambda text=text, value=value: align_value_to_text(
                    text,
                    "/field",
                    value,
                    tokenizer=pipeline_tok,
                    options=align_opts,
                ),
                target_total_seconds=args.target_seconds,
                repeats=args.repeats,
            )
        )
    results.append(
        _run_bench(
            "align exact-substring (en, 10,000 chars)",
            lambda: align_value_to_text(
                texts[("en", 10_000)],
                "/field",
                samples["en"].exact_substring_value(),
                tokenizer=pipeline_tok,
                options=align_opts,
            ),
            target_total_seconds=args.target_seconds,
            repeats=args.repeats,
        )
    )

    # Chunking benchmarks (vary max_char_buffer over a fixed text)
    chunk_text = texts[("en", 10_000)]
    chunk_sizes: list[int | None] = [None, 256, 512, 1_024, 2_048]
    for size in chunk_sizes:
        label = "None" if size is None else f"{size:,}"
        results.append(
            _run_bench(
                f"chunk iter_chunks (en, 10,000 chars, max={label})",
                lambda size=size: _iter_all_chunks(
                    chunk_text, max_char_buffer=size, tokenizer=pipeline_tok
                ),
                target_total_seconds=args.target_seconds,
                repeats=args.repeats,
            )
        )

    # Tokenization benchmarks (throughput; regex vs unicode)
    tok_regex = RegexTokenizer()
    tok_unicode = UnicodeTokenizer()
    tokenization_results: list[tuple[BenchResult, int, int]] = []

    for lang in ("en", "de", "fr"):
        text = texts[(lang, 10_000)]
        chars = len(text)

        tokens_regex = len(tok_regex.tokenize(text).tokens)
        tokenization_results.append(
            (
                _run_bench(
                    f"tokenize RegexTokenizer ({lang}, 10,000 chars)",
                    lambda text=text: tok_regex.tokenize(text),
                    target_total_seconds=args.target_seconds,
                    repeats=args.repeats,
                ),
                chars,
                tokens_regex,
            )
        )

        try:
            tokens_unicode = len(tok_unicode.tokenize(text).tokens)
        except ImportError:
            print(
                "Warning: UnicodeTokenizer unavailable; skipping unicode tokenization benchmarks."
            )
            break
        tokenization_results.append(
            (
                _run_bench(
                    f"tokenize UnicodeTokenizer ({lang}, 10,000 chars)",
                    lambda text=text: tok_unicode.tokenize(text),
                    target_total_seconds=args.target_seconds,
                    repeats=args.repeats,
                ),
                chars,
                tokens_unicode,
            )
        )

    results.extend([r for r, _, _ in tokenization_results])

    print("Environment")
    print(f"- python: {platform.python_version()} ({platform.python_implementation()})")
    print(f"- platform: {platform.platform()}")
    try:
        import regex  # type: ignore
    except Exception:
        print("- regex: (not installed)")
    else:
        version = getattr(regex, "__version__", None)
        print(f"- regex: {version or '(unknown version)'}")
    print()

    _print_table(results)
    _print_tokenization_throughput(tokenization_results)


if __name__ == "__main__":
    main()
