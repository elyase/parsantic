from __future__ import annotations

import functools
import textwrap
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

from .types import AlignmentStatus, ExtractResult, FieldEvidence

_PALETTE = [
    "#D2E3FC",
    "#C8E6C9",
    "#FEF0C3",
    "#F9DEDC",
    "#FFDDBE",
    "#EADDFF",
    "#C4E9E4",
    "#FCE4EC",
    "#E8EAED",
    "#DDE8E8",
]


def _escape_html(text: str) -> str:
    return escape(text, quote=True)


def _build_highlighted_html(
    text: str,
    evidence_list: Sequence[FieldEvidence],
    palette: Sequence[str] | Mapping[str, str],
) -> str:
    if not text:
        return '<div class="pv-empty">No source text available.</div>'

    path_to_color: dict[str, str]
    if isinstance(palette, Mapping):
        path_to_color = dict(palette)
    else:
        path_to_color = {}
        for ev in evidence_list:
            if ev.path not in path_to_color:
                color = palette[len(path_to_color) % len(palette)] if palette else _PALETTE[0]
                path_to_color[ev.path] = color

    text_len = len(text)
    start_events: dict[int, list[FieldEvidence]] = {}
    end_events: dict[int, list[FieldEvidence]] = {}

    for ev in evidence_list:
        if ev.char_interval is None:
            continue
        start, end = ev.char_interval
        start = max(0, min(start, text_len))
        end = max(0, min(end, text_len))
        if end <= start:
            continue
        start_events.setdefault(start, []).append(ev)
        end_events.setdefault(end, []).append(ev)

    if not start_events:
        return _escape_html(text)

    points = sorted({0, text_len, *start_events.keys(), *end_events.keys()})
    active: list[FieldEvidence] = []
    html_parts: list[str] = []

    for index, start in enumerate(points[:-1]):
        if start in end_events:
            for ev in end_events[start]:
                if ev in active:
                    active.remove(ev)

        if start in start_events:
            active.extend(start_events[start])

        end = points[index + 1]
        if end <= start:
            continue

        segment_html = _escape_html(text[start:end])
        if not active:
            html_parts.append(segment_html)
            continue

        sorted_active = sorted(
            active,
            key=lambda ev: (
                ev.char_interval[0] if ev.char_interval is not None else 0,
                -(ev.char_interval[1] if ev.char_interval is not None else 0),
                ev.path,
            ),
        )

        wrapped = segment_html
        for ev in reversed(sorted_active):
            color = path_to_color.get(ev.path)
            if color is None:
                color = _PALETTE[len(path_to_color) % len(_PALETTE)]
                path_to_color[ev.path] = color
            tooltip = (
                f"Path: {ev.path} | Value: {ev.value_preview} | Status: {ev.alignment_status.value}"
            )
            wrapped = (
                f'<span class="pv-highlight" style="--pv-color: {color};" '
                f'data-field-path="{_escape_html(ev.path)}" '
                f'data-field-value="{_escape_html(ev.value_preview)}" '
                f'data-alignment-status="{_escape_html(ev.alignment_status.value)}" '
                f'data-tooltip="{_escape_html(tooltip)}">{wrapped}</span>'
            )
        html_parts.append(wrapped)

    return "".join(html_parts)


@functools.cache
def _generate_tooltip_js() -> str:
    return textwrap.dedent(
        """
        (() => {
          const tooltip = document.createElement('div');
          tooltip.id = 'pv-tooltip';
          document.body.appendChild(tooltip);

          const hideTooltip = () => {
            tooltip.classList.remove('visible');
          };

          const moveTooltip = (event) => {
            tooltip.style.left = `${event.pageX + 14}px`;
            tooltip.style.top = `${event.pageY + 14}px`;
          };

          document.addEventListener('mouseover', (event) => {
            const target = event.target.closest('[data-tooltip]');
            if (!target) {
              return;
            }
            const content = target.getAttribute('data-tooltip');
            if (!content) {
              return;
            }
            tooltip.textContent = content;
            tooltip.classList.add('visible');
            moveTooltip(event);
          });

          document.addEventListener('mousemove', (event) => {
            if (tooltip.classList.contains('visible')) {
              moveTooltip(event);
            }
          });

          document.addEventListener('mouseout', (event) => {
            const from = event.target.closest('[data-tooltip]');
            if (!from) {
              return;
            }
            const to = event.relatedTarget?.closest?.('[data-tooltip]') || null;
            if (from !== to) {
              hideTooltip();
            }
          });

          document.addEventListener('scroll', hideTooltip, true);
        })();
        """
    ).strip()


@functools.cache
def _generate_css(palette: tuple[str, ...]) -> str:
    palette_vars = "\n".join(f"  --pv-palette-{idx}: {color};" for idx, color in enumerate(palette))
    return textwrap.dedent(
        f"""
        :root {{
          --pv-bg: #f8fafc;
          --pv-surface: #ffffff;
          --pv-border: #d9e0ea;
          --pv-text: #1f2937;
          --pv-muted: #6b7280;
          --pv-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
          --pv-tooltip-bg: #111827;
          --pv-tooltip-text: #f9fafb;
{palette_vars}
        }}

        @media (prefers-color-scheme: dark) {{
          :root {{
            --pv-bg: #0f172a;
            --pv-surface: #111b2e;
            --pv-border: #273449;
            --pv-text: #e5e7eb;
            --pv-muted: #9ca3af;
            --pv-shadow: 0 8px 24px rgba(2, 6, 23, 0.45);
            --pv-tooltip-bg: #020617;
            --pv-tooltip-text: #f8fafc;
          }}
        }}

        * {{
          box-sizing: border-box;
        }}

        body {{
          margin: 0;
          padding: 1.25rem;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          line-height: 1.45;
          background: var(--pv-bg);
          color: var(--pv-text);
        }}

        .pv-container {{
          max-width: 1200px;
          margin: 0 auto;
          display: grid;
          gap: 1rem;
        }}

        .pv-header,
        .pv-card {{
          background: var(--pv-surface);
          border: 1px solid var(--pv-border);
          border-radius: 0.75rem;
          box-shadow: var(--pv-shadow);
          padding: 1rem;
        }}

        .pv-title {{
          margin: 0;
          font-size: 1.25rem;
          font-weight: 700;
        }}

        .pv-muted {{
          margin: 0.4rem 0 0;
          color: var(--pv-muted);
          font-size: 0.9rem;
        }}

        .pv-legend-list {{
          list-style: none;
          margin: 0;
          padding: 0;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 0.5rem;
        }}

        .pv-legend-item {{
          display: flex;
          align-items: center;
          gap: 0.5rem;
          min-width: 0;
        }}

        .pv-swatch {{
          width: 1rem;
          height: 1rem;
          border-radius: 0.25rem;
          border: 1px solid rgba(15, 23, 42, 0.12);
          flex-shrink: 0;
        }}

        .pv-path {{
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 0.86rem;
          overflow-wrap: anywhere;
        }}

        .pv-summary-table,
        .pv-unanchored-table {{
          width: 100%;
          border-collapse: collapse;
          margin-top: 0.6rem;
        }}

        .pv-summary-table th,
        .pv-summary-table td,
        .pv-unanchored-table th,
        .pv-unanchored-table td {{
          border: 1px solid var(--pv-border);
          padding: 0.45rem 0.55rem;
          text-align: left;
          vertical-align: top;
          font-size: 0.9rem;
        }}

        .pv-result-title {{
          margin: 0;
          font-size: 1rem;
        }}

        .pv-text-container {{
          margin-top: 0.7rem;
          border: 1px solid var(--pv-border);
          border-radius: 0.5rem;
          max-height: 28rem;
          overflow: auto;
          background: color-mix(in srgb, var(--pv-surface) 80%, transparent);
        }}

        .pv-text {{
          margin: 0;
          padding: 0.8rem;
          white-space: pre-wrap;
          word-break: break-word;
          font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
          font-size: 0.88rem;
        }}

        .pv-empty {{
          padding: 0.5rem;
          color: var(--pv-muted);
          font-style: italic;
        }}

        .pv-highlight {{
          background: linear-gradient(transparent 38%, var(--pv-color) 38%);
          border-radius: 0.2rem;
          padding: 0 0.03rem;
          cursor: help;
        }}

        #pv-tooltip {{
          position: absolute;
          z-index: 9999;
          max-width: 460px;
          padding: 0.45rem 0.6rem;
          border-radius: 0.4rem;
          background: var(--pv-tooltip-bg);
          color: var(--pv-tooltip-text);
          font-size: 0.78rem;
          line-height: 1.4;
          pointer-events: none;
          white-space: pre-wrap;
          opacity: 0;
          transform: translateY(2px);
          transition: opacity 0.08s ease;
        }}

        #pv-tooltip.visible {{
          opacity: 1;
          transform: translateY(0);
        }}
        """
    ).strip()


def _build_color_mapping(
    results: Sequence[ExtractResult[Any]], palette: Sequence[str]
) -> dict[str, str]:
    path_to_color: dict[str, str] = {}
    for result in results:
        for ev in result.evidence:
            if ev.path not in path_to_color:
                path_to_color[ev.path] = palette[len(path_to_color) % len(palette)]
    return path_to_color


def _status_count(evidence: Sequence[FieldEvidence], status: AlignmentStatus) -> int:
    return sum(1 for item in evidence if item.alignment_status == status)


def visualize(
    result: ExtractResult[Any] | list[ExtractResult[Any]],
    *,
    output_path: str | Path | None = None,
) -> str:
    results = result if isinstance(result, list) else [result]
    path_to_color = _build_color_mapping(results, _PALETTE)

    legend_items = "\n".join(
        (
            '<li class="pv-legend-item">'
            f'<span class="pv-swatch" style="background:{color};"></span>'
            f'<span class="pv-path">{_escape_html(path)}</span>'
            "</li>"
        )
        for path, color in path_to_color.items()
    )

    legend_html = (
        f'<ul class="pv-legend-list">{legend_items}</ul>'
        if path_to_color
        else '<p class="pv-muted">No field evidence to highlight.</p>'
    )

    result_sections: list[str] = []

    for index, item in enumerate(results, start=1):
        raw_text = item.raw_text or ""
        text_evidence = [ev for ev in item.evidence if ev.char_interval is not None]
        no_interval_evidence = [ev for ev in item.evidence if ev.char_interval is None]

        highlighted_text = _build_highlighted_html(raw_text, text_evidence, path_to_color)

        no_interval_rows = "\n".join(
            (
                "<tr>"
                f'<td class="pv-path">{_escape_html(ev.path)}</td>'
                f"<td>{_escape_html(ev.value_preview)}</td>"
                f"<td>{_escape_html(ev.alignment_status.value)}</td>"
                f"<td>{_escape_html(ev.source)}</td>"
                "</tr>"
            )
            for ev in no_interval_evidence
        )

        no_interval_html = (
            '<table class="pv-unanchored-table">'
            "<thead><tr><th>Field Path</th><th>Value</th><th>Status</th><th>Source</th></tr></thead>"
            f"<tbody>{no_interval_rows}</tbody></table>"
            if no_interval_evidence
            else '<p class="pv-muted">No evidence without character intervals.</p>'
        )

        result_sections.append(
            textwrap.dedent(
                f"""
                <section class="pv-card pv-result">
                  <h2 class="pv-result-title">Result {index}</h2>
                  <table class="pv-summary-table">
                    <thead>
                      <tr>
                        <th>Document ID</th>
                        <th>Total Fields</th>
                        <th>Exact</th>
                        <th>Lesser</th>
                        <th>Fuzzy</th>
                        <th>Unmatched</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>{_escape_html(item.document_id or "(none)")}</td>
                        <td>{len(item.evidence)}</td>
                        <td>{_status_count(item.evidence, AlignmentStatus.MATCH_EXACT)}</td>
                        <td>{_status_count(item.evidence, AlignmentStatus.MATCH_LESSER)}</td>
                        <td>{_status_count(item.evidence, AlignmentStatus.MATCH_FUZZY)}</td>
                        <td>{_status_count(item.evidence, AlignmentStatus.UNMATCHED)}</td>
                      </tr>
                    </tbody>
                  </table>

                  <h3>Highlighted Source Text</h3>
                  <div class="pv-text-container">
                    <pre class="pv-text">{highlighted_text}</pre>
                  </div>

                  <h3>Evidence Without Character Intervals</h3>
                  {no_interval_html}
                </section>
                """
            ).strip()
        )

    html_output = textwrap.dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Parsantic Interactive Visualization</title>
          <style>{_generate_css(tuple(_PALETTE))}</style>
        </head>
        <body>
          <main class="pv-container">
            <header class="pv-header">
              <h1 class="pv-title">Parsantic Interactive HTML Visualization</h1>
              <p class="pv-muted">Hover any highlighted span to inspect field-level evidence.</p>
            </header>

            <section class="pv-card pv-legend">
              <h2 class="pv-result-title">Field Legend</h2>
              {legend_html}
            </section>

            {"".join(result_sections)}
          </main>
          <script>{_generate_tooltip_js()}</script>
        </body>
        </html>
        """
    ).strip()

    if output_path is not None:
        Path(output_path).write_text(html_output, encoding="utf-8")

    return html_output
