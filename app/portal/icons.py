"""Inline SVG icons drawn on one 24x24 grid with a 1.75px stroke, so weight and size match.

Icons use ``currentColor`` and inherit the colour of the surrounding text in either theme.
"""
from __future__ import annotations

_PATHS = {
    # theme switch
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4'
           'M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
    # product pillars and pipeline stages
    "database": '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.5"/><path d="M4.5 5.5v13c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5v-13'
                'M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5"/>',
    "shield": '<path d="M12 3l7.5 3v5.5c0 4.6-3.1 8.3-7.5 9.5-4.4-1.2-7.5-4.9-7.5-9.5V6z"/><path d="M8.8 12.2l2.2 2.2 4.3-4.4"/>',
    "alert": '<path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.2M12 17v.01"/>',
    "filter": '<path d="M3.5 5h17l-6.5 7.5v6l-4 2v-8z"/>',
    "layers": '<path d="M12 3l9 4.5-9 4.5-9-4.5z"/><path d="M3 12l9 4.5 9-4.5M3 16.5l9 4.5 9-4.5"/>',
    "user": '<circle cx="12" cy="8" r="3.8"/><path d="M4.5 20.5c1.1-3.6 4-5.5 7.5-5.5s6.4 1.9 7.5 5.5"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "flag": '<path d="M5.5 21V4M5.5 4.5h11l-2 4 2 4h-11"/>',
    "file": '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M9.5 13h7M9.5 17h7"/>',
    # footer
    "code": '<path d="M8.5 7L3.5 12l5 5M15.5 7l5 5-5 5M13.5 4.5l-3 15"/>',
    "terminal": '<path d="M4 6.5l5 5.5-5 5.5M11 18h9"/>',
    "refresh": '<path d="M20 11.5a8 8 0 0 0-14.3-4.5L4 9M4 12.5a8 8 0 0 0 14.3 4.5L20 15"/><path d="M4 4.5V9h4.5M20 19.5V15h-4.5"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5l5 5"/>',
    "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "x": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
}

_GITHUB = ('<path fill="currentColor" stroke="none" d="M12 2.2a9.9 9.9 0 0 0-3.1 19.3c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.3-1.3-3.3-1.3'
           '-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.9.8.1-.6.3-1.1.6-1.3-2.2-.3-4.5-1.1-4.5-4.9'
           ' 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.7 0 0 .8-.3 2.7 1a9.4 9.4 0 0 1 5 0c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.4.1 2.7.6.7 1 1.6 1'
           ' 2.7 0 3.8-2.3 4.6-4.5 4.9.4.3.7.9.7 1.9v2.8c0 .3.2.6.7.5A9.9 9.9 0 0 0 12 2.2z"/>')


def icon(name: str, size: int = 16, label: str | None = None) -> str:
    body = _GITHUB if name == "github" else _PATHS[name]
    aria = f'role="img" aria-label="{label}"' if label else 'aria-hidden="true"'
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" {aria}>{body}</svg>')
