"""Reusable UI helpers: theme CSS injection and small display components."""

from typing import Iterable, Optional

import streamlit as st

_CSS = """
<style>
/* Rounded, shadowed images everywhere */
div[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
    transition: transform 0.18s ease;
}
div[data-testid="stImage"] img:hover {
    transform: translateY(-3px);
}

/* Hero backdrop banner */
.hero {
    position: relative;
    width: 100%;
    height: 340px;
    border-radius: 16px;
    background-size: cover;
    background-position: center 25%;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.2rem;
}
.hero::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(11, 11, 15, 0) 30%, rgba(11, 11, 15, 0.35) 60%, rgba(11, 11, 15, 0.95) 100%);
}

/* Pill chips (genres, languages) */
.chip {
    display: inline-block;
    padding: 3px 12px;
    margin: 2px 6px 2px 0;
    border-radius: 999px;
    background: rgba(229, 9, 20, 0.14);
    border: 1px solid rgba(229, 9, 20, 0.5);
    color: #F5F5F5;
    font-size: 0.85rem;
}

/* Rounded buttons */
.stButton > button {
    border-radius: 10px;
}

/* Bigger metric values */
div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}
</style>
"""


def apply_theme() -> None:
    """Inject the app-wide stylesheet."""
    st.markdown(_CSS, unsafe_allow_html=True)


def chips(items: Iterable[str]) -> None:
    """Render a list of items as pill chips."""
    if not items:
        return
    html = "".join(f'<span class="chip">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def hero(backdrop_url: str, movie_title: str, subtitle: Optional[str] = None) -> None:
    """Render a backdrop hero banner with the movie title overlaid."""
    if not backdrop_url:
        st.title(f"{movie_title}")
        if subtitle:
            st.caption(subtitle)
        return
    st.markdown(
        f'<div class="hero" style="background-image: url({backdrop_url});">'
        f'<div style="position:absolute; bottom:18px; left:24px; right:24px;">'
        f'<span style="font-size:2.1rem; font-weight:700; color:#fff; text-shadow:0 2px 10px rgba(0,0,0,.7);">{movie_title}</span>'
        + ((f'<div style="color:#d9d9de; margin-top:6px; text-shadow:0 1px 6px rgba(0,0,0,.8);">{subtitle}</div>') if subtitle else '')
        + '</div></div>',
        unsafe_allow_html=True,
    )