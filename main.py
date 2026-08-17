from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit_option_menu
from streamlit_extras.stoggle import stoggle

from processing import preprocess, ui
from processing.display import Main
from processing.embeddings import semantic_recommend
from processing.hybrid import hybrid_recommend
from processing.preferences import (
    UserPreferences,
    filter_by_preferences,
    preference_controls,
)

st.



=

def show_scored_movie_grid(
    movies: List[Tuple[str, int, float]], meta: Dict[int, Tuple[float, str]]
) -> None:
    """Display a grid of movies that include a similarity score."""
    cols = st.columns(5)
    for col, (title, movie_id, score) in zip(cols, movies[:5]):
        rating, year = meta.get(movie_id, (0, ''))
        with col:
            st.image(preprocess.fetch_posters(movie_id), width="stretch")
            caption = format_caption(title, rating, year)
            caption += f"  \nMatch: {score:.0%}"
            st.caption(caption)


def recommend_page(new_df: pd.DataFrame, meta: Dict[int, Tuple[float, str]]) -> None:
    st.title("Movie Recommender System")
    st.caption("Pick a movie and get similar suggestions based on tags, genres, keywords, cast, and production company.")

    selected_movie = st.selectbox("Select a movie...", new_df["title"].values)

    if st.button("Recommend", type="primary", use_container_width=True):
        st.session_state.selected_movie = selected_movie
        with st.spinner("Finding similar movies..."):
            st.session_state.recs = compute_recommendations(new_df, selected_movie)
            st.session_state.semantic_recs = semantic_recommend(new_df, selected_movie, top_n=25)

    if recs := st.session_state.get("recs"):
        st.subheader(f"Recommendations for **{st.session_state.selected_movie}**")

        # Content-based tabs + Semantic (TF-IDF) tab
        tab_labels = list(recs.keys()) + ["Semantic (TF-IDF)"]
        tabs = st.tabs(tab_labels)

        for tab, (label, movies) in zip(tabs[:-1], recs.items()):
            with tab:
                show_movie_grid(movies[:5], meta)

        # Semantic tab
        with tabs[-1]:
            sem = st.session_state.get("semantic_recs", [])
            if sem:
                show_scored_movie_grid(sem, meta)
            else:
                st.info("Click **Recommend** to generate semantic results.")


def details_page(new_df: pd.DataFrame) -> None:
    st.title("Movie Details")
    st.caption("Overview, cast, release info, and more.")

    titles = new_df["title"].values
    default_movie = st.session_state.get("selected_movie", titles[0])
    default_index = list(titles).index(default_movie) if default_movie in titles else 0

    selected_movie = st.selectbox("Choose a movie...", titles, index=default_index)
    info = preprocess.get_details(selected_movie)

    year = info.release_date[:4] if info.release_date else ''
    ui.hero(info.backdrop, selected_movie, subtitle=f"{year} · {', '.join(info.genres)}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rating", f"{info.rating}/10")
    col2.metric("Votes", f"{info.vote_count:,}")
    col3.metric("Runtime", f"{info.runtime} min")
    col4.metric("Release", info.release_date)

    st.write("**Overview**")
    st.write(info.overview)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Budget**")
        st.markdown(f"<span style='font-size:1.15rem;'>${info.budget:,}</span>", unsafe_allow_html=True)
    with col2:
        st.write("**Revenue**")
        st.markdown(f"<span style='font-size:1.15rem;'>${info.revenue:,}</span>", unsafe_allow_html=True)
    with col3:
        st.write("**Directed by**")
        st.markdown(f"<span style='font-size:1.15rem;'>{info.director[0] if info.director else '—'}</span>",
                    unsafe_allow_html=True)

    st.write("**Genres**")
    ui.chips(info.genres)
    st.write("**Available in**")
    ui.chips(info.languages)

    st.header("Cast")
    cols = st.columns(5)
    for col, (name, person_id) in zip(cols, zip(info.cast[:5], info.cast_ids[:5])):
        with col:
            url, biography = preprocess.fetch_person_details(person_id)
            st.image(url, width="stretch")
            st.markdown(f"<div style='text-align:center; font-weight:600;'>{name}</div>", unsafe_allow_html=True)
            stoggle("Show More", biography or "No biography available.")


def all_movies_page(
    movies: pd.DataFrame, meta: Dict[int, Tuple[float, str]]
) -> None:
    st.title("All Movies")
    st.caption("Browse the full collection.")

    search = st.text_input("Search movies", placeholder="e.g. Avatar, Batman, Inception...")
    if search:
        filtered = movies[movies["title"].str.contains(search, case=False, na=False)]
    else:
        filtered = movies

    num_pages = max((len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    page = min(st.session_state.get("page", 0), num_pages - 1)

    col1, col2, col3 = st.columns([1, 9, 1])
    with col1:
        if st.button("Prev", use_container_width=True, disabled=page == 0):
            st.session_state.page = page - 1
    with col3:
        if st.button("Next", use_container_width=True, disabled=page >= num_pages - 1):
            st.session_state.page = page + 1
    with col2:
        st.caption(f"Page {page + 1} of {num_pages} · {len(filtered):,} movies")

    page_rows = filtered.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    for row_start in range(0, len(page_rows), 5):
        cols = st.columns(5)
        for col, (_, row) in zip(cols, page_rows.iloc[row_start:row_start + 5].iterrows()):
            with col:
                rating, year = meta.get(row["movie_id"], (0, ''))
                st.image(preprocess.fetch_posters(row["movie_id"]), width="stretch")
                st.caption(format_caption(row["title"], rating, year))


def personalized_page(
    new_df: pd.DataFrame,
    movies2: pd.DataFrame,
    meta: Dict[int, Tuple[float, str]],
) -> None:
    """Personalized For You — hybrid recommendations filtered by user preferences."""
    st.title("Personalized For You")
    st.caption("Get tailored recommendations by setting your preferences and picking a seed movie.")

    # --- User Preferences Panel ---
    with st.expander("Set Your Preferences", expanded=True):
        prefs = preference_controls()

    st.divider()

    # --- Seed Movie & Blend Control ---
    selected_movie = st.selectbox(
        "Pick a seed movie...", new_df["title"].values, key="personalized_seed"
    )

    content_weight = st.slider(
        "Blend: Content ↔ Semantic",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 = pure semantic (TF-IDF), 1.0 = pure content (BoW), 0.5 = balanced hybrid.",
    )

    if st.button("Get Personalized Picks", type="primary", use_container_width=True):
        with st.spinner("Crafting your personalized recommendations..."):
            raw = hybrid_recommend(
                new_df, selected_movie, content_weight=content_weight, top_n=50
            )
            filtered = filter_by_preferences(raw, movies2, prefs)
            st.session_state.personalized_recs = filtered
            st.session_state.personalized_movie = selected_movie
            st.session_state.personalized_weight = content_weight

    if recs := st.session_state.get("personalized_recs"):
        seed = st.session_state.get("personalized_movie", "")
        w = st.session_state.get("personalized_weight", 0.5)
        st.subheader(f"Picks based on **{seed}**")

        blend_label = (
            "Pure Semantic" if w == 0.0
            else "Pure Content" if w == 1.0
            else f"Hybrid ({w:.0%} content, {1 - w:.0%} semantic)"
        )
        st.caption(f"Strategy: {blend_label}")

        if not recs:
            st.warning("No movies match your current preferences. Try widening your filters.")
        else:
            # Show up to 3 rows of 5
            for row_start in range(0, min(len(recs), 15), 5):
                show_scored_movie_grid(recs[row_start:row_start + 5], meta)


def main() -> None:
    with st.spinner("Loading movie data..."):
        new_df, movies, movies2 = load_data()
    meta = movie_meta(movies2)

    choice = streamlit_option_menu.option_menu(
        menu_title="What are you looking for?",
        options=[
            'Recommend me a similar movie',
            'Describe me a movie',
            'Check all Movies',
            'Personalized For You',
        ],
        icons=['film', 'film', 'film', 'stars'],
        menu_icon='list',
        orientation="horizontal",
        default_index=0,
    )

    if choice == 'Recommend me a similar movie':
        recommend_page(new_df, meta)
    elif choice == 'Describe me a movie':
        details_page(new_df)
    elif choice == 'Check all Movies':
        all_movies_page(movies, meta)
    elif choice == 'Personalized For You':
        personalized_page(new_df, movies2, meta)

    st.divider()
    st.caption("Data and images provided by The Movie Database (TMDB) and Kaggle.")


if __name__ == '__main__':
    main()
