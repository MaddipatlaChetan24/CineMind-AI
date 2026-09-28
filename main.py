from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit_option_menu
from streamlit_extras.stoggle import stoggle

from processing import preprocess, ui
from processing.display import Main= num_pages - 1):
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
        all_movies_page(movies, meta)
    elif choice == 'Personalized For You':
        personalized_page(new_df, movies2, meta)

    st.divider()
    st.caption("Data and images provided by The Movie Database (TMDB) and Kaggle.")


if __name__ == '__main__':
    main()
