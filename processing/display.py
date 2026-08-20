"""Data loading and similarity-matrix generation (cached as pickles in Files/)."""

import os
import pickle
from typing import List, Optional, Tuple




    def vectorise(self, col_name: str):
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vec_tags = cv.fit_transform(self.new_df[col_name])
        return cosine_similarity(vec_tags, dense_output=False)

    def get_similarity(self, col_name: str) -> None:
        path = f'Files/similarity_tags_{col_name}.pkl'
        if os.path.exists(path):
            return
        with open(path, 'wb') as pickle_file:
            pickle.dump(self.vectorise(col_name), pickle_file)

    def main_(self) -> None:
        self.get_df()
        for col in SIMILARITY_COLUMNS:
            self.get_similarity(col)
        compute_tfidf_similarity(self.new_df)
