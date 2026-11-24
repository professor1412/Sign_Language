"""Head C applies a shared projection to frozen fastText rows."""
from .text_similarity import TextSimilarityHead


class ProjectedTextHead(TextSimilarityHead):
    def __init__(self, input_dim, vectors, special_ids, projection_dim=512, **kwargs):
        if type(projection_dim) is not int or projection_dim < 1:
            raise ValueError("Text projection dimension must be a positive integer")
        super().__init__(input_dim, vectors, special_ids, projection_dim=projection_dim, **kwargs)
