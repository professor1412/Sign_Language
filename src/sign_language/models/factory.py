"""Construct heads and restore self-contained A/B/C checkpoints."""
from .heads.linear import LinearHead
from .heads.normalized_linear import NormLinear
from .heads.text_similarity import TextSimilarityHead
from .heads.projected_text import ProjectedTextHead
from .heads.controls import control_vectors
from ..text.embedding_table import load_table


HEADS = {"a", "a_norm", "b", "c", "b_rand", "b_shuf"}


def create_head(input_dim, classes, config, vectors=None, special_ids=None):
    kind = config.get("kind", "a")
    if kind == "a":
        return LinearHead(input_dim, classes)
    if kind == "a_norm":
        return NormLinear(input_dim, classes)
    if kind not in HEADS or vectors is None or len(vectors) != classes:
        raise ValueError("Text heads need an ID-matched embedding table")
    vectors = control_vectors(vectors, special_ids, kind, config.get("control_seed", 42))
    kwargs = {"temperature": config.get("temperature", 0.07),
              "learn_temperature": config.get("learn_temperature", True)}
    if kind == "c":
        return ProjectedTextHead(input_dim, vectors, special_ids,
                                 projection_dim=config.get("projection_dim", 512), **kwargs)
    return TextSimilarityHead(input_dim, vectors, special_ids, **kwargs)


def build_recognizer(vocabulary, config, state_dict=None):
    from .recognizer import Recognizer
    head = config.get("head", {})
    vectors, special_ids = None, None
    if head.get("kind", "a") in {"b", "c", "b_rand", "b_shuf"}:
        if state_dict is not None:
            vectors = state_dict["head.text_vectors"]
            special_ids = state_dict["head.special_ids"].tolist()
        else:
            if not head.get("embeddings"):
                raise ValueError("Set head.embeddings to an embeddings.pt bundle")
            bundle = load_table(head["embeddings"], vocabulary)
            vectors, special_ids = bundle["vectors"], bundle["special_ids"]
    model = Recognizer(len(vocabulary["id_to_gloss"]), head_config=head,
                       vectors=vectors, special_ids=special_ids, **config["model"])
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model
