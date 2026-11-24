"""Optional binary fastText loader; no downloads or import during head training."""


def load_fasttext(path):
    if not path.is_file():
        raise ValueError("German fastText .bin model missing; see docs/phase3.md")
    try:
        import fasttext
    except ImportError as error:
        raise ValueError("Install the optional fasttext package to build embeddings; see docs/phase3.md") from error
    model = fasttext.load_model(str(path))
    if model.get_dimension() != 300:
        raise ValueError("Expected a 300-dimensional German fastText model")
    return model
