"""Generate inspectable fastText embeddings without changing label IDs."""
from ...text.embedding_table import build_table
from ...text.fasttext_encoder import load_fasttext
from ...core.settings import read_json, write_json
from ...core.provenance import file_sha256
from ...features.cache import atomic_torch_save


def build_embeddings(args):
    if args.output.exists():
        raise ValueError("Embedding output exists; choose a new directory")
    model = load_fasttext(args.fasttext_model)
    bundle = build_table(read_json(args.vocabulary), model.get_word_vector,
                         {"kind": "fasttext", "model_sha256": file_sha256(args.fasttext_model),
                          "filename": args.fasttext_model.name}, args.strip_prefixes)
    args.output.mkdir(parents=True)
    write_json(args.output / "audit.json", {k: v for k, v in bundle.items() if k != "vectors"})
    atomic_torch_save(bundle, args.output / "embeddings.pt")
    print(f"Saved {len(bundle['rows'])} ID-ordered rows to {args.output}")
