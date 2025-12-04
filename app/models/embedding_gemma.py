from app.models.hf_embedding import HFEmbeddingModel


class EmbeddingGemmaEmbedding(HFEmbeddingModel):
    """Embedding Gemma wrapper using the shared HF embedding implementation."""

    def __init__(self, hf_repo_id: str, device: str = "cuda") -> None:
        super().__init__(hf_repo_id, device)
