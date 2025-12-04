from app.models.hf_embedding import HFEmbeddingModel


class BgeM3Embedding(HFEmbeddingModel):
    """BGE-M3 embedding model wrapper.

    This delegates to the shared HFEmbeddingModel implementation.
    """

    def __init__(self, hf_repo_id: str, device: str = "cuda") -> None:
        super().__init__(hf_repo_id, device)
