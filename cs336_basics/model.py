import torch
import torch.nn as nn
import math
import einops


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(
            self.W,
            std=math.sqrt(2 / (in_features + out_features)),
            a=-3.0,
            b=3.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.W, x, "d_out d_in, batch ... d_in -> batch ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int: Dimension of the embedding vector, ie d_model
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        indices = einops.rearrange(token_ids, "b idx -> (b idx)")
        embeddings = torch.index_select(self.W, 0, indices)
        return einops.rearrange(
            embeddings,
            "(batch seq_len) emb_dim-> batch seq_len emb_dim",
            batch=batch,
            seq_len=seq_len,
        )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model)
        and return a tensor of the same shape
        """
        pass
