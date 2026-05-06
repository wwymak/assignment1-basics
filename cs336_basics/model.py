import torch
import torch.nn as nn
import math
import einops
# from jaxtyping import Float, Tensor


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
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model)
        and return a tensor of the same shape
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        batch_size, sequence_length, d_model = x.shape

        rms = torch.sqrt(self.eps + einops.einsum(torch.pow(x, 2), "b seq_len d_model -> b seq_len") / self.d_model)
        gain = einops.repeat(
            self.gain,
            "d_model -> batch_size sequence_length d_model",
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        rms = einops.repeat(rms, "b seq_len -> b seq_len d_model", d_model=d_model)
        output = x * gain / rms
        return output.to(in_dtype)
