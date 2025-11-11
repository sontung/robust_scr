import os

import torch
import torch.nn as nn

DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_sinkhorn_iterations(
    Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


# Code from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py)
def log_optimal_transport(
    scores: torch.Tensor, alpha: torch.Tensor, iters: int
) -> torch.Tensor:
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)

    bins = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([scores, bins], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class DINOv2(nn.Module):

    def __init__(
        self,
        model_name="dinov2_vitb14",
        num_trainable_blocks=4,
        norm_layer=True,
        return_token=True,
    ):
        super().__init__()
        print(
            f"Loading DINOv2 model {model_name} with {num_trainable_blocks} trainable blocks"
        )

        if "dinov2" in model_name:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            self.num_channels = DINOV2_ARCHS[model_name]

        else:
            self.model = torch.hub.load(
                "../dinov3",
                "dinov3_vitb16",
                source="local",
                weights="checkpoints/dinov3_b16",
            )
            self.num_channels = 768
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        if self.num_trainable_blocks == 0:
            # First blocks are frozen
            with torch.no_grad():
                for blk in self.model.blocks:
                    x = blk(x)
            x = x.detach()
        else:
            # First blocks are frozen
            with torch.no_grad():
                for blk in self.model.blocks[: -self.num_trainable_blocks]:
                    x = blk(x)
            x = x.detach()

            # Last blocks are trained
            for blk in self.model.blocks[-self.num_trainable_blocks :]:
                x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)
        if self.return_token:
            return f, t
        return f


class DINOv3(nn.Module):

    def __init__(
        self,
        num_trainable_blocks=4,
        norm_layer=True,
        return_token=True,
    ):
        super().__init__()

        self.model = torch.hub.load(
            "../dinov3",
            "dinov3_vitb16",
            source="local",
            weights="checkpoints/dinov3_b16",
        )
        self.num_channels = 768
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        f, t, _ = self.model.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True, return_extra_tokens=True
        )[0]
        if self.return_token:
            return f, t
        return f


class SALAD_full(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.

    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        num_channels=768,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )

        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        # Dustbin parameter z
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        print(f"{self.num_clusters} clusters, {self.cluster_dim} cluster dim")

    def forward(self, feat):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = feat

        f = self.cluster_features(x).flatten(2)
        t = self.token_features(t)

        p = self.score(x).flatten(2)

        p = log_optimal_transport(p, self.dust_bin, 3)

        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )

        return nn.functional.normalize(f, p=2, dim=-1)


class FullModel(nn.Module):
    def __init__(
        self,
        nb_clusters=64,
        dino_arch="dinov2_vitb14",
        pretrained=False,
        trainable_blocks=2,
    ):
        super().__init__()
        self.encoder = DINOv2(
            model_name=dino_arch, num_trainable_blocks=trainable_blocks
        )

        self.agg = SALAD_full(
            num_clusters=nb_clusters, num_channels=self.encoder.num_channels
        )

        temp_net = self.encoder.model.blocks[-self.encoder.num_trainable_blocks :]
        temp_net2 = self.encoder.model.norm
        torch.save(self.agg.state_dict(), "checkpoints/t1.pth")
        torch.save(temp_net.state_dict(), "checkpoints/t2.pth")
        torch.save(temp_net2.state_dict(), "checkpoints/t3.pth")
        w1 = os.path.getsize("checkpoints/t1.pth")
        w2 = os.path.getsize("checkpoints/t2.pth")
        if self.encoder.num_trainable_blocks == 0:
            w2 = 0
        w3 = os.path.getsize("checkpoints/t3.pth")
        print(f"Size = {(w1 + w2+w3)/1e6} mb")

        if pretrained:
            state_dict = torch.load("checkpoints/cliquemining.ckpt", weights_only=True)[
                "state_dict"
            ]

            enc_state_dict = self.encoder.state_dict()
            agg_state_dict = self.agg.state_dict()
            for k in state_dict.keys():
                if "backbone" in k:
                    enc_state_dict[k.replace("backbone.", "")] = state_dict[k]
                elif "aggregator" in k:
                    agg_state_dict[k.replace("aggregator.", "")] = state_dict[k]
            print("Loading pretrained weights")
            self.encoder.load_state_dict(enc_state_dict)
            self.agg.load_state_dict(agg_state_dict)

    def forward(self, x):
        f, t = self.encoder(x)
        return self.agg((f, t))


class FullModel_DINOV3(nn.Module):
    def __init__(
        self,
        nb_clusters=64,
    ):
        super().__init__()
        self.encoder = DINOv3(num_trainable_blocks=4)

        # Freeze the entire encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.agg = SALAD_full(
            num_clusters=nb_clusters, num_channels=self.encoder.num_channels
        )

        print()

    def forward(self, x):
        f, t = self.encoder(x)
        return self.agg((f, t))


if __name__ == "__main__":
    inp = torch.randn(2, 3, 224, 224)
    model = FullModel(nb_clusters=16, pretrained=False)
    out = model(inp)
    print(out.shape)

    model = FullModel_DINOV3(nb_clusters=16)
    out = model(inp)
    print(out.shape)
    print()
