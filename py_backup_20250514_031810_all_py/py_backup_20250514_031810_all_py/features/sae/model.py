import torch, torch.nn as nn

class SAE(nn.Module):
    """Sparse Auto-Encoder for geometric descriptor."""
    def __init__(self, bottleneck=512):
        super().__init__()
        bottleneck = int(bottleneck)
        print(f"[SAE] bottleneck = {bottleneck}")
        ch = [64,128,256,512]

        # ───────── Encoder ─────────
        self.enc = nn.Sequential(
            nn.Conv2d(3,ch[0],4,2,1), nn.ReLU(),
            nn.Conv2d(ch[0],ch[1],4,2,1), nn.ReLU(),
            nn.Conv2d(ch[1],ch[2],4,2,1), nn.ReLU(),
            nn.Conv2d(ch[2],ch[3],4,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(ch[3], bottleneck)

        # ───────── Decoder ─────────
        self.fc_dec  = nn.Linear(bottleneck, ch[3]*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch[3],ch[2],4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(ch[2],ch[1],4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(ch[1],ch[0],4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(ch[0],3,4,2,1), nn.Tanh()
        )

    # ---------- public API ----------
    def encode(self, x):
        return self.fc(self.flatten(self.enc(x)))

    def decode(self, z):
        h = self.dec[0].in_channels          # =512，避免 magic number
        y = self.fc_dec(z).view(-1, h, 4, 4)
        return self.dec(y)

    # ---------- training / inference ----------
    def forward(self, x, mask=None):
        """
        Args:
            x (tensor)  : 3×256×256 input
            mask (None|tensor) : same shape as z，若給定就先 element-wise 相乘
        Returns:
            recon (tensor) : 3×64×64
            z_full         : 未遮罩 latent
            z_sparse       : 遮罩後 latent（若 mask=None，回傳 z_full）
        """
        z_full = self.encode(x)
        z_sparse = z_full if mask is None else z_full * mask.detach()  # ← 條件式遮罩
        recon = self.decode(z_sparse)
        return recon, z_full, z_sparse
