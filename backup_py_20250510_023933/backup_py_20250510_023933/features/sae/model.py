import torch, torch.nn as nn

class SAE(nn.Module):
    """Sparse Auto‑Encoder for geometric descriptor."""
    def __init__(self, bottleneck=512):
        bottleneck = int(bottleneck)
        print(f"[SAE] bottleneck = {bottleneck}")
        super().__init__()
        ch = [64,128,256,512]
        self.enc = nn.Sequential(
            nn.Conv2d(3,ch[0],4,2,1), nn.ReLU(),
            nn.Conv2d(ch[0],ch[1],4,2,1), nn.ReLU(),
            nn.Conv2d(ch[1],ch[2],4,2,1), nn.ReLU(),
            nn.Conv2d(ch[2],ch[3],4,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(ch[3], bottleneck)
        # decoder (mirror)
        self.fc_dec  = nn.Linear(bottleneck, ch[3]*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch[3],ch[2],4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(ch[2],ch[1],4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(ch[1],ch[0],4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(ch[0],3,4,2,1), nn.Tanh()
        )
    def encode(self,x):
        z = self.flatten(self.enc(x))
        return self.fc(z)
    def forward(self,x):
        z = self.encode(x)
        y = self.fc_dec(z).view(-1,512,4,4)
        return self.dec(y), z
        # ───────── 新增：讓外部 script 可直接呼叫 model.decode(z) ─────────
    def decode(self, z):
        """
        只負責『把 latent z 還原成 64×64 圖』，**不**做 sparsity 遮罩內部判斷
        """
        h = self.dec[0].in_channels         # = 512，避免寫死 magic number
        y = self.fc_dec(z).view(-1, h, 4, 4)
        return self.dec(y)