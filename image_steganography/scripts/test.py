import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.encoder_decoder import Encoder, Decoder
import torch

encoder = Encoder()
decoder = Decoder()

cover = torch.randn(2, 3, 256, 256)
secret = torch.randn(2, 1, 256, 256)

stego = encoder(cover, secret)
recovered = decoder(stego)

print(stego.shape)      # (2, 3, 256, 256)
print(recovered.shape)  # (2, 1, 256, 256)
