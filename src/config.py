import dataclasses
import torch

@dataclasses.dataclass
class MicaConfig:
    # Text constraints
    block_size: int = 512        # Maximum context length
    vocab_size: int = 5000       # Custom tokenizer size (do not use 50k+)

    # The Mica Geometry (Small, compressed manifold)
    n_layer: int = 10             # Depth
    n_head: int = 11             # Attention heads (1 large + 2 medium + 8 small)
    n_embd: int = 192            # d_model (forces semantic compression)
    
    # Regularization
    dropout: float = 0.1         # High dropout for low-data regime
    bias: bool = False           # True: bias in Linears/LayerNorms. False: better stability.
    
    # MLP expansion ratio (2 = small, 4 = standard GPT)
    ffn_ratio: int = 4
    
    # Origami folding
    max_loops: int = 8          # Maximum loop count per block (loop-specific LNs)
    
    # Apple Silicon constraint
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
