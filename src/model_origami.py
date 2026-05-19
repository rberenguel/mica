import torch
import torch.nn as nn
from torch.nn import functional as F
from config import MicaConfig
from model import MultiResolutionAttention, MLP


class OrigamiBlock(nn.Module):
    def __init__(self, config, max_loops: int = 8):
        super().__init__()
        self.max_loops = max_loops
        self.ln_1 = nn.ModuleList([
            nn.LayerNorm(config.n_embd, bias=config.bias) for _ in range(max_loops)
        ])
        self.ln_2 = nn.ModuleList([
            nn.LayerNorm(config.n_embd, bias=config.bias) for _ in range(max_loops)
        ])
        self.attn = MultiResolutionAttention(config)
        self.mlp  = MLP(config)
        self.depth_embed = nn.Embedding(max_loops - 1, config.n_embd) if max_loops > 1 else None
        if self.depth_embed is not None:
            nn.init.zeros_(self.depth_embed.weight)

    def forward(self, x, start_pos: int = 0, loop_idx: int = 0):
        assert 0 <= loop_idx < self.max_loops, (
            f"loop_idx {loop_idx} out of range [0, {self.max_loops})"
        )
        if loop_idx > 0 and self.depth_embed is not None:
            x = x + self.depth_embed(torch.tensor(loop_idx - 1, device=x.device))
        x = x + self.attn(self.ln_1[loop_idx](x), start_pos)
        x = x + self.mlp(self.ln_2[loop_idx](x))
        return x


class OrigamiTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n = config.n_layer

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([OrigamiBlock(config, max_loops=config.max_loops) for _ in range(n)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Segment layout: list of (physical_block_index, loop_count)
        # Default: all independent
        self.segments = [(i, 1) for i in range(n)]

        self.apply(self._init_weights)
        for block in self.transformer.h:
            if block.depth_embed is not None:
                nn.init.zeros_(block.depth_embed.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos: int = 0):
        b, t = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        h = self.transformer.h

        for phys_idx, loop_count in self.segments:
            block = h[phys_idx]
            for loop_idx in range(loop_count):
                x = block(x, start_pos, loop_idx=loop_idx)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss   = None

        return logits, loss

    @property
    def layout(self):
        parts = []
        for _, loop_count in self.segments:
            if loop_count == 1:
                parts.append("1")
            else:
                parts.append(f"*{loop_count}")
        return " ".join(parts)

    @property
    def bottom_loops(self):
        """Backward compat: loop count of first segment if folded, else 1."""
        if self.segments and self.segments[0][1] > 1:
            return self.segments[0][1]
        return 1

    @property
    def top_loops(self):
        """Backward compat: loop count of last segment if folded, else 1."""
        if self.segments and self.segments[-1][1] > 1:
            return self.segments[-1][1]
        return 1


# ── Layout parser ─────────────────────────────────────────────────────────────

def parse_layout(layout: str, n_layer: int, max_loops: int = None) -> list[int]:
    """
    Parse a layout string into a list of loop counts.

    Examples:
        "6"        -> [1, 1, 1, 1, 1, 1]   (6 independent)
        "*2 4"     -> [2, 1, 1, 1, 1]       (1 folded + 4 independent)
        "*2 *2 2"  -> [2, 2, 1, 1]          (2 folded + 2 independent)
        "2 *2 2"   -> [1, 1, 2, 1, 1]       (2 ind + 1 folded + 2 ind)
        "[*2 4]"   -> same as "*2 4"        (brackets stripped)
    """
    layout = layout.strip()
    if layout.startswith('[') and layout.endswith(']'):
        layout = layout[1:-1]
    layout = layout.strip()

    if not layout:
        raise ValueError("Empty layout string")

    tokens = layout.split()
    spec = []
    for tok in tokens:
        if tok.startswith('*'):
            loop_count = int(tok[1:])
            if loop_count < 2:
                raise ValueError(f"Folded segment must have loop count >= 2, got {tok}")
            spec.append(loop_count)
        else:
            count = int(tok)
            if count < 1:
                raise ValueError(f"Independent segment must have count >= 1, got {tok}")
            spec.extend([1] * count)

    total = sum(spec)
    if total != n_layer:
        raise ValueError(
            f"Layout sums to {total} logical layers, but n_layer={n_layer}"
        )

    if max_loops is not None:
        for loop_count in spec:
            if loop_count > max_loops:
                raise ValueError(
                    f"Loop count {loop_count} exceeds max_loops={max_loops}"
                )

    return spec


# ── Layout application ────────────────────────────────────────────────────────

def _device_of(model: OrigamiTransformer):
    return next(model.parameters()).device


def rebuild_aliases(model: OrigamiTransformer):
    """After load_state_dict or layout change, re-establish parameter aliases."""
    h = model.transformer.h
    for phys_idx, loop_count in model.segments:
        if loop_count > 1:
            for i in range(1, loop_count):
                alias_idx = phys_idx + i
                if alias_idx < len(h):
                    h[alias_idx] = h[phys_idx]


def apply_layout(model: OrigamiTransformer, layout: str):
    """
    Transition to a new layout.  Folds create aliases; unfolds create fresh
    blocks initialised from the shared anchor they were folded from.
    
    LN-aware unfold: newly-unfolded blocks inherit the loop-specific LayerNorm
    that matches their former loop role, making the forward pass continuous.
    """
    n = model.config.n_layer
    h = model.transformer.h
    device = _device_of(model)

    spec = parse_layout(layout, n, max_loops=model.config.max_loops)

    # Build target segments with leftmost anchors
    target_segments = []
    logical_pos = 0
    for loop_count in spec:
        target_segments.append((logical_pos, loop_count))
        logical_pos += loop_count

    # Build current logical -> physical mapping
    current_map = []
    for phys_idx, loop_count in model.segments:
        for _ in range(loop_count):
            current_map.append(phys_idx)

    # Build current segment start positions (for loop_idx lookup)
    current_seg_start = {}
    seg_start = 0
    for phys_idx, loop_count in model.segments:
        current_seg_start[phys_idx] = seg_start
        seg_start += loop_count

    # Find currently independent physical blocks
    current_independent = set()
    for phys_idx, loop_count in model.segments:
        if loop_count == 1:
            current_independent.add(phys_idx)

    # Initialise newly-unfolded blocks from their previous shared anchor
    logical_pos = 0
    for phys_idx, loop_count in target_segments:
        if loop_count == 1:
            if phys_idx not in current_independent:
                prev_phys = current_map[logical_pos]
                if prev_phys != phys_idx:
                    new_block = OrigamiBlock(model.config, max_loops=model.config.max_loops)
                    new_block.load_state_dict(h[prev_phys].state_dict())
                    new_block.to(device)

                    # LN-aware: map the old loop_idx's LNs to the new block's ln_1[0]/ln_2[0]
                    old_loop_count = model.segments[[s[0] for s in model.segments].index(prev_phys)][1]
                    if old_loop_count > 1:
                        old_loop_idx = logical_pos - current_seg_start[prev_phys]
                        anchor = h[prev_phys]
                        # Copy the loop-specific LNs that match this block's old role
                        with torch.no_grad():
                            new_block.ln_1[0].load_state_dict(anchor.ln_1[old_loop_idx].state_dict())
                            new_block.ln_2[0].load_state_dict(anchor.ln_2[old_loop_idx].state_dict())

                    h[phys_idx] = new_block
            logical_pos += 1
        else:
            logical_pos += loop_count

    model.segments = target_segments
    rebuild_aliases(model)


def restore_segments(model: OrigamiTransformer, segments):
    """Restore segment layout without creating new blocks. Use when resuming."""
    model.segments = [tuple(s) for s in segments]
    rebuild_aliases(model)


def _fuse_on_fold(model, target_layout, optimizer):
    """Detect newly-folded segments and fuse their blocks before aliasing."""
    n = model.config.n_layer
    current_segments = model.segments
    current_map = []
    for phys_idx, loop_count in current_segments:
        for _ in range(loop_count):
            current_map.append(phys_idx)

    target_spec = parse_layout(target_layout, n)
    logical_pos = 0
    for loop_count in target_spec:
        if loop_count > 1:
            phys_in_range = set()
            for i in range(logical_pos, logical_pos + loop_count):
                if i < len(current_map):
                    phys_in_range.add(current_map[i])
            if len(phys_in_range) > 1:
                indices = sorted(phys_in_range)
                print(f"  Smart-fusing blocks {indices} → {indices[0]}...")
                fuse_blocks_smart(model, indices, optimizer)
        logical_pos += loop_count


def fuse_blocks_smart(model: OrigamiTransformer, indices: list[int], optimizer: torch.optim.Optimizer):
    """
    Fuse multiple blocks into the first block using precision-weighted averaging.
    Writes fused weights into indices[0] block.
    
    Precision is derived from AdamW's exp_avg_sq (lower variance = higher confidence).
    Parameters that only exist in the anchor (e.g. loop-specific LNs for loop_idx>0)
    are left untouched.
    """
    if len(indices) <= 1:
        return
    
    blocks = model.transformer.h
    anchor = blocks[indices[0]]
    anchor_params = dict(anchor.named_parameters())

    # Build a quick lookup from param id to its optimizer state
    param_to_state = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                param_to_state[id(p)] = optimizer.state[p]

    with torch.no_grad():
        for name, anchor_param in anchor_params.items():
            fused = torch.zeros_like(anchor_param)
            total_prec = torch.zeros_like(anchor_param)
            has_other = False

            for idx in indices:
                block = blocks[idx]
                block_params = dict(block.named_parameters())
                if name not in block_params:
                    # This param only exists in the anchor (loop-specific LN for loop_idx>0)
                    continue
                param = block_params[name]

                # Get precision from exp_avg_sq (AdamW second moment)
                state = param_to_state.get(id(param), {})
                if 'exp_avg_sq' in state:
                    exp_avg_sq = state['exp_avg_sq']
                    precision = 1.0 / (exp_avg_sq + 1e-8)
                else:
                    precision = torch.ones_like(param)

                fused += precision * param
                total_prec += precision
                has_other = True

            if not has_other:
                continue  # keep anchor's value

            fused = fused / (total_prec + 1e-8)
            anchor_param.copy_(fused)



# ── Backward-compat wrappers (old symmetric fold API) ─────────────────────────

def soft_tie(target: OrigamiBlock, source: OrigamiBlock):
    """Copy source weights into target."""
    target.load_state_dict(source.state_dict())


def restore_folds(model: OrigamiTransformer, bottom_loops: int, top_loops: int):
    """Restore old-style symmetric layout after loading a checkpoint."""
    n = model.config.n_layer
    h = model.transformer.h

    segments = []
    if bottom_loops > 1:
        segments.append((0, bottom_loops))
    for i in range(bottom_loops, n - top_loops):
        segments.append((i, 1))
    if top_loops > 1:
        segments.append((n - 1, top_loops))

    model.segments = segments

    # Old-style aliases: bottom folds into h[0], top folds into h[n-1]
    for i in range(1, bottom_loops):
        h[i] = h[0]
    for i in range(1, top_loops):
        h[n - 1 - i] = h[n - 1]


def apply_fold(model: OrigamiTransformer) -> int:
    """Old-style symmetric fold. Returns new bottom_loops."""
    bl = model.bottom_loops
    n = model.config.n_layer
    h = model.transformer.h
    if bl >= model.config.max_loops:
        return 0
    h[bl] = h[0]
    h[n - 1 - bl] = h[n - 1]

    segments = []
    if bl + 1 > 1:
        segments.append((0, bl + 1))
    for i in range(bl + 1, n - bl - 1):
        segments.append((i, 1))
    if bl + 1 > 1:
        segments.append((n - 1, bl + 1))
    model.segments = segments
    return bl + 1


def apply_fold_bottom(model: OrigamiTransformer) -> int:
    """Old-style bottom-only fold."""
    bl = model.bottom_loops
    n = model.config.n_layer
    h = model.transformer.h
    h[bl] = h[0]

    segments = []
    if bl + 1 > 1:
        segments.append((0, bl + 1))
    for i in range(bl + 1, n):
        segments.append((i, 1))
    model.segments = segments
    return bl + 1


def apply_fold_top(model: OrigamiTransformer) -> int:
    """Old-style top-only fold."""
    tl = model.top_loops
    n = model.config.n_layer
    h = model.transformer.h
    h[n - 1 - tl] = h[n - 1]

    segments = []
    for i in range(n - tl - 1):
        segments.append((i, 1))
    if tl + 1 > 1:
        segments.append((n - 1, tl + 1))
    model.segments = segments
    return tl + 1


def apply_unfold(model: OrigamiTransformer, bottom_target: int, top_target: int):
    """Old-style symmetric unfold."""
    n = model.config.n_layer
    h = model.transformer.h
    device = _device_of(model)

    while model.bottom_loops > bottom_target:
        pos = model.bottom_loops - 1
        new_block = OrigamiBlock(model.config, max_loops=model.config.max_loops)
        new_block.load_state_dict(h[0].state_dict())
        new_block.to(device)
        h[pos] = new_block
        # Update segments
        bl = model.bottom_loops - 1
        segments = []
        if bl > 1:
            segments.append((0, bl))
        for i in range(bl, n - model.top_loops):
            segments.append((i, 1))
        if model.top_loops > 1:
            segments.append((n - 1, model.top_loops))
        model.segments = segments

    while model.top_loops > top_target:
        pos = n - model.top_loops
        new_block = OrigamiBlock(model.config, max_loops=model.config.max_loops)
        new_block.load_state_dict(h[n - 1].state_dict())
        new_block.to(device)
        h[pos] = new_block
        # Update segments
        tl = model.top_loops - 1
        segments = []
        if model.bottom_loops > 1:
            segments.append((0, model.bottom_loops))
        for i in range(model.bottom_loops, n - tl):
            segments.append((i, 1))
        if tl > 1:
            segments.append((n - 1, tl))
        model.segments = segments
