# Mica — Fine-tuning Ideas

Three directions worth pursuing after Phase 2 completes, in rough order of effort.

---

## 1. Narrow Style Specialisation

**Idea**: Fine-tune on a strict sub-corpus — ideally a single author or a single work — to sharpen from "hardboiled register" to a specific voice.

**Why it works**: Phase 2 establishes the genre. A third training pass with a lower LR and a smaller, focused corpus overwrites the blended average with something more distinctive. The model already speaks the language; this teaches it an accent.

**Practical setup**:
- Pick the sub-corpus (e.g. only Continental Op stories, ~300K tokens)
- Fine-tune from `mica_weights.pt` with max LR around `5e-5` — even lower than Phase 2
- Short run: 1000–2000 steps, eval every 100
- Watch the train/val gap closely — with a tiny corpus the model can memorise fast

**Risk**: With a few hundred thousand tokens, memorisation is real. Keep dropout at 0.2, watch for val loss stopping its descent or ticking back up, and stop there.

**Signal that it worked**: Ask it to continue a sentence. The cadence and word choices should feel markedly different from a cold Phase 2 generation.

---

## 2. Format / Form Injection

**Idea**: Teach the model to follow a prompt rather than just continue text freely. Construct a dataset of `[seed sentence] → [paragraph continuation]` pairs from the existing corpus and fine-tune on that format.

**Dataset construction** (no manual labelling needed):
```python
# From corpus text, split into overlapping (prompt, continuation) pairs:
# - prompt: first 1–2 sentences of a paragraph
# - continuation: the rest of the paragraph
# Wrap with a separator token so the model learns the format boundary
```

A separator like `\n###\n` works. The model sees thousands of examples of "short seed → developed prose" and learns to treat the `###` as a mode switch.

**Fine-tuning**:
- Generate the pairs programmatically from `corpus/corpus.txt`
- Only compute loss on the *continuation* side — mask the prompt tokens so the model isn't rewarded for predicting the seed
- LR around `3e-5`, 2000–3000 steps

**At inference**: prepend your seed, add `###`, let the model complete. Works best when the seed is in the style of the training corpus — a detective opening line, a scene-setting sentence.

**What masking the prompt looks like in code**:
```python
# In the training loop, replace prompt token targets with -1
# F.cross_entropy ignores index -1 by default
targets[:, :prompt_len] = -1
```

---

## 3. DPO on Personal Taste

**The most interesting one.** Direct Preference Optimisation lets you encode your own aesthetic judgment — what sounds like authentic noir, what sounds flat — directly into the weights, with no reward model, no RL infrastructure, and a dataset that can be as small as 50 examples.

### What DPO does

Standard fine-tuning moves the model toward text that appears in the training data. DPO instead moves it toward text *you prefer* over text *you don't*, using paired comparisons.

You provide triplets: `(prompt, chosen completion, rejected completion)`. DPO adjusts the weights so the model assigns higher probability to chosen and lower to rejected, *relative to a frozen reference copy of the model*. The reference copy acts as a regulariser — it prevents the model from drifting so far toward your preferences that it forgets how to generate coherent text.

### The loss (one equation, then we're done with theory)

```
L = -log σ( β · (log π(chosen|prompt) - log π_ref(chosen|prompt))
              - β · (log π(rejected|prompt) - log π_ref(rejected|prompt)) )
```

`β` (typically 0.1–0.3) controls how hard the model is pushed. High β = aggressive preference learning but risks instability. Low β = gentle nudge.

`π` is the model being trained. `π_ref` is a frozen copy of `mica_weights.pt`. Both run on every batch; only `π` gets gradients.

The log probabilities are computed as the *sum of per-token log-softmax values over the completion* (not the prompt). So the model is scored on how well it predicts its own preference completions.

### Building the dataset

You need 50–100 triplets. Three sources, combinable:

**Source A — generate and judge**
Run `generate.py` with different temperatures on the same 20–30 prompts. For each prompt, keep the best generation as `chosen` and the worst as `rejected`. Takes an evening but produces the most authentic signal.

**Source B — corpus vs. model**
Use actual corpus sentences as `chosen` and model-generated continuations of the same prompt as `rejected`. This is a blunter instrument (you're just pushing toward the training data again) but requires zero manual effort.

**Source C — adversarial pairs**
Write `rejected` completions yourself: deliberately flat prose, wrong-era vocabulary, over-explained emotion — the things that break the noir register. Pair them against corpus originals. This is the most surgical and likely the most effective.

**Source C² — interactive ranking via the web UI**
A more ergonomic version of C that requires no writing. Extend the web interface with a *rank mode*: given a prompt, generate N completions in parallel (say 5), display them side by side, click to mark the best and worst, accumulate pairs in browser storage, export to `dpo_pairs.json` when done. An afternoon of clicking produces 50–100 ranked pairs without touching a text editor.

The additions to `llm/` needed for this:

- **Rank mode toggle** in `index.html` — switches the output area from a single streamed generation to a grid of N completed outputs
- **`rank.js`** — generates N completions concurrently (Promise.all over N calls to the generation loop, each with a different random seed), renders them, handles click-to-rank
- **`pairs.js`** — accumulates `{prompt, chosen, rejected}` objects in `localStorage`, renders a counter ("12 pairs collected"), provides an Export button that downloads `dpo_pairs.json`
- **Temperature spread** — optionally generate the N completions at slightly varied temperatures (e.g. 0.7, 0.8, 0.9, 1.0, 1.1) to get more diverse candidates per prompt

The interaction loop:

```
Type a prompt → Generate 5 → Read them → Click ★ on the best → Click ✗ on the worst
→ Pair saved → New prompt (or same prompt again) → repeat
```

The chosen/rejected pair doesn't have to come from the same generation batch — you can star one from batch 1 and cross one from batch 2 on the same prompt, and the UI tracks it. This naturally accumulates the most diverse rejected examples, which is what makes DPO signal strong.

A mix of A and C² gives the richest signal.

**Format**:
```json
[
  {
    "prompt": "The rain hadn't stopped in three days.",
    "chosen": "I lit a cigarette and watched the street from the window. Nobody moved. Nobody ever moved on a street like that after midnight.",
    "rejected": "I felt very sad and tired because of all the rain. The weather was making me feel depressed and I didn't know what to do next."
  },
  ...
]
```

### Implementation sketch

```python
# dpo_train.py
import torch, json, copy
from torch.nn import functional as F
from config import MicaConfig
from model import MicaTransformer
from tokenizers import Tokenizer

β = 0.1
max_steps = 500
lr = 1e-5

config = MicaConfig()
tokenizer = Tokenizer.from_file("mica_tokenizer.json")

# Active model (gets gradients)
model = MicaTransformer(config).to(config.device)
model.load_state_dict(torch.load("mica_weights.pt", map_location=config.device))

# Frozen reference (no gradients, ever)
ref = copy.deepcopy(model)
for p in ref.parameters():
    p.requires_grad_(False)
ref.eval()

dataset = json.load(open("dpo_pairs.json"))
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def log_prob(m, prompt_ids, completion_ids):
    """Sum of log P(token_t | token_<t) over completion tokens only."""
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=config.device)
    targets = ids.clone()
    targets[0, :len(prompt_ids)] = -1          # mask prompt
    logits, loss = m(ids, targets)
    # loss is mean; recover sum for comparability across lengths
    return -loss * len(completion_ids)


model.train()
for step in range(max_steps):
    pair = dataset[step % len(dataset)]
    
    prompt_ids     = tokenizer.encode(pair["prompt"]).ids
    chosen_ids     = tokenizer.encode(pair["chosen"]).ids
    rejected_ids   = tokenizer.encode(pair["rejected"]).ids

    log_pi_chosen   = log_prob(model, prompt_ids, chosen_ids)
    log_pi_rejected = log_prob(model, prompt_ids, rejected_ids)

    with torch.no_grad():
        log_ref_chosen   = log_prob(ref, prompt_ids, chosen_ids)
        log_ref_rejected = log_prob(ref, prompt_ids, rejected_ids)

    ratio = (log_pi_chosen - log_ref_chosen) - (log_pi_rejected - log_ref_rejected)
    loss = -F.logsigmoid(β * ratio)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:>4}: loss {loss.item():.4f}  ratio {ratio.item():.4f}")

torch.save(model.state_dict(), "mica_dpo.pt")
```

### What to watch

- **`ratio`** should trend positive — meaning the model increasingly prefers chosen over rejected relative to the reference. If it goes negative and stays there, lower β.
- **KL divergence from reference**: not shown above but worth monitoring. If the model drifts too far, generations will degrade. The β parameter controls this implicitly.
- **Generations before and after**: the real test. Run `generate.py` (pointed at `mica_dpo.pt`) on 10 prompts and compare against Phase 2 output. Trust your ear over the numbers.

### What to expect at this scale

DPO at 2.7M params on 50–100 examples is genuinely uncharted. The model is small enough that even a light push should produce measurable changes in output character. The risk is over-optimisation: with so few examples, 500 steps may be too many. Try 100–200 first, generate a sample, and judge by ear before continuing.

The most likely outcome — if the pairs are good — is that the model's output sounds *less averaged*, less like a blend of everything in the corpus, and more like the specific quality you were selecting for. Whether that's tighter sentence rhythm, more economical description, better dialogue punctuation — it'll be whatever your `chosen` examples had in common.

---

## Sequencing

If doing all three, the natural order is:

1. **Format injection** — establishes the prompt→completion contract
2. **Narrow specialisation** — tightens the voice within that contract  
3. **DPO** — final aesthetic polish, starting from the specialised weights

Each step uses the previous step's output as its starting point and a learning rate an order of magnitude lower than the step before.
