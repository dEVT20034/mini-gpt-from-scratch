# mini-gpt-from-scratch
A from-scratch GPT (decoder-only Transformer) in PyTorch trained on TinyShakespeare. Multi-head causal attention, 10.8M params, with training logs and an overfitting analysis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12jRWKGL40IlP-Kesv2ciRU8kONJcZ5Od?usp=sharing)

## About

This is a small GPT model that I built from scratch in PyTorch to understand how LLMs actually work. It reads text one character at a time and learns to guess the next character. I trained it on TinyShakespeare, so it writes text that looks like a Shakespeare play.

Everything is in one notebook, `Mini_GPT.ipynb`, with comments on every part.

## What's inside

- Character-level tokenizer
- Token and position embeddings
- Multi-head causal self-attention
- Transformer blocks (LayerNorm, residual connections, feed-forward layer)
- Training loop with AdamW and a learning rate schedule
- Text generation with temperature and top-k sampling

## Model details

| | |
|---|---|
| Dataset | TinyShakespeare (about 1.1M characters, 65 unique characters) |
| Parameters | 10.77M |
| Layers | 6 |
| Attention heads | 6 |
| Embedding size | 384 |
| Context length | 256 characters |
| Training | 5,000 steps on a Colab GPU, about 38 minutes |

## Results

| Step | Train loss | Validation loss |
|---|---|---|
| 0 | 4.245 | 4.239 |
| 500 | 1.519 | 1.699 |
| 1000 | 1.250 | 1.518 |
| 1500 | 1.114 | **1.463** |
| 3000 | 0.678 | 1.691 |
| 5000 | 0.344 | 2.029 |

The loss starts at about 4.2, which is the same as random guessing over 65 characters. The best validation loss was 1.463.

Sample output:

```text
KING RICHARD II:
Ay, but thou wilt say to me, I say.

DUCHESS OF YORK:
Why, brother, I cannot tell it so,
I am content: but 'tis thy wisdom to come;
```

## How it works

1. **Text to numbers.** Every unique character gets a number. The whole text becomes one long list of numbers, split 90% for training and 10% for validation.
2. **Batches.** I pick random chunks of text. The answer for each chunk is the same chunk shifted by one character, so every position learns to predict the next character.
3. **Model.** Each character becomes a vector (character embedding + position embedding), goes through 6 Transformer blocks, and the last layer gives a score for each of the 65 possible next characters.
4. **Training.** The model guesses, the loss shows how wrong it was, backpropagation finds the fix, and AdamW updates the weights.
5. **Generation.** The model predicts the next character, I sample one, add it to the text, and repeat.

Attention in each head is just this formula, with a mask so a character cannot see the future:

```
softmax(Q · Kᵀ / √d) · V
```

## How to run

**On Colab (easiest):**
1. Click the Colab button at the top.
2. Go to Runtime, Change runtime type, and pick GPU.
3. Click Runtime, Run all. The dataset downloads by itself.

**On your computer:**
```bash
git clone https://github.com/<your-username>/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch
pip install torch
jupyter notebook Mini_GPT.ipynb
```

The notebook picks a smaller model automatically if you don't have a GPU.

## Try it yourself

After training, run this in a new cell:

```python
def complete(prompt, n=300, temperature=0.8, top_k=40):
    prompt = "".join(c for c in prompt if c in stoi)
    if not prompt:
        prompt = "\n"
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    out = model.generate(ids, max_new_tokens=n, temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())

print(complete("ROMEO:"))
```

Prompts that look like the training data work best (like `ROMEO:`). A lower temperature gives safer text and a higher one gives more random text.

## What I learned

- Training loss kept going down, but validation loss was lowest around step 1,500 and then started going up. This is overfitting, because the dataset is only about 1 MB and the model sees it around 80 times. Next time I will save the best checkpoint instead of the last one.
- A Transformer is mostly the same block repeated: attention lets characters talk to each other, and the feed-forward layer lets each one think on its own.
- The causal mask is what makes next-character training work, because it stops the model from looking at the answer.

## Next steps

- Save the best checkpoint and plot the loss curves
- Switch to a BPE tokenizer
- Train on a bigger dataset
- Try RoPE and RMSNorm
- Fine-tune a pretrained model with LoRA

## Credits

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his Neural Networks: Zero to Hero videos
- TinyShakespeare dataset from [char-rnn](https://github.com/karpathy/char-rnn)
