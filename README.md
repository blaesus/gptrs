gptrs
=====

Implementation of GPT-2 in Rust from scratch, only using Rust primitives.
For experimentation and education purposes.
Roughly follows Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY).

Progress
---------
- [x] bbpe tokenizer
- [x] Neural network
  - [x] Fully connected
      - [x] forward
      - [x] backward
  - [x] ReLU
  - [x] MSE
- [x] Mini-batches
- [ ] Embedding
- [ ] Transformer unit
- [ ] Softmax
- [ ] Parallel training with CPU
- [ ] Proper initialization

### Maybe
- [ ] Cross-entropy
- [ ] ADAM
- [ ] Autograd
- [ ] GPU
