# Quantization Evolution Strategies

> Yes you can train your quantized model even further at inference cost.

**News**
- 2026/2/5: Initial code release! 🚀 We encourage you to test it out. Our team is actively working on performance improvements and expanding support for additional tasks, models, and configurations.
- 2026/2/4: We released the first version of QES: https://arxiv.org/abs/2602.03120 (First version of code will be released tomorrow)


## Run the code
You can use the `int4_perturb.py` for INT4/INT8 model training, the `int4_baseline_quzo.py` for QuZO baseline, and the `wa8a_perturb.py` for w8a8 format.

We use `vllm=0.11.0` and you will need `gptqmodel` to support vLLM inference with quantized models.

You can use the `run*.sh` to replicate the experiment for int4, int8 and W8A8.

The codes are tested under:
```
python=3.11
gptqmodel==5.6.12
vllm==0.11.0
```

We use the following hyperparameters:
| Implementation | Model | Quant | Sigma (σ) | Alpha (α) |
| --- | --- | --- | --- | --- |
| Seed Replay | 1.5B | INT4 | 0.01 | 0.0005 |
| Seed Replay | 3B | INT4 | 0.005 | 0.0003 |
| Seed Replay | 1.5B | INT8 | 0.001 | 0.0001 |
| Seed Replay | 3B | INT8 | 0.001 | 0.0001 |
| Seed Replay | 1.5B | W8A8 | 0.01 | 0.001 |
| Seed Replay | 3B | W8A8 | 0.01 | 0.001 |
| Full Residual | 1.5B | INT4 | 0.01 | 0.0005 |
| Full Residual | 3B | INT4 | 0.005 | 0.0003 |
| Full Residual | 1.5B | INT8 | 0.001 | 0.0001 |
| Full Residual | 3B | INT8 | 0.001 | 0.0001 |
| Full Residual | 1.5B | W8A8 | 0.01 | 0.001 |
| Full Residual | 3B | W8A8 | 0.01 | 0.001 |