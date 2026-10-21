# PIC training / PIC 训练

Research extension based on ms-swift v4.6.0 (`ad818a2cdfafaf4305ac40c4acd892a211978bc2`).
本分支基于上述原始版本，仅包含核心实现和使用说明，不包含实验产物或对齐测试套件。
Original Swift commands are unchanged; PIC uses separate `swift pic` and `megatron pic` entry points.
原生 Swift 命令保持不变；PIC 尚未接入 `swift sft` 的统一数据与模板流程。

## Install and train / 安装与训练

Install an accelerator-compatible PyTorch build first, then from this checkout:
先安装与设备匹配的 PyTorch，再从当前仓库安装：

```bash
pip install -e .
pip install -r requirements/pic.txt
swift pic --help
megatron pic --help
```

HF dependencies are pinned in `requirements/pic.txt`; Megatron additionally needs
Swift's compatible Megatron/MCore-Bridge/Transformer Engine environment.
Megatron 还需要匹配的 Megatron、MCore-Bridge 和 Transformer Engine 环境。

The segmented JSONL schema contains token IDs from the selected model's tokenizer:
分段 JSONL 使用所选模型 tokenizer 的 token ID；训练和验证 ID/trace_id 必须互不重叠。

```json
{"id":"train-1","segments":[[10,11],[12,13]],"query_ids":[14],"answer_ids":[15,2],"system_ids":[],"cache_prefix_ids":[]}
```

```bash
swift pic --algorithm c2kv --architecture qwen3 \
  --model /path/to/model --train-data /path/to/train.jsonl \
  --eval-data /path/to/validation.jsonl --output outputs/pic \
  --dtype bf16 --steps 100 --learning-rate 2e-5
```

C2KV defaults to strict 1:1 segmented reuse. `--algorithm-config` selects additional
recipes/configuration; `--tuning lora --lora-scope gist --rank 8 --alpha 16` enables
gist-projection LoRA. See the argument parser and algorithm implementation for options.
C2KV 默认严格 1:1；其他 recipe 由算法配置选择，LoRA 可用于 gist projection。

ATTUNER uses `--algorithm attuner`, frozen independently encoded memory, query-only
LoRA and teacher-to-student KL. Prepare fixed teacher responses with
`python -m swift.pic.prepare_teacher --help`; reference-label responses are an
explicit ablation, not the default recipe. `--cache-bytes 0` disables snapshot storage.
ATTUNER 使用冻结 memory、query LoRA 和教师 KL；需先准备教师回答，不默认以参考答案替代。
缓存容量为 0 时不生成 CPU 快照；超过容量的单条 memory 不会清空已有缓存。

For Megatron, use `megatron pic --help` for `pic_*` arguments and Swift's native
training arguments. It uses native MCore checkpoints, not HF export.
Megatron 通过原生训练器执行，使用 MCore checkpoint；HF gist 初始化导入受限于新建 C2KV 训练。

## Modules and extension points / 模块与扩展

| Module | Responsibility / 职责 |
|---|---|
| `swift/pic/contracts.py`, `registry.py` | Architecture/algorithm/memory contracts and plugin registration / 扩展契约与注册 |
| `swift/pic/models/` | Decoder adapters and C2KV model kernels / 模型适配 |
| `swift/pic/algorithms/` | C2KV recipes, ATTUNER objectives and parameter selection / 算法、监督与参数选择 |
| `swift/pic/cache.py`, `data.py`, `objectives.py`, `tuning.py` | KV extraction, data, loss and tuning / KV、数据、损失与微调 |
| `swift/pic/train.py`, `trainer.py` | Swift HF trainer integration / HF 训练入口 |
| `swift/pic/megatron/` | Native Megatron trainer, TP kernels and checkpoint conversion / Megatron 后端 |

Register new HF algorithms or architectures using `register_algorithm` /
`register_architecture`, import via `--plugin`, and supply JSON configuration via
`--algorithm-config` / `--architecture-config`. Implement memory encoding, assembly
and reading through `Architecture`; memory need not be a decoder KV tuple.
HF 新算法/架构通过注册与 plugin 导入，encoder/decoder 可实现独立 memory 契约。
Megatron requires a separate runtime implementation, TP layout and native checkpoint
handling; HF plugin registration does not automatically implement a distributed backend.
Megatron 扩展还需单独实现 runtime、TP 分片及 checkpoint，并验证梯度与重载。

## Boundaries / 边界

- HF entry: single process, microbatch 1; no validated DDP/DeepSpeed/packing path.
- Megatron: dense Qwen-family/Llama adapters, strict 1:1, microbatch 1, DP/PP/CP=1;
  no SP/packing/recompute, MoE, sliding attention or FP8/FP4.
- BF16 cross-backend numerical parity is unresolved; the latest input RMSNorm
  correction still needs GPU regression. This branch does not claim benchmark parity.
- Encoder/decoder is an extension interface, not an implemented production model.
- 本分支为研究实现，不代表已验证所有模型、并行配置或 BF16 数值一致性。
- Imported C2KV code retains its provenance and copyright notices; see
  `swift/pic/models/c2kv/PROVENANCE.md`. Private use does not establish redistribution rights.
  导入代码许可仍需核对，不能因置于 Swift 仓库而自动按 Apache 重新授权。
