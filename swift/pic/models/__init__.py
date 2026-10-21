from functools import partial

from ..registry import register_architecture
from .decoder import DecoderArchitecture

for _name, _prefix, _module, _template in [
    ('qwen3', 'Qwen3', 'qwen3', 'qwen3_nothinking'),
    ('qwen2', 'Qwen2', 'qwen2_5', 'qwen2_5'),
    ('llama', 'Llama', 'llama', 'llama3'),
]:
    register_architecture(_name, partial(DecoderArchitecture, _name, _prefix, _module, _template))
