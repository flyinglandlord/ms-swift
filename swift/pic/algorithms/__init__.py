from ..registry import register_algorithm
from .attuner import Attuner
from .c2kv import C2KV

register_algorithm('c2kv', C2KV)
register_algorithm('attuner', Attuner)
