"""Algorithm-specific extraction with a common opaque Memory boundary."""
from collections import OrderedDict


class C2KVExtractor:
    """Never reuse differentiable memories across optimizer steps."""

    def extract(self, model, architecture, batch):
        memories = architecture.encode_many(model, batch['context_input_ids'][0])
        return architecture.assemble(model, memories, batch)


class FrozenArtifactExtractor:
    """Bounded CPU LRU scoped to one immutable base-model instance and dtype.

    Keys include actual token IDs and cache-building prefix, not example IDs.
    The algorithm must disable adapters and enter eval/no_grad before calling.
    """

    def __init__(self, max_bytes=64 * 1024 * 1024):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError('Cache capacity must be a nonnegative integer number of bytes')
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.owner = None

    def extract(self, model, architecture, batch):
        if self.owner is not None and self.owner is not model:
            raise ValueError('A frozen cache cannot be shared between model instances')
        self.owner = model
        memories = []
        prefix = batch['cache_prefix_ids']
        prefix_key = tuple(prefix[0].tolist())
        for padded in batch['context_input_ids'][0]:
            tokens = padded[padded != -100][None]
            key = (prefix_key, tuple(tokens[0].tolist()))
            if key in self.entries:
                memory, size = self.entries.pop(key)
                self.entries[key] = (memory, size)
                self.hits += 1
                # Clones protect the reusable snapshot from downstream mutation.
                memories.append(architecture.materialize_memory(memory, tokens.device))
                continue
            self.misses += 1
            memory = architecture.encode(model, tokens, prefix=prefix)
            memories.append(memory)
            if self.max_bytes == 0:
                continue
            snapshot, size = architecture.snapshot_memory(memory)
            # An uncacheable artifact must not evict useful smaller entries.
            if size > self.max_bytes:
                continue
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = (snapshot, size)
            self.bytes += size
        return memories
