"""Contracts deliberately do not require decoder-only KV, RoPE, or a shared encoder/reader."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    # May be layer KV, encoder hidden states, learned latent slots, or a custom object.
    payload: Any
    token_count: int
    metadata: dict = field(default_factory=dict)


class Architecture(ABC):
    capabilities = frozenset()

    @abstractmethod
    def load(self, path, dtype):
        """Return a model and processor with native Swift metadata attached."""

    @abstractmethod
    def encode(self, model, tokens, **kwargs) -> Memory:
        """Build one artifact's representation. Algorithm owns gradient/freeze policy."""

    def encode_many(self, model, padded_tokens):
        """Default ragged implementation; architectures can batch the encoder efficiently."""
        return [self.encode(model, tokens[tokens != -100][None]) for tokens in padded_tokens]

    @abstractmethod
    def assemble(self, model, memories, batch) -> Memory:
        """Map offline representations to the reader's conditioning format."""

    @abstractmethod
    def read(self, model, batch, memory):
        """Return suffix-aligned logits. May call a decoder with encoder memory."""

    @abstractmethod
    def full_reference(self, model, batch):
        """Return full-context teacher logits aligned to the same suffix tokens."""

    def response_logits(self, output, batch):
        """Override for encoder/decoder target alignment; default is causal-LM shift."""
        from .objectives import answer_logits
        return answer_logits(output.logits, batch['labels'])

    def configure_query_adapter(self, model, rank, alpha):
        raise NotImplementedError('query_lora capability needs a query-adapter implementation')

    def offline_context(self, model):
        """Context manager: frozen, deterministic teacher/cache encoder without reader adapters."""
        raise NotImplementedError

    def snapshot_memory(self, memory):
        """Return immutable CPU snapshot and its byte size; custom memory layouts override this."""
        raise NotImplementedError

    def materialize_memory(self, memory, device):
        """Return reader-owned conditioning without mutating persistent storage."""
        raise NotImplementedError

    def generate_teacher(self, model, processor, record, max_new_tokens):
        """Return response token IDs; encoder/decoder plugins define their own prompt mapping."""
        raise NotImplementedError

    def prepare_c2kv(self):
        """Select the model implementation needed by the C2KV capability before load()."""
        raise NotImplementedError

    def configure_c2kv(self, model, options):
        """Architecture-specific extractor/QKV parameter policy."""
        raise NotImplementedError

    def context_forward(self, model, inputs):
        """Optional original C2KV combined forward for upstream recipe compatibility."""
        raise NotImplementedError('c2kv_context capability is required')


class Algorithm(ABC):
    required_capabilities = frozenset()
    objective_name = 'custom'
    adapter_checkpoint = False
    accepts_loss_kwargs = False
    preserve_explicit_checkpointing = False

    def __init__(self, architecture, options):
        self.architecture = architecture
        self.options = options
        missing = self.required_capabilities - architecture.capabilities
        if missing:
            raise ValueError(f'Architecture lacks required capabilities: {sorted(missing)}')

    def collator(self):
        from .data import SegmentCollator
        return SegmentCollator()

    @abstractmethod
    def configure(self, model):
        """Select trainable parameters; optionally return a PEFT-wrapped model."""

    @abstractmethod
    def loss(self, model, batch):
        """Return (scalar loss, model outputs). Teacher/cache policy belongs here."""

    def statistics(self):
        return {}

    def trainer_loss(self, model, batch, num_items_in_batch=None):
        return self.loss(model, batch)

    def validate_records(self, train, evaluation):
        """Algorithm-specific supervision contract, independently of the CLI."""

    def load_adapter(self, model, path):
        """Override for non-PEFT adaptation/checkpoint formats."""
        from peft import PeftModel
        return PeftModel.from_pretrained(model, str(path))
