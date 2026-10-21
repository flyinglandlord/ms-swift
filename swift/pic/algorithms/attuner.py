from ..cache import FrozenArtifactExtractor
from ..contracts import Algorithm
from ..objectives import response_kl


class Attuner(Algorithm):
    required_capabilities = frozenset({'frozen_artifact', 'query_lora', 'full_reference'})
    objective_name = 'forward_kl_teacher_student'
    adapter_checkpoint = True

    def __init__(self, architecture, options):
        super().__init__(architecture, options)
        self.extractor = FrozenArtifactExtractor(options.cache_bytes)

    def configure(self, model):
        return self.architecture.configure_query_adapter(model, self.options.rank, self.options.alpha)

    def validate_records(self, train, evaluation):
        if not self.options.allow_reference_responses:
            if not all(row.get('teacher_generated') is True for row in train + evaluation):
                raise ValueError('ATTUNER requires fixed teacher-generated responses; use prepare_teacher, '
                                 'or explicitly select the reference-response ablation')

    def loss(self, model, batch):
        # One frozen backbone serves both teacher and offline encoder. No teacher grads,
        # dropout, or adapters may affect stored artifact KV (including deeper layers).
        with self.architecture.offline_context(model):
            reference = self.architecture.full_reference(model, batch)
            teacher, _ = self.architecture.response_logits(reference, batch)
            teacher = teacher.detach()
            memories = self.extractor.extract(model, self.architecture, batch)
        memory = self.architecture.assemble(model, memories, batch)
        outputs = self.architecture.read(model, batch, memory)
        student, _ = self.architecture.response_logits(outputs, batch)
        loss = response_kl(student, teacher, temperature=self.options.temperature)
        outputs.loss = loss
        return loss, outputs

    def statistics(self):
        return {
            'artifact_cache_hits': self.extractor.hits,
            'artifact_cache_misses': self.extractor.misses,
            'artifact_cache_bytes': self.extractor.bytes
        }
