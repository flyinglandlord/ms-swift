import torch
import torch.nn.functional as F


def answer_logits(logits, labels):
    mask = labels[:, 1:] != -100
    if not mask.any():
        raise ValueError('No supervised answer tokens')
    return logits[:, :-1][mask], labels[:, 1:][mask]


def answer_ce(logits, labels):
    scores, targets = answer_logits(logits, labels)
    return F.cross_entropy(scores.float(), targets)


def response_kl(student, teacher, labels=None, temperature=1.0, chunk_size=16):
    """Forward KL(teacher || student), Eq. 5; answer-only causal shift, FP32 softmax."""
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if labels is not None:
        student, _ = answer_logits(student, labels)
        teacher, _ = answer_logits(teacher, labels)
    teacher = teacher.detach()
    if student.shape != teacher.shape:
        raise ValueError('Teacher/student response alignment mismatch')
    total = student.new_zeros((), dtype=torch.float32)
    for s, t in zip(student.split(chunk_size), teacher.split(chunk_size)):
        target = F.log_softmax(t.float() / temperature, dim=-1)
        predicted = F.log_softmax(s.float() / temperature, dim=-1)
        total = total + F.kl_div(predicted, target, log_target=True, reduction='sum')
    return total * temperature**2 / student.shape[0]
