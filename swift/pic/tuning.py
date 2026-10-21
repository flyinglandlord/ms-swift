import re
from peft import LoraConfig, get_peft_model


def freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def lora(model, targets, rank, alpha):
    if rank <= 0:
        raise ValueError('LoRA rank must be positive')
    freeze(model)
    pattern = r'.*\.(?:' + '|'.join(map(re.escape, targets)) + r')$'
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=pattern, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM'))
    return model
