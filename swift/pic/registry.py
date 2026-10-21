"""Two independent extension axes. Plugins register factories, not model-name switches."""
ARCHITECTURES = {}
ALGORITHMS = {}


def _register(registry, name, factory):
    if name in registry:
        raise ValueError(f'Duplicate PIC registration: {name}')
    registry[name] = factory


def register_architecture(name, factory):
    _register(ARCHITECTURES, name, factory)


def register_algorithm(name, factory):
    _register(ALGORITHMS, name, factory)


def register_builtins():
    from . import algorithms, models  # noqa: F401
