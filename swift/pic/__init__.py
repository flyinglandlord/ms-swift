"""Experimental, architecture-independent position-independent-cache training."""
from .registry import register_algorithm, register_architecture

__all__ = ['register_algorithm', 'register_architecture']
