"""
Relational Gradient - Collective Optimization Beyond Adam

版本：v0.7
作者：Pi Xi (虾皮)
机构：OpenClaw AI Lab
"""

from .optimizer import RelationalGradient
from .sparse import RelationalGradientSparse

__version__ = '0.7.0'
__author__ = 'Pi Xi'

__all__ = ['RelationalGradient', 'RelationalGradientSparse']
