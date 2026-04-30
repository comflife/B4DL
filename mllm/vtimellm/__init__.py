import os as _os
# torch 1.9 + setuptools>=60 compat: must run before any transformers/accelerate import.
_os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
import distutils.version  # noqa: F401  -- force submodule registration

import torch as _torch
if not hasattr(_torch.cuda, "is_bf16_supported"):
    _torch.cuda.is_bf16_supported = lambda *a, **k: (
        _torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 8)
if not hasattr(_torch.backends, "mps"):
    class _MpsShim:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def is_built(): return False
    _torch.backends.mps = _MpsShim()

from .model import VTimeLLMLlamaForCausalLM
