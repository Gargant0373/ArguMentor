from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("finetune-regression.py")
_SPEC = spec_from_file_location("src.finetune_regression_impl", _MODULE_PATH)

if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load regression finetuning module from {_MODULE_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

FinetuneConfig = _MODULE.FinetuneConfig
FinetunePipeline = _MODULE.FinetunePipeline
run_finetune = _MODULE.run_finetune