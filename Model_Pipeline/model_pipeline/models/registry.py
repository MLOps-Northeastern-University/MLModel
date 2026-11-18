"""Builds models from the config, and loads saved ones: family name to class.

Owner: P2.

Every step that makes or opens a model goes through here, so a family is named in one place:

    models = candidates(cfg)                      # [(name, unfitted Model)], from cfg["models"]
    model = load(folder)                          # any saved model: its manifest names the family

The config's models section (config/model.yaml):

    models:
      targets: [sleep, state, tremor]             # what every candidate predicts
      horizons: [now, next_hour]
      candidates:
        - name: window_boosting                   # unique; how the candidate is logged and chosen
          family: window_boosting                 # a key of FAMILIES
          params: {iterations: 200}               # optional; merged into the family's defaults

A candidate may also set its own targets and horizons. Two candidates can share a family with
different params (a variant), as long as their names differ.

Families are listed here, as fixed import paths, and imported only when asked for. So reading the
config never imports PyTorch, and no name from a config or a manifest can import anything else.
Unknown families, unknown keys and duplicate names are refused with a ConfigError before anything
trains; a saved folder of an unknown family is refused with a ModelError.
"""

import importlib
import re
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Type

from model_pipeline.core.data import WindowSpec
from model_pipeline.core.errors import ConfigError, ModelError
from model_pipeline.core.model import MANIFEST, Model
from model_pipeline.utils.file_io import read_json
from model_pipeline.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Family name -> "module:Class". A new family is one line here.
FAMILIES: Dict[str, str] = {
    "window_logistic": "model_pipeline.models.window_classifier:WindowLogistic",
    "window_boosting": "model_pipeline.models.window_classifier:WindowBoosting",
    "motor_causal_transformer": "model_pipeline.models.motor_causal_transformer.model:MotorCausalTransformer",
}

CANDIDATE_KEYS = {"name", "family", "params", "targets", "horizons"}
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def family_class(family: str) -> Type[Model]:
    """The Model subclass of a family, imported on first use.

    Raises:
        ConfigError: An unknown family.
        ModelError: The listed class isn't a Model of that family (a mistake in FAMILIES).
    """
    if family not in FAMILIES:
        raise ConfigError(f"unknown model family {family!r}; known: {sorted(FAMILIES)}")
    module_name, class_name = FAMILIES[family].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, Model)) or cls.family != family:
        raise ModelError(f"FAMILIES[{family!r}] is {FAMILIES[family]}, which isn't the {family!r} Model")
    return cls


def candidates(cfg: Mapping) -> List[Tuple[str, Model]]:
    """Every candidate in cfg["models"], built and unfitted, in config order.

    Raises:
        ConfigError: No models section or no candidates, an unknown key or family, a bad or repeated
            name, or settings the family refuses.
    """
    section = cfg.get("models")
    if not isinstance(section, Mapping):
        raise ConfigError("config needs a models section with targets, horizons and candidates")
    unknown = set(section) - {"targets", "horizons", "candidates"}
    if unknown:
        raise ConfigError(f"models has unknown key(s) {sorted(unknown)}")
    entries = section.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("models.candidates must be a list of at least one candidate")
    spec = WindowSpec.from_config(cfg) if "windows" in cfg else None   # a family that reads windows needs it
    built, names = [], set()
    for i, entry in enumerate(entries):
        name, model = build(entry, spec, section.get("targets"), section.get("horizons"), where=f"candidate {i + 1}")
        if name in names:
            raise ConfigError(f"two candidates are named {name!r}; names must be unique")
        names.add(name)
        built.append((name, model))
    logger.info("%d candidate model(s): %s", len(built), ", ".join(n for n, _ in built),
                extra={"event": "candidates", "candidates": [n for n, _ in built],
                       "families": [m.family for _, m in built]})
    return built


def build(entry: Mapping, spec: WindowSpec, targets=None, horizons=None, where="candidate") -> Tuple[str, Model]:
    """One candidate from its config entry: (name, unfitted model).

    Args:
        entry: {name, family, params?, targets?, horizons?}.
        spec: The window spec, for families that read windows.
        targets, horizons: The models section's defaults, used where the entry sets none.
    """
    if not isinstance(entry, Mapping):
        raise ConfigError(f"{where} must be a mapping with name and family, not {type(entry).__name__}")
    unknown = set(entry) - CANDIDATE_KEYS
    if unknown:
        raise ConfigError(f"{where} has unknown key(s) {sorted(unknown)}; allowed: {sorted(CANDIDATE_KEYS)}")
    name = entry.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ConfigError(f"{where}: name must be lowercase letters, digits and underscores, not {name!r}")
    where = f"candidate {name!r}"
    cls = family_class(_required(entry, "family", where))
    params = entry.get("params") or {}
    if not isinstance(params, Mapping):
        raise ConfigError(f"{where}: params must be a mapping, not {type(params).__name__}")
    targets = _names(entry.get("targets", targets), "targets", where)
    horizons = _names(entry.get("horizons", horizons), "horizons", where)
    try:
        model = cls(targets, horizons, dict(params), spec if "windows" in cls.needs else None)
    except ConfigError as e:
        raise ConfigError(f"{where}: {e}") from None
    return name, model


def load(folder) -> Model:
    """A saved model of any family, rebuilt as its manifest says.

    Raises:
        ModelError: No manifest, or a family this code doesn't know.
    """
    folder = Path(folder)
    if not (folder / MANIFEST).exists():
        raise ModelError(f"no {MANIFEST} in {folder}")
    family = read_json(folder / MANIFEST).get("family")
    if family not in FAMILIES:
        raise ModelError(f"{folder} holds a {family!r} model, which this code doesn't know; known: {sorted(FAMILIES)}")
    return family_class(family).load(folder)


def _required(entry, key, where):
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} needs a {key}")
    return value


def _names(value, key, where):
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: {key} must be a non-empty list of names (set it in models or the candidate)")
    return list(value)


__all__ = ["FAMILIES", "build", "candidates", "family_class", "load"]
