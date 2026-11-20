"""Which inputs drive the model's scores.

The window models read 1,024 harnet10 features per window, which mean nothing on their own, so SHAP
on them would say little. What a clinician can act on is which kind of input matters: the movement,
the doses and check-ins, the patient fields. So the units explained are the model's input groups
(core/data.py GROUPS, the ones in model.needs):

    windows    movement        each worn window's signal
    events     doses and check-ins
    patients   patient fields  more affected side, DBS
    features   hourly features the 7 hourly features (hourly baselines)

With so few groups the Shapley values are computed exactly, not sampled: the model predicts every
hour once for each combination of groups kept, with the others left out, and each group's value is
its average contribution over all those combinations (the definition SHAP approximates). Leaving a
group out replaces it with an uninformative version, keeping everything else:

    windows    each patient's worn windows' signals shuffled among themselves (same movement, no timing)
    events     no events at all
    patients   each patient given another explained patient's fields
    features   each patient's hours shuffled among themselves

Shuffles are repeated (explain.repeats) and averaged. What is explained is a probability per hour,
such as P(OFF) of state_now (explain.outputs). The values add up: for every hour, the groups'
Shapley values sum to P with every group minus P with none.

Also reported, from the same predictions: the ablation, how each output's AUC changes when one group
is left out (leaving the rest in), over the hours with an answer and a worn window.

The final saved models are explained on explain.patients: development patients (what each model
relies on, on the patients it learned from) or test patients (only once they've been scored). At most
explain.max_participants patients, one at a time, so memory stays at one patient's windows.

Writes artifacts/explain/<candidate>/shap.json and plots, and artifacts/explain/explain_report.md.
python -m model_pipeline.steps.explain [--config config/model.yaml]
"""

import argparse
import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from model_pipeline.core.data import Windows
from model_pipeline.core.errors import ConfigError, SchemaError
from model_pipeline.core.metrics import QUESTIONS, auc, auc_within
from model_pipeline.core.targets import HOUR_INDEX, LABEL_NAMES, PARTICIPANT, get_target, parse_label
from model_pipeline.experiments import rows, slug
from model_pipeline.steps.bias import number, slice_values
from model_pipeline.utils.file_io import read_json, write_json
from model_pipeline.utils.logging_setup import get_logger, log_step

logger = get_logger(__name__)

GROUP_NAMES = {"windows": "movement", "events": "doses and check-ins", "patients": "patient fields",
               "features": "hourly features"}


# ---- settings -----------------------------------------------------------------------------------

def settings(cfg):
    """cfg["explain"], checked, with defaults filled in.

    Raises:
        ConfigError: A missing or unusable value.
    """
    section = cfg.get("explain")
    if not isinstance(section, dict):
        raise ConfigError("config needs an explain section: patients, outputs, ...")
    patients = section.get("patients", "development")
    if patients not in ("development", "test"):
        raise ConfigError(f"explain.patients must be development or test, not {patients!r}")
    limit = section.get("max_participants", 8)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ConfigError(f"explain.max_participants must be a whole number of 1 or more, or null, not {limit!r}")
    outputs = []
    for i, entry in enumerate(section.get("outputs") or []):
        label, question = (entry or {}).get("label"), (entry or {}).get("question")
        if label not in LABEL_NAMES:
            raise ConfigError(f"explain.outputs[{i}]: unknown label {label!r}")
        target, _ = parse_label(label)
        if question not in QUESTIONS[target]:
            raise ConfigError(f"explain.outputs[{i}]: {label} has no question {question!r}; "
                              f"known: {sorted(QUESTIONS[target])} (quote \"off\": unquoted, YAML reads it as false)")
        outputs.append((label, question))
    if not outputs:
        raise ConfigError("explain.outputs must list at least one {label, question}")
    names = section.get("candidates") or []
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ConfigError("explain.candidates must be a list of candidate names")
    return {"patients": patients, "max_participants": limit,
            "repeats": int(number(section, "repeats", 3, low=1)), "seed": int(number(section, "seed", 0, low=0)),
            "candidates": names, "outputs": outputs}


# ---- leaving a group out ------------------------------------------------------------------------

def leave_out(data, group, rng, donors=None):
    """data with one input group replaced by an uninformative version (see the top of this file).

    Args:
        data: The Dataset (usually one patient's).
        group: One of GROUP_NAMES.
        rng: numpy Generator for the shuffles.
        donors: For patients, the patients table to draw other patients' fields from.
    """
    if group == "windows":
        shuffled = {}
        for pid in data.participants:
            w = data.windows[pid]
            signal = np.array(w.signal)
            worn = np.flatnonzero(np.asarray(w.worn))
            signal[worn] = signal[rng.permutation(worn)]
            shuffled[pid] = Windows(time_s=np.asarray(w.time_s), worn=np.asarray(w.worn), signal=signal)
        return replace(data, windows=shuffled)
    if group == "events":
        return replace(data, events=data.events.iloc[0:0].copy())
    if group == "patients":
        pool = data.patients if donors is None else donors
        fields = [c for c in pool.columns if c != PARTICIPANT]
        swapped = data.patients.copy()
        for i, pid in enumerate(swapped[PARTICIPANT]):
            others = pool[pool[PARTICIPANT] != pid]
            if len(others):
                donor = others.iloc[int(rng.integers(len(others)))]
                for f in fields:
                    swapped.iloc[i, swapped.columns.get_loc(f)] = donor[f]
        return replace(data, patients=swapped)
    if group == "features":
        features = data.features.copy()
        for pid in data.participants:
            idx = np.flatnonzero((data.hours[PARTICIPANT] == pid).to_numpy())
            features.iloc[idx] = features.iloc[rng.permutation(idx)].to_numpy()
        return replace(data, features=features)
    raise ConfigError(f"can't leave out unknown input group {group!r}; known: {sorted(GROUP_NAMES)}")


def explained_groups(model):
    """The model's input groups that can be left out, in a fixed order."""
    return [g for g in GROUP_NAMES if g in model.needs]


# ---- Shapley values -----------------------------------------------------------------------------

def coalitions(groups):
    """Every subset of groups (the groups kept), smallest first."""
    return [frozenset(c) for n in range(len(groups) + 1) for c in itertools.combinations(groups, n)]


def shapley(values, groups):
    """Exact Shapley values: {group: per-hour array}, from values {kept groups: per-hour array}.

    phi_g = sum over S without g of |S|! (n - |S| - 1)! / n! * (v(S + g) - v(S)).
    """
    n = len(groups)
    out = {}
    for g in groups:
        phi = 0.0
        for s in coalitions([x for x in groups if x != g]):
            weight = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
            phi = phi + weight * (values[s | {g}] - values[s])
        out[g] = np.asarray(phi, dtype=float)
    return out


def probability(frame, label, question):
    """P(question) per hour, from a predictions frame's class columns."""
    target, _ = parse_label(label)
    classes = get_target(target).classes
    return frame[[classes[i] for i in QUESTIONS[target][question]]].sum(axis=1).to_numpy(float)


def explain_participant(model, data, outputs, repeats, seed, donors):
    """One patient's per-hour Shapley values and the predictions behind them.

    Returns {"hours", "shap": {(label, question): {group: array}}, "values": {(label, question):
    {kept groups: array}}}: values averaged over the repeats.
    """
    groups = explained_groups(model)
    full = rows("explain", model.predict(data), data, fold="explain", seed=0)
    first = full[outputs[0][0]]
    hours = first[[PARTICIPANT, HOUR_INDEX, "worn"]].copy()
    sums = {key: {c: np.zeros(len(hours)) for c in coalitions(groups)} for key in outputs}
    everything = frozenset(groups)
    for r in range(repeats):
        rngs = {g: np.random.default_rng([seed, r, i]) for i, g in enumerate(groups)}
        replaced = {g: getattr(leave_out(data, g, rngs[g], donors), g) for g in groups}
        for kept in coalitions(groups):
            if kept == everything:
                frames = full
            else:
                masked = replace(data, **{g: replaced[g] for g in groups if g not in kept})
                frames = rows("explain", model.predict(masked), masked, fold="explain", seed=0)
            for key in outputs:
                sums[key][kept] += probability(frames[key[0]], *key)
    values = {key: {c: v / repeats for c, v in by.items()} for key, by in sums.items()}
    for key in outputs:
        hours[f"y_{key[0]}"] = full[key[0]]["y_true"].to_numpy(float)
    return {"hours": hours, "values": values, "shap": {key: shapley(v, groups) for key, v in values.items()}}


def summarise(parts, outputs, groups, slices=None):
    """Join the patients' results and summarise each output: Shapley values, ablation, per slice."""
    hours = pd.concat([p["hours"] for p in parts], ignore_index=True)
    everything = frozenset(groups)
    out = {}
    for key in outputs:
        label, question = key
        shap = {g: np.concatenate([p["shap"][key][g] for p in parts]) for g in groups}
        values = {c: np.concatenate([p["values"][key][c] for p in parts]) for c in coalitions(groups)}
        worn = hours["worn"].to_numpy(bool)
        entry = {
            "hours": int(worn.sum()),
            "base": float(values[frozenset()][worn].mean()) if worn.any() else None,
            "mean_prediction": float(values[everything][worn].mean()) if worn.any() else None,
            "groups": {g: {"name": GROUP_NAMES[g], "mean_abs_shap": _mean(np.abs(shap[g][worn])),
                           "mean_shap": _mean(shap[g][worn])} for g in groups},
        }
        y = hours[f"y_{label}"].to_numpy(float)
        answered = worn & ~np.isnan(y)
        if answered.any():
            members = QUESTIONS[parse_label(label)[0]][question]
            yes = np.isin(y[answered], members)
            people = hours[PARTICIPANT].to_numpy()[answered]

            def scores(kept):
                p = values[kept][answered]
                return {"auc": _clean(auc(yes, p)), "auc_within": _clean(auc_within(people, yes, p))}

            full = scores(everything)
            entry["ablation"] = {"all_groups": full, "without": {}}
            for g in groups:
                without = scores(everything - {g})
                entry["ablation"]["without"][g] = {
                    **without, "auc_drop": _diff(full["auc"], without["auc"]),
                    "auc_within_drop": _diff(full["auc_within"], without["auc_within"])}
        if slices is not None:
            joined = hours[[PARTICIPANT, HOUR_INDEX]].merge(slices, on=[PARTICIPANT, HOUR_INDEX], how="left")
            entry["by_slice"] = {}
            for column in [c for c in slices.columns if c not in (PARTICIPANT, HOUR_INDEX)]:
                entry["by_slice"][column] = {}
                for group_name, idx in joined.groupby(column).indices.items():
                    idx = idx[worn[idx]]
                    if len(idx):
                        entry["by_slice"][column][str(group_name)] = {
                            g: _mean(np.abs(shap[g][idx])) for g in groups}
        out[f"{label}/{question}"] = entry
    return out, hours


def _mean(a):
    return float(np.mean(a)) if len(a) else None


def _clean(v):
    return None if v is None or not math.isfinite(v) else float(v)


def _diff(a, b):
    return None if a is None or b is None else a - b


# ---- the step -----------------------------------------------------------------------------------

class ExplainModels:
    """Step G (SHAP and ablation): explain each final model by its input groups. See the top of this file."""

    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.opts = settings(cfg)
        self.out = Path(cfg["resolved_paths"]["artifacts_dir"]) / "explain"
        self.slice_columns = (cfg.get("bias") or {}).get("slice_columns") or []

    def run(self, candidates, data=None, split=None, models=None):
        """Explain the candidates in explain.candidates (all when empty).

        Args:
            candidates: What train (or bias) returned: each needs name and model_dir.
            data: A Dataset with the patients to explain on and every group the models need; None loads it.
            split: The patient split; None reads paths.split_file.
            models: {name: fitted Model}; None loads each candidate's model_dir.

        Returns:
            {candidate: summary}, as written to artifacts/explain/<candidate>/shap.json.
        """
        chosen = [c for c in candidates if not self.opts["candidates"] or c["name"] in self.opts["candidates"]]
        if not chosen:
            raise ConfigError("explain: none of explain.candidates was trained")
        split = split or _read_split(self.cfg)
        pool = split.development if self.opts["patients"] == "development" else split.test
        people = self._pick(pool)
        loaded = {c["name"]: (models or {}).get(c["name"]) or _load_model(c["model_dir"]) for c in chosen}
        if data is None:
            from model_pipeline.data.loaders import load_dataset
            groups = sorted({g for m in loaded.values() for g in m.needs} | {"patients"})
            data = load_dataset(self.cfg, groups=groups, participants=people)
        else:
            data = data.subset(people)
        split_name = "train" if self.opts["patients"] == "development" else "test"
        slices = self._slices(data)

        results = {}
        with self.tracker.run("explain", tags={"owner": "P3", "patients": self.opts["patients"]}):
            self.tracker.log_params({"explain": {**self.opts, "participants": len(people)}})
            for c in chosen:
                name, model = c["name"], loaded[c["name"]]
                groups = explained_groups(model)
                outputs = [o for o in self.opts["outputs"] if o[0] in model.labels]
                if not outputs:
                    logger.warning("explain: %s predicts none of explain.outputs", name)
                    continue
                with self.tracker.run(f"explain-{name}", tags={"family": model.family}), \
                        log_step(logger, "explain", candidate=name):
                    parts = [explain_participant(model, data.subset([pid]), outputs, self.opts["repeats"],
                                                 self.opts["seed"], data.patients) for pid in people]
                    summary, hours = summarise(parts, outputs, groups, slices)
                    result = {"candidate": name, "family": model.family, "patients": self.opts["patients"],
                              "participants": list(people), "groups": groups, "outputs": summary}
                    if "patients" in groups and len(people) < 2:
                        result["warning"] = "one patient only: the patient fields couldn't be left out"
                    folder = self.out / name
                    write_json(result, folder / "shap.json")
                    plot_candidate(result, folder)
                    self.tracker.log_metrics(_tracker_metrics(summary, split_name))
                    self.tracker.log_artifact(folder, folder=f"explain/{name}")
                results[name] = result
            report = self.out / "explain_report.md"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(render_report(results, self.opts), encoding="utf-8")
            self.tracker.log_artifact(report, folder="explain")
        return results

    def _pick(self, pool):
        people = sorted(pool)
        if not people:
            raise SchemaError(f"explain: no {self.opts['patients']} patients in the split")
        limit = self.opts["max_participants"]
        if limit is not None and len(people) > limit:
            rng = np.random.default_rng(self.opts["seed"])
            people = sorted(rng.choice(people, size=limit, replace=False).tolist())
        return people

    def _slices(self, data):
        if not self.slice_columns:
            return None
        try:
            return slice_values(data, self.slice_columns)
        except SchemaError as e:
            logger.warning("explain: no Shapley values by slice: %s", e)
            return None


def _tracker_metrics(summary, split):
    out = {}
    for key, entry in summary.items():
        label, question = key.split("/")
        for g, v in entry["groups"].items():
            if v["mean_abs_shap"] is not None:
                out[f"{split}/{label}/shap_{slug(question)}_{g}"] = v["mean_abs_shap"]
        for g, v in (entry.get("ablation") or {}).get("without", {}).items():
            if v["auc_drop"] is not None:
                out[f"{split}/{label}/ablation_{slug(question)}_auc_drop_{g}"] = v["auc_drop"]
    return out


def _load_model(folder):
    from model_pipeline.models.registry import load
    return load(folder)


def _read_split(cfg):
    from model_pipeline.data.folds import read_split
    return read_split(cfg)


def _fmt(v, sign=False):
    if v is None:
        return "n/a"
    return f"{v:+.3f}" if sign else f"{v:.3f}"


def render_report(results, opts):
    """explain_report.md: per candidate and output, the groups' Shapley values and ablation."""
    lines = ["# What drives the models", "",
             f"Exact Shapley values over each model's input groups, on {opts['patients']} patients "
             f"(at most {opts['max_participants']}), each left-out group shuffled {opts['repeats']} times. "
             "Mean |SHAP| is how much a group moves the probability per hour, on average. The ablation is "
             "the AUC with every group, minus the AUC with that group left out: positive means the group helps.", ""]
    for name, r in results.items():
        lines += [f"## {name}", ""]
        if r.get("warning"):
            lines += [f"Note: {r['warning']}.", ""]
        for key, entry in r["outputs"].items():
            lines += [f"### {name}: P({key.split('/')[1]}) of {key.split('/')[0]}", "",
                      f"Over {entry['hours']} worn hours: mean probability {_fmt(entry['mean_prediction'])}, "
                      f"{_fmt(entry['base'])} with every group left out.", "",
                      "| Input group | Mean abs SHAP | Mean SHAP | AUC drop without it | Within-patient AUC drop |",
                      "| --- | --- | --- | --- | --- |"]
            without = (entry.get("ablation") or {}).get("without", {})
            for g, v in entry["groups"].items():
                w = without.get(g, {})
                lines.append(f"| {v['name']} | {_fmt(v['mean_abs_shap'])} | {_fmt(v['mean_shap'], True)} | "
                             f"{_fmt(w.get('auc_drop'), True)} | {_fmt(w.get('auc_within_drop'), True)} |")
            lines.append("")
    lines += ["## How to read this", "",
              "- These describe what each model relies on; on development patients the model has seen them, so "
              "the ablation's AUCs are higher than on new patients. Read the drops as relative, not as "
              "performance.",
              "- Leaving movement out shuffles each patient's windows in time: it removes when movement "
              "happened, not what kind of movement the patient has.",
              "- A group the model doesn't read isn't listed.", ""]
    return "\n".join(lines)


def plot_candidate(result, folder):
    """shap.png (mean |SHAP| per group, one panel per output) and ablation.png (AUC drop per group)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = result["outputs"]
    names = [GROUP_NAMES[g] for g in result["groups"]]
    for kind, title in (("shap", "mean |SHAP| (probability)"), ("ablation", "AUC drop without the group")):
        fig, axes = plt.subplots(1, len(outputs), figsize=(3.4 * len(outputs), 3.2), squeeze=False)
        for ax, (key, entry) in zip(axes[0], outputs.items()):
            if kind == "shap":
                values = [entry["groups"][g]["mean_abs_shap"] or 0 for g in result["groups"]]
            else:
                without = (entry.get("ablation") or {}).get("without", {})
                values = [(without.get(g) or {}).get("auc_drop") or 0 for g in result["groups"]]
            ax.barh(names, values, color="tab:blue" if kind == "shap" else "tab:orange")
            ax.axvline(0, color="black", linewidth=0.6)
            ax.set_title(f"P({key.split('/')[1]}), {key.split('/')[0]}", fontsize=9)
            ax.tick_params(labelsize=8)
        fig.suptitle(f"{result['candidate']}: {title}", fontsize=9)
        fig.tight_layout()
        fig.savefig(folder / f"{kind}.png", dpi=110)
        plt.close(fig)


def main(argv=None):
    from model_pipeline.tracking.mlflow_tracker import MlflowTracker
    from model_pipeline.utils.config import load_config
    from model_pipeline.utils.logging_setup import configure_logging

    parser = argparse.ArgumentParser(
        description="Explain the final models by input group (model_pipeline/steps/explain.py)")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg)
    artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])
    path = artifacts / "candidates.json"
    if not path.exists():
        raise SchemaError(f"no {path}: run train first")
    results = ExplainModels(cfg, MlflowTracker.from_config(cfg)).run(read_json(path))
    for name, r in results.items():
        for key, entry in r["outputs"].items():
            ranked = sorted(entry["groups"].items(), key=lambda kv: -(kv[1]["mean_abs_shap"] or 0))
            print(f"{name} {key}: " + ", ".join(f"{v['name']} {_fmt(v['mean_abs_shap'])}" for _, v in ranked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
