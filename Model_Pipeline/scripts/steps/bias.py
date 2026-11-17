"""Checks the model across slices of patients and mitigates bias.

For every candidate train trained, from its predictions (artifacts/predictions/<label>.csv):

    1. slice: join each hour to its patient's groups (bias.slice_columns: sex, age group, DBS, ...),
       from the hours table, or the patients table for a column the hours table doesn't have;
    2. score each group with Fairlearn's MetricFrame, using the same scores as everyone else
       (core/metrics.label_scores), plus each question's recall, precision and false-positive rate
       at bias.threshold. Each seed is scored on its own and the seeds averaged, as cv scores are;
    3. the gate (bias.checks): for each check and slice column, the gap between the best and worst
       judged group. A group is judged only with min_slice_rows hours and min_slice_participants
       patients: most slices describe the person, so a group of one patient measures that person;
    4. mitigation, for a candidate that fails: cross-validate it again with reweighted training hours
       (Kamiran and Calders' reweighing over the failing columns, computed on each fold's training
       patients only), check the result the same way, refit it on every development patient and
       save it. It becomes a new candidate, <name>_reweighted, that select can choose.

By default the out-of-fold predictions are sliced, so every development patient counts, and the test
patients stay unseen until the one run that scores them (bias.predictions: test reads those instead).

A candidate passes only when at least one check was judged and none failed: a gate that could judge
nothing (too few patients in every group) blocks, rather than passing by default.

Writes artifacts/bias/: bias.json (every group's scores, each check, the mitigation and its cost),
bias_report.md and plots/. The candidates go on in the one shared file, artifacts/candidates.json,
which train writes and each later step reads and updates. Every candidate gets:

    gates["bias"]   True or False
    gates_passed    True only when every gate in gates passed: the field select reads
    bias            a summary: what failed, and how many checks were judged

Evaluate runs first and sets gates_passed for validation. That verdict is kept as gates["validation"],
so a validation failure is never overwritten. A mitigated candidate, which evaluate never saw, is held
to the same validation_gate.min with evaluate's own gate_failures.

A failing candidate sends an alert. If no candidate passes the bias gate, even after mitigation, the
step raises GateError, which blocks the release.

    python -m scripts.steps.bias [--config config/model.yaml]
"""

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.alerts import notify
from scripts.core.errors import ConfigError, GateError, SchemaError
from scripts.core.metrics import QUESTIONS, higher_is_better, label_scores, mean_scores
from scripts.core.targets import HOUR_INDEX, LABEL_NAMES, PARTICIPANT, get_target, parse_label
from scripts.experiments import concat, fresh, named, rows, score, slug
from scripts.utils.file_io import read_json, write_json
from scripts.utils.logging_setup import get_logger, log_step

logger = get_logger(__name__)

UNKNOWN = "unknown"
TEST_FOLD = "test"
MITIGATED_SUFFIX = "_reweighted"
PREDICTION_COLUMNS = (PARTICIPANT, HOUR_INDEX, "fold", "seed", "model", "worn", "y_true")


# ---- settings -----------------------------------------------------------------------------------

def settings(cfg):
    """cfg["bias"], checked, with defaults filled in.

    Raises:
        ConfigError: A missing or unusable value.
    """
    section = cfg.get("bias")
    if not isinstance(section, dict):
        raise ConfigError("config needs a bias section: slice_columns, checks, ...")
    columns = section.get("slice_columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        raise ConfigError("bias.slice_columns must be a non-empty list of column names")
    default_gap = number(section, "max_recall_gap", 0.10, low=0)
    checks = []
    for i, check in enumerate(section.get("checks") or []):
        if not isinstance(check, dict) or set(check) - {"label", "metric", "max_gap"}:
            raise ConfigError(f"bias.checks[{i}] must be {{label, metric, max_gap?}}")
        label, metric = check.get("label"), check.get("metric")
        if label not in LABEL_NAMES:
            raise ConfigError(f"bias.checks[{i}]: unknown label {label!r}; known: {sorted(LABEL_NAMES)}")
        if not isinstance(metric, str) or not metric:
            raise ConfigError(f"bias.checks[{i}] needs a metric")
        checks.append({"label": label, "metric": metric, "max_gap": number(check, "max_gap", default_gap, low=0)})
    if not checks:
        raise ConfigError("bias.checks must list at least one {label, metric}")
    source = section.get("predictions", "cv")
    if source not in ("cv", "test"):
        raise ConfigError(f"bias.predictions must be cv or test, not {source!r}")
    mitigation = section.get("mitigation") or {}
    names = mitigation.get("candidates") or []
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ConfigError("bias.mitigation.candidates must be a list of candidate names")
    return {
        "slice_columns": columns,
        "checks": checks,
        "min_rows": int(number(section, "min_slice_rows", 20, low=1)),
        "min_participants": int(number(section, "min_slice_participants", 2, low=1)),
        "predictions": source,
        "threshold": number(section, "threshold", 0.5, low=0, high=1),
        "mitigation": {"enabled": bool(mitigation.get("enabled", True)), "candidates": names,
                       "power": number(mitigation, "power", 1.0, low=0),
                       "max_weight": number(mitigation, "max_weight", 5.0, low=1)},
    }


def number(section, key, default, low=None, high=None):
    """section[key] (or default) as a float, checked to be within low..high."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or (low is not None and value < low) or (high is not None and value > high):
        raise ConfigError(f"setting {key} must be a number from {low} to {high}, not {value!r}")
    return float(value)


# ---- slices -------------------------------------------------------------------------------------

def slice_values(data, columns):
    """One row per hour of data: participant_id, hour_index and each slice column, as text.

    A column comes from data.hours, or from data.patients (one value per patient) when the hours
    don't have it. Missing values become "unknown", so they're a visible group, never dropped.

    Raises:
        SchemaError: A column is in neither table.
    """
    hours = data.hours.reset_index(drop=True)
    out = hours[[PARTICIPANT, HOUR_INDEX]].copy()
    patients = data.patients
    for column in columns:
        if column in hours.columns:
            values = hours[column]
        elif patients is not None and column in patients.columns:
            values = out[PARTICIPANT].map(dict(zip(patients[PARTICIPANT], patients[column])))
        else:
            raise SchemaError(f"slice column {column!r} is in neither the hours nor the patients table")
        out[column] = _as_group(values)
    return out


def _as_group(values):
    """Group names as text: booleans as yes/no, whole numbers without .0, missing as unknown."""
    def name(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return UNKNOWN
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)
    return pd.Series([name(v) for v in values], index=values.index, dtype=object)


# ---- predictions --------------------------------------------------------------------------------

def read_predictions(folder, labels):
    """{label: table} from artifacts/predictions/<label>.csv, for each label that has a file.

    Raises:
        SchemaError: No predictions at all, or a file without the columns train writes.
    """
    folder = Path(folder)
    out = {}
    for label in labels:
        path = folder / f"{label}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path, low_memory=False)
        classes = list(get_target(parse_label(label)[0]).classes)
        missing = [c for c in (*PREDICTION_COLUMNS, *classes) if c not in table.columns]
        if missing:
            raise SchemaError(f"{path} has no column(s) {missing}")
        table["fold"] = table["fold"].astype(str)
        table["worn"] = table["worn"].astype(str).str.lower().isin(("true", "1"))
        out[label] = table
    if not out:
        raise SchemaError(f"no predictions in {folder}: run train first")
    return out


def scored_rows(table, source):
    """The rows the bias check scores: worn hours with an answer, out-of-fold (cv) or test."""
    keep = table["worn"].astype(bool) & table["y_true"].notna()
    keep &= (table["fold"] == TEST_FOLD) if source == "test" else (table["fold"] != TEST_FOLD)
    return table[keep]


def attach_slices(table, slices, columns):
    """The table with each hour's slice columns; an hour missing from slices gets "unknown"."""
    merged = table.drop(columns=[c for c in columns if c in table.columns]).merge(
        slices, on=[PARTICIPANT, HOUR_INDEX], how="left")
    for column in columns:
        merged[column] = merged[column].fillna(UNKNOWN)
    return merged


# ---- scores per group ---------------------------------------------------------------------------

def group_scores(label, frame, threshold):
    """Every score of one label on these hours: core/metrics.label_scores, plus per question
    <question>_recall, _precision and _fpr, counting P(yes) >= threshold as yes."""
    target, _ = parse_label(label)
    classes = list(get_target(target).classes)
    y = frame["y_true"].to_numpy(float)
    probs = frame[classes].to_numpy(float)
    out = label_scores(label, y, probs, frame[PARTICIPANT].to_numpy())
    for question, members in QUESTIONS[target].items():
        yes = np.isin(y, members)
        said_yes = probs[:, list(members)].sum(axis=1) >= threshold
        out[f"{question}_recall"] = _rate(said_yes & yes, yes)
        out[f"{question}_precision"] = _rate(said_yes & yes, said_yes)
        out[f"{question}_fpr"] = _rate(said_yes & ~yes, ~yes)
    return out


def _rate(hits, of):
    n = int(of.sum())
    return float(hits.sum() / n) if n else math.nan


def slice_scores(frame, label, column, threshold):
    """{group: {"hours", "participants", "metrics"}} for one label and slice column.

    Fairlearn's MetricFrame splits the hours by group and applies each score. The scores need the
    probabilities and patients too, so MetricFrame is given row positions as y_true and y_pred and
    each score looks its rows up. Each seed is scored separately and the seeds averaged.
    """
    from fairlearn.metrics import MetricFrame

    per_seed = []
    for _, part in frame.groupby("seed", sort=True):
        part = part.reset_index(drop=True)
        cache = {}

        def scores_of(positions, part=part, cache=cache):
            idx = np.asarray(positions, dtype=int)
            key = idx.tobytes()
            if key not in cache:
                cache[key] = group_scores(label, part.iloc[idx], threshold)
            return cache[key]

        names = [m for m in scores_of(np.arange(len(part))) if m != "hours"]
        metrics = {m: (lambda y_t, y_p, m=m: scores_of(y_t)[m]) for m in names}
        positions = np.arange(len(part))
        by_group = MetricFrame(metrics=metrics, y_true=positions, y_pred=positions,
                               sensitive_features=part[column].to_numpy()).by_group
        per_seed.append(by_group.astype(float))
    averaged = pd.concat(per_seed).groupby(level=0).mean()

    counts = frame.groupby(column).agg(hours=(HOUR_INDEX, "size"), participants=(PARTICIPANT, "nunique"))
    seeds = max(1, frame["seed"].nunique())
    out = {}
    for group in averaged.index:
        out[str(group)] = {
            "hours": int(counts.loc[group, "hours"] // seeds),
            "participants": int(counts.loc[group, "participants"]),
            "metrics": {m: _finite_or_none(v) for m, v in averaged.loc[group].items()},
        }
    return out


def overall_scores(frame, label, threshold):
    """The label's scores over all the hours, averaged over seeds."""
    per_seed = {seed: {label: group_scores(label, part, threshold)} for seed, part in frame.groupby("seed")}
    return {m: _finite_or_none(v) for m, v in mean_scores(per_seed)[label].items()}


def _finite_or_none(v):
    v = float(v)
    return v if math.isfinite(v) else None


# ---- the gate -----------------------------------------------------------------------------------

def check_gaps(slices_by_label, checks, columns, opts):
    """One result per check and slice column: the gap between the best and worst judged group.

    status is "fail" when the gap is over max_gap, "pass" when it isn't, and "not_judged" when fewer
    than two groups have enough hours, patients and a score.

    Raises:
        ConfigError: A check names a score its label doesn't have.
    """
    results = []
    for check in checks:
        label, metric = check["label"], check["metric"]
        for column in columns:
            groups = slices_by_label.get(label, {}).get(column)
            if not groups:
                results.append({**check, "column": column, "status": "not_judged", "gap": None,
                                "judged_groups": [], "reason": f"no {label} predictions"})
                continue
            if metric not in next(iter(groups.values()))["metrics"]:
                raise ConfigError(f"bias check {label}/{metric}: {label} has no score {metric!r}")
            judged = {g: v["metrics"][metric] for g, v in groups.items()
                      if g != UNKNOWN and v["hours"] >= opts["min_rows"]
                      and v["participants"] >= opts["min_participants"] and v["metrics"][metric] is not None}
            result = {**check, "column": column, "judged_groups": sorted(judged),
                      "skipped_groups": sorted(set(groups) - set(judged))}
            if len(judged) < 2:
                results.append({**result, "status": "not_judged", "gap": None,
                                "reason": "fewer than two groups with enough hours and patients"})
                continue
            better = higher_is_better(metric)
            best = (max if better else min)(judged, key=judged.get)
            worst = (min if better else max)(judged, key=judged.get)
            gap = abs(judged[best] - judged[worst])
            results.append({**result, "gap": gap, "best_group": best, "worst_group": worst,
                            "best": judged[best], "worst": judged[worst],
                            "status": "fail" if gap > check["max_gap"] else "pass"})
    return results


def evaluate_candidate(frames, slices, opts):
    """Slice scores, overall scores and the gate for one candidate's predictions."""
    columns = opts["slice_columns"]
    by_label, overall = {}, {}
    for label, table in frames.items():
        scored = attach_slices(scored_rows(table, opts["predictions"]), slices, columns)
        if scored.empty:
            continue
        overall[label] = overall_scores(scored, label, opts["threshold"])
        by_label[label] = {c: slice_scores(scored, label, c, opts["threshold"]) for c in columns}
    checks = check_gaps(by_label, opts["checks"], columns, opts)
    judged = sum(c["status"] != "not_judged" for c in checks)
    failed = any(c["status"] == "fail" for c in checks)
    return {"passed": judged > 0 and not failed, "judged": judged, "checks": checks,
            "overall": overall, "slices": by_label}


# ---- mitigation ---------------------------------------------------------------------------------

def answer_of(label, metric, y):
    """What a failing metric is about, per hour, for reweighing: the question's yes/no
    (off_recall -> OFF or not), a class's yes/no (recall_slight_akinesia), or the class itself."""
    target, _ = parse_label(label)
    for question, members in QUESTIONS[target].items():
        if metric.startswith(question + "_"):
            return np.where(np.isnan(y), np.nan, np.isin(y, members).astype(float))
    for i, name in enumerate(get_target(target).classes):
        if metric.endswith("_" + name):
            return np.where(np.isnan(y), np.nan, (y == i).astype(float))
    return y


def reweighing_weights(labels, slices, failures, power=1.0, max_weight=5.0):
    """One weight per hour: Kamiran and Calders' reweighing for each failing (label, metric, column).

    For a group g and answer a, w = P(g) P(a) / P(g, a): the weight that makes group and answer
    independent in the training hours, so no group's answers are under-represented. Hours with no
    answer for that label keep weight 1. The weights of several failures multiply; each is raised to
    power (0 turns reweighing off), and the product is kept between 1/max_weight and max_weight.

    Args:
        labels: The training hours' labels (data.labels), in hours order.
        slices: slice_values() of the same hours, in the same order.
        failures: The failing checks: {"label", "metric", "column"}.
    """
    weights = np.ones(len(labels))
    for f in failures:
        answer = answer_of(f["label"], f["metric"], labels[f["label"]].to_numpy(float))
        group = slices[f["column"]].to_numpy()
        known = ~np.isnan(answer)
        if not known.any():
            continue
        table = pd.DataFrame({"g": group[known], "a": answer[known]})
        n = len(table)
        p_g = table["g"].value_counts() / n
        p_a = table["a"].value_counts() / n
        p_ga = table.groupby(["g", "a"]).size() / n
        w = np.array([p_g[g] * p_a[a] / p_ga[(g, a)] for g, a in zip(table["g"], table["a"])])
        weights[known] *= w ** power
    return np.clip(weights, 1.0 / max_weight, max_weight)


def mitigate(name, template, dev, split, seeds, failures, opts, artifacts, tracker):
    """Cross-validate the candidate again with reweighted training hours, then refit and save it.

    The weights of each fit come from its own training patients only. Returns (candidate, frames):
    the candidate as train describes one, and its out-of-fold predictions.
    """
    new_name = name + MITIGATED_SUFFIX
    columns = opts["slice_columns"]
    mit = opts["mitigation"]
    seed_scores, frames, best_epochs = {}, [], []

    def weights_for(data):
        return reweighing_weights(data.labels.reset_index(drop=True), slice_values(data, columns),
                                  failures, mit["power"], mit["max_weight"])

    with tracker.run(new_name, tags={"family": template.family, "mitigated_from": name}) as run_id:
        tracker.log_params({"mitigation": {
            "method": "reweighing", "power": mit["power"], "max_weight": mit["max_weight"],
            "failures": [f"{f['label']}/{f['metric']}/{f['column']}" for f in failures]}})
        for seed in seeds:
            seed_frames = []
            for k, (train_ids, val_ids) in enumerate(split.pairs(), start=1):
                with tracker.run(f"seed-{seed}-fold-{k}", tags={"fold": str(k), "seed": str(seed)}):
                    train, val = dev.subset(train_ids), dev.subset(val_ids)
                    model = fresh(template).fit(train, val=val, sample_weight=weights_for(train), seed=seed)
                    fold_frames = rows(new_name, model.predict(val), val, fold=str(k), seed=seed)
                    tracker.log_metrics(named("val", score(fold_frames)))
                    best = (getattr(model, "training_result", None) or {}).get("best_epoch")
                    if best is not None:
                        best_epochs.append(int(best))
                    seed_frames.append(fold_frames)
            merged = concat(seed_frames)
            seed_scores[seed] = score(merged)
            frames.append(merged)
        metrics = named("cv", mean_scores(seed_scores))
        tracker.log_metrics(metrics)

        final_params = {}
        if "final_epochs" in template.params and best_epochs:
            final_params["final_epochs"] = max(1, math.floor(float(np.median(best_epochs)) + 0.5))
        with tracker.run("final", tags={k: str(v) for k, v in final_params.items()}):
            model = fresh(template, final_params)
            model.fit(dev, sample_weight=weights_for(dev), seed=seeds[0])
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
            folder = model.save(Path(artifacts) / "models" / f"{new_name}-{stamp}")
            tracker.log_artifact(folder, folder=f"model/{new_name}")
    candidate = {"name": new_name, "family": template.family, "run_id": run_id, "model_dir": str(folder),
                 "metrics": metrics, "split_hash": split.hash, "tuned": False, "final_params": final_params,
                 "mitigated_from": name}
    return candidate, concat(frames)


def mitigation_summary(original, mitigated, before, after, selection_metric):
    """What reweighing changed: each failing check's gap, and the selection metric's change."""
    after_checks = {(c["label"], c["metric"], c["column"]): c for c in after["checks"]}
    gaps = []
    for c in before["checks"]:
        if c["status"] != "fail":
            continue
        a = after_checks.get((c["label"], c["metric"], c["column"]), {})
        gaps.append({"check": f"{c['label']}/{c['metric']} by {c['column']}", "before": c["gap"],
                     "after": a.get("gap"), "max_gap": c["max_gap"], "fixed": a.get("status") == "pass"})
    cost = {}
    if selection_metric:
        b = (original.get("metrics") or {}).get(selection_metric)
        a = mitigated["metrics"].get(selection_metric)
        cost[selection_metric] = {"before": b, "after": a, "change": None if b is None or a is None else a - b}
    return {"candidate": mitigated["name"], "method": "reweighing", "gaps": gaps,
            "passed_after": after["passed"], "selection_metric": cost}


# ---- the step -----------------------------------------------------------------------------------

class CheckBias:
    """Step H: slice, gate and mitigate every candidate. See the top of this file."""

    def __init__(self, cfg, tracker):
        self.cfg, self.tracker = cfg, tracker
        self.opts = settings(cfg)
        self.artifacts = Path(cfg["resolved_paths"]["artifacts_dir"])
        self.out = self.artifacts / "bias"
        self.selection_metric = (cfg.get("selection") or {}).get("metric")
        self._built = None

    def run(self, candidates, data=None, split=None, templates=None):
        """Check every candidate; return them with their bias gate, plus any mitigated ones.

        Args:
            candidates: What train returned (or artifacts/candidates.json).
            data: A Dataset with the patients the predictions name, and every group the models need
                if mitigation may run; None loads it.
            split: The patient split; None reads paths.split_file.
            templates: {name: unfitted Model}, for mitigation; None builds them from the config.

        Raises:
            GateError: No candidate passed the bias gate, even after mitigation.
        """
        if not candidates:
            raise ConfigError("the bias step needs the candidates train trained")
        split = split or _read_split(self.cfg)
        predictions = read_predictions(self.artifacts / "predictions", sorted(LABEL_NAMES))
        if data is None:
            data = self._load(split, [c["name"] for c in candidates], templates)
        slices = slice_values(data, self.opts["slice_columns"])

        results, mitigations, out = {}, {}, []
        with self.tracker.run("bias", tags={"owner": "P3", "split_hash": split.hash,
                                            "predictions": self.opts["predictions"]}):
            self.tracker.log_params({"bias": self.opts})
            for c in candidates:
                with log_step(logger, "bias_check", candidate=c["name"]):
                    frames = {label: t[t["model"] == c["name"]] for label, t in predictions.items()}
                    result = evaluate_candidate(frames, slices, self.opts)
                    results[c["name"]] = result
                    self._log(c["name"], result)
                out.append(with_gate(c, result))
                # Only a failed check has groups and answers to reweigh; "nothing judged" has none.
                failed = any(r["status"] == "fail" for r in result["checks"])
                if failed and self._should_mitigate(c["name"]):
                    mitigated, mit_result, summary = self._mitigate(c, result, data, split, templates)
                    results[mitigated["name"]] = mit_result
                    mitigations[c["name"]] = summary
                    out.append(with_gate(mitigated, mit_result))

            report = {"predictions": self.opts["predictions"], "split_hash": split.hash,
                      "settings": self.opts, "candidates": results, "mitigation": mitigations}
            write_json(report, self.out / "bias.json")
            write_json(out, self.artifacts / "candidates.json")
            (self.out / "bias_report.md").write_text(render_report(report), encoding="utf-8")
            plot_dir = plot_results(results, self.opts, self.out / "plots")
            for path in (self.out / "bias.json", self.out / "bias_report.md", plot_dir):
                self.tracker.log_artifact(path, folder="bias")

        self._alert(results)
        passed = [c["name"] for c in out if c["gates"]["bias"]]
        logger.info("bias gate: %d of %d candidates passed", len(passed), len(out),
                    extra={"event": "bias_gate", "passed": passed})
        if not passed:
            raise GateError("bias", f"no candidate passed, even after mitigation; see {self.out / 'bias_report.md'}")
        return out

    def _load(self, split, names, templates):
        from scripts.data.loaders import load_dataset
        people = list(split.development) + (list(split.test) if self.opts["predictions"] == "test" else [])
        groups = {"patients"}
        if self.opts["mitigation"]["enabled"]:
            for name in names:
                template = self._template(name, templates)
                if template is not None:
                    groups |= set(template.needs)
        return load_dataset(self.cfg, groups=sorted(groups), participants=people)

    def _template(self, name, templates):
        if templates is not None:
            return templates.get(name)
        if self._built is None:
            from scripts.models.registry import candidates as build_candidates
            self._built = dict(build_candidates(self.cfg))
        return self._built.get(name)

    def _should_mitigate(self, name):
        mit = self.opts["mitigation"]
        return mit["enabled"] and (not mit["candidates"] or name in mit["candidates"])

    def _mitigate(self, candidate, result, data, split, templates):
        name = candidate["name"]
        template = self._template(name, templates)
        if template is None:
            raise ConfigError(f"can't mitigate {name}: it isn't a candidate in the config")
        failures = [c for c in result["checks"] if c["status"] == "fail"]
        seeds = (self.cfg.get("train") or {}).get("seeds") or [0]
        dev = data.subset(split.development)
        with log_step(logger, "bias_mitigation", candidate=name):
            mitigated, frames = mitigate(name, template, dev, split, seeds, failures, self.opts,
                                         self.artifacts, self.tracker)
            mit_result = evaluate_candidate(frames, slice_values(dev, self.opts["slice_columns"]),
                                            {**self.opts, "predictions": "cv"})
            self._log(mitigated["name"], mit_result)
        minimums = (self.cfg.get("validation_gate") or {}).get("min")
        if minimums:
            # Evaluate never saw this candidate: hold it to the same validation gate.
            from scripts.steps.evaluate import gate_failures
            failures = gate_failures(mitigated["metrics"], minimums)
            mitigated["gates_passed"], mitigated["validation_failures"] = not failures, failures
        summary = mitigation_summary(candidate, mitigated, result, mit_result, self.selection_metric)
        logger.info("mitigated %s: %s", name, "; ".join(
            f"{g['check']} gap {_fmt(g['before'])} -> {_fmt(g['after'])}" for g in summary["gaps"]),
            extra={"event": "bias_mitigated", "candidate": name})
        return mitigated, mit_result, summary

    def _log(self, name, result):
        split = "test" if self.opts["predictions"] == "test" and not name.endswith(MITIGATED_SUFFIX) else "cv"
        metrics = {f"{split}/{r['label']}/bias_gap_{slug(r['metric'])}_{slug(r['column'])}": r["gap"]
                   for r in result["checks"] if r["gap"] is not None}
        with self.tracker.run(f"bias-{name}", tags={"passed": str(result["passed"]).lower()}):
            if metrics:
                self.tracker.log_metrics(metrics)

    def _alert(self, results):
        lines = [f"{name}: {c['label']}/{c['metric']} by {c['column']}: {c['worst_group']} {c['worst']:.3f} "
                 f"vs {c['best_group']} {c['best']:.3f}, gap {c['gap']:.3f} > {c['max_gap']}"
                 for name, r in results.items() for c in r["checks"] if c["status"] == "fail"]
        lines += [f"{name}: nothing could be judged (too few hours or patients in every group)"
                  for name, r in results.items() if not r["judged"]]
        if lines:
            notify("[Model pipeline] Bias gate failed", "\n".join(lines), self.cfg)


def with_gate(candidate, result):
    """A copy of the candidate with gates["bias"], gates_passed and a short bias summary.

    gates_passed is True only when every gate in gates passed (validation, bias, ...): the one field
    select reads, so a failed bias check blocks the model. Evaluate runs first and sets gates_passed
    for validation; that verdict is kept as gates["validation"], so it's never overwritten.
    """
    c = dict(candidate)
    gates = dict(c.get("gates") or {})
    if "gates_passed" in c:
        gates.setdefault("validation", bool(c["gates_passed"]))
    gates["bias"] = result["passed"]
    c["gates"] = gates
    c["gates_passed"] = all(gates.values())
    c["bias"] = {"passed": result["passed"], "judged": result["judged"],
                 "failed": [f"{r['label']}/{r['metric']} by {r['column']}: gap {r['gap']:.3f} > {r['max_gap']}"
                            for r in result["checks"] if r["status"] == "fail"],
                 "not_judged": sum(r["status"] == "not_judged" for r in result["checks"])}
    return c


def _read_split(cfg):
    from scripts.data.folds import read_split
    return read_split(cfg)


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


# ---- the report and plots -----------------------------------------------------------------------

def render_report(report):
    """bias_report.md: what was checked, what failed, what mitigation did and cost, and the caveats."""
    opts = report["settings"]
    source = ("out-of-fold predictions of every development patient" if report["predictions"] == "cv"
              else "test patients' predictions")
    lines = ["# Bias report", "",
             f"Split `{report['split_hash']}`, scored on the {source}. Slices: "
             f"{', '.join(f'`{c}`' for c in opts['slice_columns'])}. A group is judged with at least "
             f"{opts['min_rows']} scored hours and {opts['min_participants']} patients; the rest are reported "
             "but not judged. Recall, precision and false-positive rate count P(yes) at or above "
             f"{opts['threshold']} as yes.", "", "## Gate", "",
             "| Candidate | Result | Failed checks | Not judged |", "| --- | --- | --- | --- |"]
    for name, r in report["candidates"].items():
        failed = sum(c["status"] == "fail" for c in r["checks"])
        not_judged = sum(c["status"] == "not_judged" for c in r["checks"])
        verdict = "pass" if r["passed"] else "**fail**" if failed else "**fail** (nothing judged)"
        lines.append(f"| {name} | {verdict} | {failed} | {not_judged} |")
    lines.append("")

    for name, r in report["candidates"].items():
        lines += [f"## {name}", "", "| Check | Slice | Gap | Allowed | Worst group | Best group | Result |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for c in r["checks"]:
            worst = f"{c['worst_group']} ({c['worst']:.3f})" if c.get("worst_group") else "-"
            best = f"{c['best_group']} ({c['best']:.3f})" if c.get("best_group") else "-"
            status = {"fail": "**fail**", "pass": "pass", "not_judged": "not judged"}[c["status"]]
            lines.append(f"| {c['label']}/{c['metric']} | {c['column']} | {_fmt(c['gap'])} | {c['max_gap']} | "
                         f"{worst} | {best} | {status} |")
        lines.append("")
        for label in sorted({c["label"] for c in r["checks"]} & set(r["slices"])):
            metrics = list(dict.fromkeys(c["metric"] for c in opts["checks"] if c["label"] == label))
            for column, groups in r["slices"][label].items():
                lines += [f"### {name}: {label} by {column}", "",
                          "| Group | Hours | Patients | " + " | ".join(metrics) + " |",
                          "| --- | --- | --- | " + " | ".join("---" for _ in metrics) + " |"]
                for g, v in sorted(groups.items()):
                    values = " | ".join(_fmt(v["metrics"].get(m)) for m in metrics)
                    lines.append(f"| {g} | {v['hours']} | {v['participants']} | {values} |")
                lines.append("")

    lines += ["## Mitigation", ""]
    if not report["mitigation"]:
        lines += ["No candidate needed mitigation, or mitigation was turned off.", ""]
    for name, m in report["mitigation"].items():
        lines += [f"{name} was retrained with reweighing as {m['candidate']}: each fold's training hours were "
                  "weighted so that group and answer are independent (Kamiran and Calders), with the weights "
                  "computed on that fold's training patients only. After mitigation it "
                  f"{'passes' if m['passed_after'] else 'still fails'} the gate.", "",
                  "| Check | Gap before | Gap after | Allowed | Fixed |", "| --- | --- | --- | --- | --- |"]
        for g in m["gaps"]:
            lines.append(f"| {g['check']} | {_fmt(g['before'])} | {_fmt(g['after'])} | {g['max_gap']} | "
                         f"{'yes' if g['fixed'] else 'no'} |")
        lines.append("")
        for metric, v in m["selection_metric"].items():
            change = "n/a" if v["change"] is None else f"{v['change']:+.3f}"
            lines += [f"Trade-off: the selection metric `{metric}` went from {_fmt(v['before'])} to "
                      f"{_fmt(v['after'])} ({change}). Select weighs both candidates on the same gates.", ""]

    failing = [(n, c) for n, r in report["candidates"].items() for c in r["checks"] if c["status"] == "fail"]
    if failing:
        lines += ["## Suggested next steps for the gaps that remain", ""]
        for n, c in failing:
            lines.append(f"- {n}, {c['label']}/{c['metric']} by {c['column']}: the worst group is "
                         f"{c['worst_group']}. Options: reweighing (applied above where enabled), resampling that "
                         "group's hours, or recruiting more patients like them.")
        lines.append("")
    lines += ["## How to read this", "",
              "- Most slices describe the patient, not the hour, so a group's score rests on few patients. "
              "Check the Patients column before reading a gap as bias: with few patients a gap may show how "
              "two people differ.",
              "- Per-group decision thresholds could close a recall gap without retraining, but they would need "
              "each patient's group at prediction time and would make one score mean different things for "
              "different patients, so they aren't used.",
              "- A group named unknown has no value in the data and is never judged.", ""]
    return "\n".join(lines)


def plot_results(results, opts, folder):
    """One figure per candidate and check: the metric per group, one panel per slice column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    columns = opts["slice_columns"]
    for name, r in results.items():
        by_check = {(c["label"], c["metric"], c["column"]): c for c in r["checks"]}
        for check in opts["checks"]:
            label, metric = check["label"], check["metric"]
            if label not in r["slices"]:
                continue
            fig, axes = plt.subplots(1, len(columns), figsize=(2.6 * len(columns), 3.4), sharey=True,
                                     squeeze=False)
            for ax, column in zip(axes[0], columns):
                groups = r["slices"][label][column]
                names = sorted(groups)
                values = [groups[g]["metrics"].get(metric) for g in names]
                c = by_check[(label, metric, column)]
                judged = set(c.get("judged_groups", []))
                ax.bar(range(len(names)), [v if v is not None else 0 for v in values],
                       color=["tab:blue" if g in judged else "lightgrey" for g in names])
                for i, g in enumerate(names):
                    ax.text(i, 0.02, f"n={groups[g]['participants']}", ha="center", fontsize=7, rotation=90)
                overall = r["overall"].get(label, {}).get(metric)
                if overall is not None:
                    ax.axhline(overall, color="black", linestyle="--", linewidth=0.8)
                failed = c["status"] == "fail"
                ax.set_title(f"{column}\ngap {_fmt(c['gap'])}" + (" FAIL" if failed else ""), fontsize=8,
                             color="tab:red" if failed else "black")
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            axes[0][0].set_ylabel(metric)
            fig.suptitle(f"{name}: {label}/{metric} by group (grey: not judged, n: patients, dashed: all)",
                         fontsize=9)
            fig.tight_layout()
            fig.savefig(folder / f"{name}__{label}__{slug(metric)}.png", dpi=110)
            plt.close(fig)
    return folder


def main(argv=None):
    from scripts.tracking.mlflow_tracker import MlflowTracker
    from scripts.utils.config import load_config
    from scripts.utils.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Check every candidate for bias (see scripts/steps/bias.py)")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg)
    candidates_file = Path(cfg["resolved_paths"]["artifacts_dir"]) / "candidates.json"
    if not candidates_file.exists():
        raise SchemaError(f"no {candidates_file}: run train first")
    try:
        results = CheckBias(cfg, MlflowTracker.from_config(cfg)).run(read_json(candidates_file))
    except GateError as e:
        logger.error(str(e))
        return 1
    print(json.dumps({c["name"]: c["bias"] for c in results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
