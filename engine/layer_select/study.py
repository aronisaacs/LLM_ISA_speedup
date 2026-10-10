"""Shared interfaces between local calibration, allocation and evaluation.

Studies declare candidate policies, sample schedules and comparison rules.
This module measures matched local candidates, selects settings, and assembles
whole-model allocations without knowing their compression algorithm.

New records distinguish target_saving, measured_saving and allocation_saving.
Historical budget/compression fields remain compatibility aliases for saved
studies; callers should use explicit fields for interpretation.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

from engine.eval_runner.cache import identities_match
from engine.layer_select.calibration import measurement, perplexity
from engine.layer_select.greedy.calibrated import allocate


def read_json(path):
    return json.loads(Path(path).read_text())


def measure_candidates(rows, folder, *, tolerance, budget_rule='minimum'):
    """Measure one-slot policies on the same chunks and scoring protocol as dense.

    minimum is for methods guaranteeing at least the requested savings;
    absolute is for controllers that aim to remain near their target.
    These policies are supplied by the study, not inferred from method names.
    """
    if budget_rule not in ('minimum', 'absolute') or tolerance < 0:
        raise ValueError('invalid calibration budget rule')
    folder = Path(folder)
    dense = read_json(folder / 'dense.json')
    baseline = perplexity(dense)
    baseline_scores = dense['chunk_scores']
    measured = []
    for candidate in rows:
        payload = read_json(folder / f"{candidate['name']}.json")
        if payload['simulation']['kv'] != candidate['kv']:
            raise ValueError(f"calibration policy differs: {candidate['name']}")
        # Candidate and baseline may differ only in their compression policy.
        baseline_identity = {**dense['simulation'], 'kv': {'pipeline': []}}
        candidate_identity = {**payload['simulation'], 'kv': {'pipeline': []}}
        if not identities_match(baseline_identity, candidate_identity):
            raise ValueError('calibration model or evaluation protocol differs')
        scores = payload['chunk_scores']
        if [(r['chunk'], r['tokens']) for r in scores] != [(r['chunk'], r['tokens']) for r in baseline_scores]:
            raise ValueError('calibration samples differ')
        if len(scores) < 2 or payload.get('scoring') != dense.get('scoring'):
            raise ValueError('calibration needs matching scoring and at least two chunks')
        stats = measurement(payload)
        target = candidate['budget']
        saving = stats['compression']
        invalid = saving < target - tolerance if budget_rule == 'minimum' else abs(saving - target) > tolerance
        if invalid or stats['shortfall_updates']:
            raise ValueError(f"candidate cannot meet measured byte budget: {candidate['name']}")
        differences = [r['nll'] - d['nll'] for r, d in zip(scores, baseline_scores)]
        ppl = perplexity(payload)
        measured.append({**candidate, **stats, 'ppl': ppl,
                         'delta_nll': math.log(ppl / baseline),
                         'paired_nll_standard_error': statistics.stdev(differences) / math.sqrt(len(differences)),
                         'chunk_scores': scores, 'target_saving': target, 'measured_saving': saving,
                         'calibration_source': {'simulation': payload['simulation'],
                                                'scoring': payload.get('scoring'),
                                                'chunks': [r['chunk'] for r in scores],
                                                'scored_tokens': [r['tokens'] for r in scores]}})
    return {'dense_ppl': baseline, 'candidates': measured}


def select_settings(calibration, *, include, group_fields=('layer', 'target', 'budget'), rank=None):
    """Choose a setting within each slot/target group, independently by method.

    include defines the method family; rank defines any scientific tie-breaks.
    Input order is retained for exact ties.
    """
    rank = rank or (lambda row: row['ppl'])
    groups = {}
    for row in calibration['candidates']:
        if not include(row):
            continue
        key = tuple(row[field] for field in group_fields)
        if key not in groups or rank(row) < rank(groups[key]):
            groups[key] = row
    return list(groups.values())


def allocate_settings(calibration, settings, layers, targets, budget, *, trim_overshoot=False, allocator=allocate):
    """Assemble a whole-model policy from local curves using nominal targets.

    Nominal targets preserve the current allocation rule. Actual calibration
    savings are kept separately, and actual whole-model savings require a new
    evaluation. The optional trim is the existing current-study behavior;
    its changed slot is explicitly marked as not directly calibrated.
    """
    selected = []
    for row in settings:
        measured = row.get('measured_saving', row['compression'])
        selected.append({**row, 'measured_calibration_compression': measured,
                         'target_saving': row.get('target_saving', row['budget']),
                         'measured_saving': measured, 'allocation_saving': row['budget'],
                         'compression': row['budget']})
    result = allocator({**calibration, 'selected': selected}, layers, targets, budget)
    # Some protocol tests substitute a minimal allocator; no policy to assemble.
    if 'assignment' not in result:
        return result
    result = deepcopy(result)
    overshoot = (result['compression'] - budget) * len(targets) * layers
    if trim_overshoot and overshoot > 1e-10:
        row = max(result['assignment'], key=lambda r: r['budget'])
        if row['budget'] <= overshoot:
            raise ValueError('cannot trim allocation')
        row['budget'] -= overshoot
        row['target_saving'] = row['budget']
        row['allocation_saving'] = row['budget']
        row['kv'] = {'pipeline': [{**step, 'saving': row['budget']} for step in row['kv']['pipeline']]}
        row['allocation_adjustment'] = {'calibrated_target_saving': row.get('calibration_target_saving', row['budget'] + overshoot),
                                        'directly_calibrated': False}
        # Legacy compression remains the pre-trim per-slot alias; explicit
        # allocation_saving is authoritative for the final assigned policy.
        result['compression'] = budget
        result['whole_kv_compression'] = budget * len(targets) / 2
    if trim_overshoot:
        result['compression'] = budget
        result['whole_kv_compression'] = budget * len(targets) / 2
    result['target_saving'] = budget
    result['allocation_saving'] = result['compression']
    result['kv'] = {'pipeline': [step for row in result['assignment'] for step in row['kv']['pipeline']]}
    return result


def require_matched_savings(rows, budgets, arms, *, tolerance):
    """Gate whole-model validation using actual bytes, never target aliases."""
    for budget in budgets:
        selected = [row for row in rows if row['budget'] == budget]
        if len(selected) != len(arms) or {row['arm'] for row in selected} != set(arms):
            raise ValueError('validation is missing a method or contains duplicates')
        savings = [row['actual_saving'] for row in selected]
        if any(not math.isfinite(saving) or abs(saving - budget) > tolerance for saving in savings) or max(savings) - min(savings) > tolerance:
            raise ValueError('holdout savings not matched within tolerance')
