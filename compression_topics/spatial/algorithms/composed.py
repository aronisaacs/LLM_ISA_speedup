"""One explicit configuration for orthogonal spatial compression components.

Legacy study names remain compatibility entry points. New studies can use
spatial_composed and describe representation, residual feature scoring,
distortion, importance and budget decisions independently. Configuration
validation rejects combinations that lack an implementation or causal inputs.
"""
from dataclasses import asdict, dataclass, fields
import math
import torch
from compression_topics.spatial.algorithms import hierarchy, online_pairs, presentation_spatial
from compression_topics.spatial.algorithms.scoring import query_energy
from engine.kv_compress.methods import METHODS, OPTION_VALIDATORS


@dataclass(frozen=True)
class SpatialConfig:
    decision: str = 'online'
    representation: str = 'pairs'
    residual_entries: int | None = None
    residual_menu: tuple | None = None
    directional: bool = True
    select_by: str = 'deviation'
    distortion: str = 'relative'
    importance: str = 'none'
    query_weights: str | None = None
    query_weights_sha256: str | None = None
    control: dict | None = None
    max_group: int = 8
    mask_layout: str = 'separate'

    def __post_init__(self):
        if self.decision not in ('fixed', 'online', 'offline'):
            raise ValueError('decision must be fixed, online or offline')
        if self.representation not in ('pairs', 'quads', 'hierarchy'):
            raise ValueError('representation must be pairs, quads or hierarchy')
        if isinstance(self.max_group, bool) or self.max_group not in (2, 4, 8):
            raise ValueError('max_group must be 2, 4 or 8')
        hierarchy.accounting(self.mask_layout)
        if self.representation != 'hierarchy' and self.mask_layout != 'separate':
            raise ValueError('shared masks currently require the online hierarchy')
        expected_menu = hierarchy.MENUS[2] if self.representation == 'hierarchy' else online_pairs.RESIDUALS
        if self.residual_menu is not None and (any(isinstance(r, bool) or not isinstance(r, int) for r in self.residual_menu)
                                               or tuple(self.residual_menu) != expected_menu):
            raise ValueError(f'adaptive residual menu for {self.representation} must be {expected_menu}')
        if not isinstance(self.directional, bool):
            raise ValueError('directional must be boolean')
        if self.select_by not in ('deviation', 'query') or self.distortion not in ('relative', 'query'):
            raise ValueError('unknown feature selection or distortion objective')
        if self.importance not in ('none', 'prefill_attention'):
            raise ValueError('unknown token importance mode')
        needs_query = self.select_by == 'query' or self.distortion == 'query'
        if needs_query != (self.query_weights is not None and self.query_weights_sha256 is not None):
            raise ValueError('query-aware components require calibrated weights and checksum')
        if not needs_query and (self.query_weights is not None or self.query_weights_sha256 is not None):
            raise ValueError('unused query weights are not allowed')
        if self.representation == 'hierarchy':
            if self.decision != 'online' or needs_query or self.importance != 'none':
                raise ValueError('hierarchy currently supports online ordinary-error decisions only')
            if self.control is not None and online_pairs.Control(**self.control).startup_tokens % 8:
                raise ValueError('hierarchical startup requires complete eight-token windows')
        if self.decision == 'fixed':
            if isinstance(self.residual_entries, bool) or not isinstance(self.residual_entries, int) or self.residual_entries < 0:
                raise ValueError('fixed decisions require a nonnegative residual count')
            if self.control is not None:
                raise ValueError('fixed decisions do not use a controller')
        else:
            if self.residual_entries is not None:
                raise ValueError('adaptive decisions use the residual menu, not a fixed count')
            if self.representation not in ('pairs', 'hierarchy'):
                raise ValueError('adaptive quad menus are not implemented yet')
            if not self.directional:
                raise ValueError('adaptive representations require direction and norm restoration')
        if not self.directional and (self.representation != 'pairs' or self.residual_entries != 0 or needs_query):
            raise ValueError('raw representation supports fixed merge-only pairs')
        if self.decision != 'online' and self.control is not None:
            raise ValueError('only online decisions use controller settings')
        if self.control is not None:
            online_pairs.Control(**self.control)
        if self.decision == 'online' and self.importance != 'none':
            raise ValueError('prefill attention importance uses later queries; not causal for online decisions')

    def pipeline(self, saving, *, k_layers='all', v_layers='all'):
        """Validate components before producing a serializable engine policy."""
        if not math.isfinite(saving) or saving <= 0:
            raise ValueError('saving must be positive')
        if self.select_by == 'query' or self.distortion == 'query':
            if v_layers not in ([], (), None):
                raise ValueError('query-aware components apply to keys; configure values separately')
        result = asdict(self)
        result['residual_menu'] = list(self.residual_menu or (hierarchy.MENUS[2] if self.representation == 'hierarchy' else online_pairs.RESIDUALS))
        return {'pipeline': [{'method': 'spatial_composed', 'saving': saving,
                              'k_layers': k_layers, 'v_layers': v_layers, **result}]}


def apply(tensor, *, layer_idx, target, saving, decision='online', representation='pairs',
          residual_entries=None, residual_menu=None, directional=True,
          select_by='deviation', distortion='relative', importance='none', query_weights=None,
          query_weights_sha256=None, control=None, max_group=8, mask_layout='separate', seq_start=0, rope_tables=None, token_weights=None):
    SpatialConfig(decision, representation, residual_entries, None if residual_menu is None else tuple(residual_menu), directional,
                           select_by, distortion, importance, query_weights, query_weights_sha256, control, max_group, mask_layout)
    if (select_by == 'query' or distortion == 'query') and target != 'k':
        raise ValueError('query-aware components apply only to keys')
    if seq_start:
        return tensor  # All current configurations leave appended tokens dense.
    if importance == 'prefill_attention':
        if token_weights is None or token_weights.shape != tensor.shape[:3] or not torch.isfinite(token_weights).all() or (token_weights < 0).any():
            raise ValueError('prefill attention importance requires matching nonnegative token weights')
    elif token_weights is not None:
        raise ValueError('token weights require an explicit importance mode')
    if representation == 'hierarchy':
        return hierarchy.apply(tensor, layer_idx=layer_idx, target=target, saving=saving,
                               max_group=max_group, control=control, rope_tables=rope_tables, mask_layout=mask_layout)
    if decision == 'fixed':
        return presentation_spatial.apply_components(tensor, layer_idx=layer_idx, target=target, saving=saving,
            residual_entries=residual_entries, group_size=2 if representation=='pairs' else 4,
            directional=directional, select_by=select_by, query_weights=query_weights,
            query_weights_sha256=query_weights_sha256, distortion=distortion,
            token_weights=token_weights, rope_tables=rope_tables)
    energy = query_energy(query_weights, layer_idx, query_weights_sha256, tensor) if select_by=='query' or distortion=='query' else None
    return online_pairs.apply_components(tensor, layer_idx=layer_idx, target=target, saving=saving,
        decision=decision, control=control, selection=select_by, distortion=distortion,
        query_energy=energy, token_weights=token_weights, rope_tables=rope_tables)


METHODS['spatial_composed'] = apply


def validate_config(options):
    component_fields = {field.name for field in fields(SpatialConfig)}
    SpatialConfig(**{key: value for key, value in options.items() if key in component_fields})


OPTION_VALIDATORS['spatial_composed'] = validate_config
