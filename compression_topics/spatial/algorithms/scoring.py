"""Independent residual-feature ranking and reconstruction distortion objectives."""
import torch


def feature_scores(deviation, *, selection='deviation', query_energy=None):
    score = deviation.square()
    if selection == 'query':
        if query_energy is None:
            raise ValueError('query feature selection requires calibrated query energy')
        score = score * query_energy
    elif selection != 'deviation':
        raise ValueError('unknown residual feature selection')
    return score


def reconstruction_error(source, restored, *, distortion='relative', query_energy=None,
                         token_weights=None):
    squared = (restored - source).square()
    if distortion == 'query':
        if query_energy is None:
            raise ValueError('query distortion requires calibrated query energy')
        squared = squared * query_energy
    elif distortion != 'relative':
        raise ValueError('unknown distortion objective')
    # Query distortion approximates absolute q·key-error squared, not a
    # relative key reconstruction error. Keep ordinary scoring unchanged.
    error = squared.sum(-1)
    if distortion == 'relative':
        error = error / source.square().sum(-1).clamp_min(1e-12)
    if token_weights is not None:
        if token_weights.shape != error.shape or not torch.isfinite(token_weights).all() or (token_weights < 0).any():
            raise ValueError('token importance must be finite, nonnegative and match grouped tokens')
        error = error * token_weights
    return error


def query_energy(path, layer_idx, digest, tensor):
    # Reuse the existing checksum-verified cache and RoPE-plane tying.
    from compression_topics.spatial.algorithms.group_rd import plane_tied, _weights
    weights = plane_tied(_weights(path, layer_idx, digest)).to(tensor.device)
    if weights.shape != (tensor.shape[1], tensor.shape[-1]) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('query weights do not match the loaded layer/head shape')
    return weights.reshape(1, tensor.shape[1], 1, 1, tensor.shape[-1])
