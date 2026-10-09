"""Rate-distortion block modes: accounting, selection, budgets and stage-A analysis."""
import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from compression_topics.spatial.algorithms import group_calibrated, group_rd, pair_gate
from compression_topics.spatial.algorithms.storage import metadata_bits, vector_group_bits
from compression_topics.spatial.scripts import group_rd_offline as offline
from engine.kv_compress.rope import RopeTables, apply_rope
from engine.kv_compress.spec import parse_kv_spec

ROPE = RopeTables(10000., 128)


def correlated_keys(heads=2, length=256, noise=.4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(1, heads, 1, 128, generator=generator)
    walk = torch.randn(1, heads, length, 128, generator=generator).cumsum(-2) * .05
    return base + walk + noise * torch.randn(1, heads, length, 128, generator=generator)


class MenuAccountingTests(unittest.TestCase):
    def test_default_menu_is_eight_fixed_size_modes(self):
        names = group_rd.mode_names(menu=group_rd.DEFAULT_MENU)
        self.assertEqual(sorted(names), sorted(group_rd.DEFAULT_MENU))  # codes follow canonical order
        self.assertEqual(group_rd.mode_bits(len(names)), 3)
        bits = group_rd.bit_table(names).tolist()
        self.assertEqual(len(set(bits)), 8)  # eight distinct fixed sizes
        self.assertEqual(max(bits), 4 * 128 * 16)

    def test_full_menu_has_41_modes_and_a_six_bit_code(self):
        names = group_rd.mode_names()
        self.assertEqual(len(names), 41)
        self.assertEqual(group_rd.mode_bits(len(names)), 6)
        self.assertEqual(names[0], 'D')
        self.assertIn('Pd+8', names)
        self.assertIn('Q32', names)
        self.assertEqual(group_rd.mode_names((8,), 'pairs'), ['D', 'Pd+8', 'P8+d', 'P8+8'])
        self.assertEqual(group_rd.mode_names((8,), 'quads'), ['D', 'Q8'])
        self.assertEqual(group_rd.mode_names(menu=['D', 'Q0', 'P4+4']), ['D', 'P4+4', 'Q0'])
        with self.assertRaises(ValueError):
            group_rd.mode_names(menu=['Q0'])  # dense fallback is required

    def test_bit_table_matches_storage_formulas(self):
        names = group_rd.mode_names()
        bits = group_rd.bit_table(names)
        pair = lambda r: vector_group_bits(r, 128, 2, include_norms=True)
        quad = lambda r: vector_group_bits(r, 128, 4, include_norms=True)
        self.assertEqual(bits[names.index('D')], 4 * 128 * 16)
        self.assertEqual(bits[names.index('Pd+8')], 2 * 128 * 16 + pair(8))
        self.assertEqual(bits[names.index('P0+32')], pair(0) + pair(32))
        self.assertEqual(bits[names.index('Q16')], quad(16))
        # Index masks: 7 bits per entry instead of a 128-bit mask per residual.
        index = group_rd.bit_table(names, mask='index')
        self.assertEqual(index[names.index('Q4')], 16 * (128 + 4 * 4 + 4) + 4 * 4 * 7)
        self.assertEqual(index[names.index('Pd+16')], 2 * 128 * 16 + 16 * (128 + 16 + 2) + 16 * 7)
        self.assertEqual(index[names.index('Q0')], bits[names.index('Q0')])
        auto = group_rd.bit_table(names, mask='auto')
        self.assertEqual(auto[names.index('Q16')], index[names.index('Q16')])  # 16 x 7 < 128
        self.assertEqual(auto[names.index('Q32')], bits[names.index('Q32')])   # 32 x 7 > 128

    def test_stats_account_every_stored_bit(self):
        x = correlated_keys(length=258)
        pair_gate.reset_stats()
        group_rd.apply(x, layer_idx=3, target='k', saving=.35, rope_tables=ROPE)
        stats = pair_gate.pop_stats()['k_layer_3']
        blocks = 2 * (258 // 4)
        self.assertEqual(stats['metadata_bits'], metadata_bits(blocks, 3))  # 8-mode default menu
        merged = sum(stats[f'pair_r{r}'] * vector_group_bits(r, 128, 2, include_norms=True)
                     + stats[f'quad_r{r}'] * vector_group_bits(r, 128, 4, include_norms=True)
                     for r in group_rd.RESIDUALS)
        dense_pairs = 2 * (blocks - stats['quad_blocks']) - stats['merged_pairs']
        expected = merged + dense_pairs * 2 * 128 * 16 + stats['metadata_bits'] + 2 * 2 * 128 * 16
        self.assertEqual(stats['stored_bits'], expected)
        self.assertEqual(stats['norm_bits'], 16 * (2 * stats['merged_pairs'] + 4 * stats['quad_blocks']))
        self.assertEqual(stats['dense_bits'], x.numel() * 16)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        pair_gate.reset_stats()

    def test_menu_selection_minimizes_distortion_plus_price(self):
        menu = group_rd.build_menu(correlated_keys(length=64), rope_tables=ROPE)
        for lam in (0., 1e-5, 1e-3):
            choice = group_rd.select(menu, lam)
            cost = menu.distortion + lam * menu.bits
            torch.testing.assert_close(cost.gather(-1, choice.unsqueeze(-1)).squeeze(-1), cost.amin(-1))
        self.assertTrue(torch.all(group_rd.select(menu, 0.) == 0))  # free bits: all dense

    def test_coarse_units_share_one_mode(self):
        menu = group_rd.build_menu(correlated_keys(length=256), rope_tables=ROPE)
        lam = 1e-4
        fine = group_rd.select(menu, lam)
        for granularity in (16, 64):
            choice = group_rd.select(menu, lam, granularity)
            per = granularity // 4
            self.assertTrue(torch.all(choice.reshape(1, 2, -1, per) == choice.reshape(1, 2, -1, per)[..., :1]))
            cost = lambda c: float((menu.distortion + lam * menu.bits).gather(-1, c.unsqueeze(-1)).sum())
            self.assertGreaterEqual(cost(choice) + 1e-9, cost(fine))

    def test_bisection_hits_budget_with_small_overshoot(self):
        x = correlated_keys(length=512)
        for granularity in (4, 16, 64):
            for saving in (.1, .3, .5, .7):
                pair_gate.reset_stats()
                group_rd.apply(x, layer_idx=0, target='k', saving=saving, granularity=granularity, rope_tables=ROPE)
                stats = pair_gate.pop_stats()['k_layer_0']
                achieved = 1 - stats['stored_bits'] / stats['dense_bits']
                self.assertGreaterEqual(achieved, saving)
                self.assertLess(achieved, saving + .02)
        with self.assertRaisesRegex(ValueError, 'maximum'):
            group_rd.apply(x, layer_idx=0, target='k', saving=.9, rope_tables=ROPE)

    def test_batched_price_search_matches_bisection(self):
        for seed in range(3):
            x = correlated_keys(length=512, seed=20 + seed)
            for menu in (list(group_rd.DEFAULT_MENU), ['D', 'P32/d', 'P32+32', 'P16/0', 'Q32', 'Q16', 'Q8', 'Q0']):
                table = group_rd.build_menu(x, rope_tables=ROPE, menu=menu)
                for granularity in (4, 16, 64):
                    for saving in (.1, .35, .5, .7):
                        target = x.numel() * 16 * (1 - saving)
                        fast = group_rd.solve_lambda(table, target, granularity)
                        slow = group_rd.solve_lambda_bisection(table, target, granularity)
                        self.assertTrue(torch.equal(group_rd.select(table, fast, granularity),
                                                    group_rd.select(table, slow, granularity)))
                        self.assertLessEqual(group_rd.total_bits(table, fast, granularity), target)

    def test_fixed_price_is_causal_per_block(self):
        x = correlated_keys(length=128)
        whole = group_rd.select(group_rd.build_menu(x, rope_tables=ROPE), 1e-4)
        head = group_rd.select(group_rd.build_menu(x[..., :64, :], rope_tables=ROPE), 1e-4)
        self.assertTrue(torch.equal(whole[..., :16], head))

    def test_rejects_values_bad_options_and_skips_decode(self):
        x = correlated_keys(length=16)
        with self.assertRaisesRegex(ValueError, 'values have no query'):
            group_rd.apply(x, layer_idx=0, target='v', saving=.2, distortion='query',
                           query_weights='unused.pt', rope_tables=ROPE)
        with self.assertRaisesRegex(ValueError, 'target'):
            group_rd.apply(x, layer_idx=0, target='q', saving=.2, rope_tables=ROPE)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            group_rd.apply(x, layer_idx=0, target='k', saving=.2, lam=1., rope_tables=ROPE)
        with self.assertRaisesRegex(ValueError, 'query_weights'):
            group_rd.apply(x, layer_idx=0, target='k', saving=.2, distortion='query', rope_tables=ROPE)
        with self.assertRaisesRegex(ValueError, 'granularity'):
            group_rd.apply(x, layer_idx=0, target='k', saving=.2, granularity=8, rope_tables=ROPE)
        self.assertTrue(torch.equal(group_rd.apply(x, layer_idx=0, target='k', saving=.2, seq_start=4,
                                                   rope_tables=ROPE), x))
        step = {'method': 'group_rd', 'k_layers': [0], 'saving': .3, 'granularity': 16,
                'menu': ['D', 'P0+0', 'Q0', 'Q8'], 'mask': 'index'}
        parse_kv_spec({'pipeline': [step]})
        with self.assertRaises(ValueError):
            parse_kv_spec({'pipeline': [{**step, 'group_size': 2}]})

    def test_query_weights_file_steers_the_choice(self):
        x = correlated_keys(length=128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pt'
            torch.save({'weights': torch.rand(4, 2, 128) + .1}, path)
            group_rd._load_weights.cache_clear()
            group_rd.apply(x, layer_idx=2, target='k', saving=.4, distortion='query',
                           query_weights=str(path), rope_tables=ROPE)
            stats = pair_gate.pop_stats()['k_layer_2']
            self.assertGreaterEqual(1 - stats['stored_bits'] / stats['dense_bits'], .4)
            weights = torch.load(path, weights_only=True)['weights'][2]
            menu = group_rd.build_menu(x, rope_tables=ROPE, distortion='query', weights=weights)
            squared = group_rd.build_menu(x, rope_tables=ROPE, distortion='squared')
            uniform = group_rd.build_menu(x, rope_tables=ROPE, distortion='query', weights=torch.ones(2, 128))
            torch.testing.assert_close(uniform.distortion, squared.distortion)
            self.assertFalse(torch.allclose(menu.distortion, squared.distortion))

    def test_query_distortion_ignores_rope_frame(self):
        # Plane-tied weights make the weighted error identical in every RoPE frame.
        weights = group_rd.plane_tied(torch.rand(2, 128))
        delta = torch.randn(1, 2, 8, 128)
        cos, sin = ROPE.cos_sin(torch.arange(8) * 37, torch.float32)
        rotated = apply_rope(delta, cos, sin, inverse=False)
        w = weights.reshape(1, 2, 1, 128)
        torch.testing.assert_close((rotated.square() * w).sum(-1), (delta.square() * w).sum(-1))


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        pair_gate.reset_stats()

    def test_fixed_choices_match_group_calibrated_bit_for_bit(self):
        x = correlated_keys(length=256)
        for target, size, kept in itertools.product('kv', (2, 4), (0, 8, 32)):
            modes = 'pairs' if size == 2 else 'quads'
            with self.subTest(target=target, size=size, kept=kept):
                pair_gate.reset_stats()
                actual = group_rd.apply(x, layer_idx=0, target=target, lam=1e9, residuals=[kept],
                                        modes=modes, menu=None, rope_tables=ROPE)
                ours = pair_gate.pop_stats()[f'{target}_layer_0']
                maximum = group_calibrated.group_saving(kept, group_size=size)
                expected = group_calibrated.apply(x, layer_idx=0, target=target, saving=maximum - 1e-3,
                                                  group_size=size, residual_entries=kept, rope_tables=ROPE)
                theirs = pair_gate.pop_stats()[f'{target}_layer_0']
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for field in ('stored_bits', 'metadata_bits', 'norm_bits', 'residual_mask_bits'):
                    self.assertEqual(ours[field], theirs[field], field)

    def test_mixed_modes_reconstruct_each_block_by_its_mode(self):
        x = correlated_keys(length=64)
        menu = group_rd.build_menu(x, rope_tables=ROPE)
        names = menu.names
        choice = torch.tensor([names.index(n) for n in ('D', 'Pd+4', 'P0+d', 'Q8') * 4]).repeat(1, 2, 1)
        out = group_rd.reconstruct(x, menu, choice, ROPE)
        blocks = out.reshape(1, 2, 16, 4, 128)
        source = x.reshape(1, 2, 16, 4, 128)
        self.assertTrue(torch.equal(blocks[:, :, 0::4], source[:, :, 0::4]))      # dense
        self.assertTrue(torch.equal(blocks[:, :, 1::4, :2], source[:, :, 1::4, :2]))  # dense first pair
        self.assertTrue(torch.equal(blocks[:, :, 2::4, 2:], source[:, :, 2::4, 2:]))  # dense second pair
        self.assertFalse(torch.equal(blocks[:, :, 3::4], source[:, :, 3::4]))
        torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))
        # Reported distortion is the cosine error of what reconstruct returns.
        cosine = torch.nn.functional.cosine_similarity(out, x, dim=-1).reshape(1, 2, 16, 4)
        torch.testing.assert_close((1 - cosine).sum(-1).double(),
                                   menu.distortion.gather(-1, choice.unsqueeze(-1)).squeeze(-1),
                                   atol=1e-4, rtol=1e-4)


class FlaggedPairTests(unittest.TestCase):
    def setUp(self):
        pair_gate.reset_stats()

    def test_flagged_names_bits_and_validation(self):
        names = group_rd.mode_names(menu=['D', 'P32/d', 'P32/0', 'Q16'])
        self.assertEqual(names, ['D', 'Q16', 'P32/d', 'P32/0'])
        bits = group_rd.bit_table(names)
        full = group_rd.mode_names()
        ordered = group_rd.bit_table(full)
        self.assertEqual(bits[2], ordered[full.index('Pd+32')] + 1)  # same size plus the flag bit
        self.assertEqual(bits[3], ordered[full.index('P32+0')] + 1)
        for bad in (['D', 'P32/32'], ['D', 'P7/d'], ['P32/d']):
            with self.assertRaises(ValueError):
                group_rd.mode_names(menu=bad)
        with self.assertRaises(ValueError):
            group_rd.mode_names(modes='quads', menu=['D', 'P32/d'])

    def test_flagged_mode_takes_the_better_orientation_per_block(self):
        x = correlated_keys(length=256, seed=8)
        flagged = group_rd.build_menu(x, rope_tables=ROPE, menu=['D', 'P32/d'])
        ordered = group_rd.build_menu(x, rope_tables=ROPE, menu=['D', 'P32+d', 'Pd+32'])
        a, b = ordered.distortion[..., ordered.names.index('P32+d')], ordered.distortion[..., ordered.names.index('Pd+32')]
        torch.testing.assert_close(flagged.distortion[..., 1], torch.minimum(a, b))
        swap = flagged.swap[..., 1]
        self.assertTrue(swap.any() and (~swap).any())  # both orientations occur

    def test_one_page_format_merges_different_pairs_in_different_blocks(self):
        x = correlated_keys(length=128, seed=9)
        # Make the second pair of even blocks and the first pair of odd blocks near-identical.
        blocks = x.reshape(1, 2, 32, 4, 128)
        blocks[:, :, 0::2, 3] = blocks[:, :, 0::2, 2] * 1.01
        blocks[:, :, 1::2, 1] = blocks[:, :, 1::2, 0] * 1.01
        x = blocks.reshape(1, 2, 128, 128)
        out = group_rd.apply(x, layer_idx=0, target='v', lam=1e9, menu=['D', 'P0/d'], granularity=64)
        stats = pair_gate.pop_stats()['v_layer_0']
        changed = (out != x).any(-1).reshape(1, 2, 32, 4)
        self.assertTrue(torch.equal(changed[:, :, 0::2, :2], torch.zeros_like(changed[:, :, 0::2, :2])))
        self.assertTrue(changed[:, :, 0::2, 2:].all())
        self.assertTrue(torch.equal(changed[:, :, 1::2, 2:], torch.zeros_like(changed[:, :, 1::2, 2:])))
        self.assertTrue(changed[:, :, 1::2, :2].all())
        pair = vector_group_bits(0, 128, 2, include_norms=True)
        self.assertEqual(stats['stored_bits'], 64 * (2 * 128 * 16 + pair + 1) + metadata_bits(4, 1))


class SharedMaskAndSelectionTests(unittest.TestCase):
    def setUp(self):
        pair_gate.reset_stats()

    def test_shared_mask_quads_store_one_mask(self):
        names = group_rd.mode_names(menu=['D', 'S16', 'Q16'])
        self.assertEqual(names, ['D', 'Q16', 'S16'])
        bits = group_rd.bit_table(names)
        self.assertEqual(bits[2], bits[1] - 3 * 128)  # three fewer 128-bit masks
        self.assertEqual(group_rd.bit_table(['D', 'S16'], mask='auto')[1], 16 * (128 + 64 + 4) + 16 * 7)
        with self.assertRaises(ValueError):
            group_rd.mode_names(modes='pairs', menu=['D', 'S16'])

    def test_shared_mask_keeps_the_same_features_for_all_four_tokens(self):
        x = correlated_keys(length=64, seed=10)
        out = group_rd.apply(x, layer_idx=0, target='v', lam=1e9, menu=['D', 'S8'])
        stats = pair_gate.pop_stats()['v_layer_0']
        self.assertEqual(stats['shared_r8'], 2 * 16)
        self.assertEqual(stats['residual_mask_bits'], 2 * 16 * 128)
        self.assertEqual(stats['stored_bits'], 2 * 16 * (16 * (128 + 32 + 4) + 128) + metadata_bits(32, 1))
        # Reconstruct by hand: mean direction plus the 8 shared features of largest summed deviation.
        unit = torch.nn.functional.normalize(x, dim=-1).reshape(1, 2, 16, 4, 128)
        mean = unit.mean(-2, keepdim=True)
        spread = unit - mean
        index = spread.square().sum(-2, keepdim=True).topk(8, dim=-1).indices.expand(1, 2, 16, 4, 8)
        recon = mean + torch.zeros_like(spread).scatter(-1, index, spread.gather(-1, index))
        expected = torch.nn.functional.normalize(recon, dim=-1).reshape(1, 2, 64, 128) * x.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_query_selection_lowers_query_weighted_error(self):
        x = correlated_keys(length=128, seed=11)
        weights = torch.rand(2, 128) ** 4 + .01  # a few dimensions dominate attention
        common = dict(rope_tables=ROPE, menu=['D', 'Q8', 'P8+8', 'S8'], distortion='query', weights=weights)
        plain = group_rd.build_menu(x, **common)
        aware = group_rd.build_menu(x, select_by='query', **common)
        for column in range(1, 4):
            self.assertLess(float(aware.distortion[..., column].sum()), float(plain.distortion[..., column].sum()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'w.pt'
            torch.save({'weights': weights.expand(3, 2, 128).clone()}, path)
            group_rd._load_weights.cache_clear()
            group_rd.apply(x, layer_idx=1, target='k', saving=.5, select_by='query', query_weights=str(path),
                           rope_tables=ROPE)
            with self.assertRaisesRegex(ValueError, 'values have no query'):
                group_rd.apply(x, layer_idx=1, target='v', saving=.5, select_by='query', query_weights=str(path))
            with self.assertRaisesRegex(ValueError, 'needs query_weights'):
                group_rd.apply(x, layer_idx=1, target='k', saving=.5, select_by='query', rope_tables=ROPE)


class ValueTests(unittest.TestCase):
    def setUp(self):
        pair_gate.reset_stats()
        group_rd.PRICES.clear()

    def test_values_need_no_rope_and_are_not_aligned(self):
        x = correlated_keys(length=128, seed=5)
        out = group_rd.apply(x, layer_idx=1, target='v', saving=.4, distortion='squared')
        self.assertEqual(out.shape, x.shape)
        # Value pairs use the stored vectors directly, so a constant run of values merges exactly.
        same = torch.randn(1, 2, 1, 128).expand(1, 2, 64, 128).clone()
        merged = group_rd.apply(same, layer_idx=1, target='v', lam=1e9)
        torch.testing.assert_close(merged, same, atol=1e-5, rtol=1e-5)
        with_rope = group_rd.build_menu(same, rope_tables=ROPE).distortion
        without = group_rd.build_menu(same, rope_tables=None).distortion
        self.assertGreater(float(with_rope[..., 1:].min()), 0)  # rotating unequal positions would cost
        self.assertLess(float(without[torch.isfinite(without)].max()), 1e-5)  # zero up to rounding

    def test_value_accounting_and_separate_slot_prices(self):
        x = correlated_keys(length=256, seed=6)
        group_rd.apply(x, layer_idx=2, target='k', saving=.3, rope_tables=ROPE)
        group_rd.apply(x * 3, layer_idx=2, target='v', saving=.5, distortion='squared')
        stats = pair_gate.pop_stats()
        for target, saving in (('k', .3), ('v', .5)):
            row = stats[f'{target}_layer_2']
            achieved = 1 - row['stored_bits'] / row['dense_bits']
            self.assertGreaterEqual(achieved, saving)
            self.assertLess(achieved, saving + .02)
            self.assertEqual(row['metadata_bits'], metadata_bits(2 * 64, 3))
        self.assertIn(('k', 2), group_rd.PRICES)
        self.assertIn(('v', 2), group_rd.PRICES)
        self.assertNotEqual(group_rd.PRICES['k', 2], group_rd.PRICES['v', 2])

    def test_squared_value_error_tracks_magnitude(self):
        # Squared error weights large values more; cosine ignores scale.
        x = correlated_keys(length=64, seed=7)
        scaled = x.clone()
        scaled[:, 1] *= 10
        cosine = group_rd.build_menu(scaled, rope_tables=None, distortion='cosine').distortion
        squared = group_rd.build_menu(scaled, rope_tables=None, distortion='squared').distortion
        finite = torch.isfinite(squared)
        ratio = squared[:, 1][finite[:, 1]].mean() / squared[:, 0][finite[:, 0]].mean()
        self.assertGreater(float(ratio), 20)
        self.assertLess(float(cosine[:, 1][finite[:, 1]].mean() / cosine[:, 0][finite[:, 0]].mean()), 5)

    def test_kv_pipeline_parses_with_both_targets(self):
        step = {'method': 'group_rd', 'k_layers': [0, 3], 'v_layers': [1], 'saving': .4,
                'distortion': 'squared', 'menu': list(group_rd.DEFAULT_MENU)}
        parse_kv_spec({'pipeline': [step]})


class DecodeTests(unittest.TestCase):
    def setUp(self):
        pair_gate.reset_stats()
        group_rd.PRICES.clear()

    def decode(self, x, prefill, steps, target='k', **options):
        """Prefill ``prefill`` tokens, then append the rest in ``steps``-token updates."""
        cache = group_rd.apply(x[..., :prefill, :], layer_idx=5, target=target, rope_tables=ROPE, **options)
        position = prefill
        while position < x.shape[-2]:
            end = min(position + steps, x.shape[-2])
            new = group_rd.apply(x[..., position:end, :], layer_idx=5, target=target, seq_start=position,
                                 rope_tables=ROPE, **options)
            self.assertTrue(torch.equal(new, x[..., position:end, :]))  # decode input is untouched
            cache = torch.cat((cache, new), dim=-2)
            group_rd.after_append(cache, target=target, layer_idx=5, start=position, end=end,
                                  rope_tables=ROPE, **options)
            position = end
        return cache

    def test_frozen_price_decode_matches_whole_sequence(self):
        x = correlated_keys(length=200, seed=3)
        cases = itertools.product('kv', (4, 16, 64), ((130, 1), (128, 3), (70, 5)))
        for target, granularity, (prefill, steps) in cases:
            with self.subTest(target=target, granularity=granularity, prefill=prefill):
                pair_gate.reset_stats()
                group_rd.PRICES.clear()
                cache = self.decode(x, prefill, steps, target=target, saving=.4, granularity=granularity)
                lam = group_rd.PRICES[target, 5]
                stats = pair_gate.pop_stats()[f'{target}_layer_5']
                closed = x.shape[-2] // granularity * granularity
                whole = group_rd.apply(x, layer_idx=5, target=target, lam=lam, granularity=granularity,
                                       rope_tables=ROPE)
                pair_gate.reset_stats()
                torch.testing.assert_close(cache[..., :closed, :], whole[..., :closed, :], atol=0, rtol=0)
                self.assertTrue(torch.equal(cache[..., closed:, :], x[..., closed:, :]))  # open unit dense
                # Every token counted dense once; closed units at their mode bits plus codes.
                menu = group_rd.build_menu(x[..., :closed, :], rope_tables=ROPE if target == 'k' else None,
                                           menu=group_rd.DEFAULT_MENU)
                choice = group_rd.select(menu, lam, granularity)
                prefill_units = 2 * (prefill // granularity)
                decode_units = 2 * (closed // granularity) - prefill_units
                expected = (float(menu.bits[choice].sum()) + metadata_bits(prefill_units, 3)
                            + 3 * decode_units + (x.shape[-2] - closed) * 2 * 128 * 16)
                self.assertEqual(stats['dense_bits'], x.numel() * 16)
                self.assertAlmostEqual(stats['stored_bits'], expected, delta=len(range(prefill, 200, steps)) + 1)

    def test_fixed_lam_decode_and_dense_decode(self):
        x = correlated_keys(length=96, seed=4)
        cache = self.decode(x, 41, 1, lam=1e-4)
        whole = group_rd.apply(x, layer_idx=5, target='k', lam=1e-4, rope_tables=ROPE)
        torch.testing.assert_close(cache, whole, atol=0, rtol=0)
        kept = self.decode(x, 41, 1, lam=1e-4, decode=False)
        self.assertTrue(torch.equal(kept[..., 40:, :], x[..., 40:, :]))
        group_rd.PRICES.clear()
        with self.assertRaisesRegex(ValueError, 'prefill'):
            group_rd.after_append(x, target='k', layer_idx=9, start=90, end=96, rope_tables=ROPE, saving=.3)


class OfflineAnalysisTests(unittest.TestCase):
    def test_envelope_interpolates_lower_hull(self):
        values = offline.envelope([0., .2, .4, .5], [0., 3., 4., 10.], [.1, .3, .45, .6])
        # (0.2, 3) lies above the chord from (0, 0) to (0.4, 4), so time sharing beats it.
        self.assertAlmostEqual(values[0], 1.)
        self.assertAlmostEqual(values[1], 3.)
        self.assertAlmostEqual(values[2], 7.)
        self.assertIsNone(values[3])

    def test_distinct_modes_per_page(self):
        choice = torch.zeros(1, 1, 32, dtype=torch.long)
        choice[..., 16:20] = 3
        choice[..., 20] = 5
        self.assertEqual(offline.distinct_per_page(choice), [1, 3])

    def test_plan_only_writes_nothing_but_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = offline.plan(Path(tmp))
            self.assertEqual(len(manifest['modes']), 41)
            self.assertEqual(manifest['sampling'], {'pool_chunks': 80, 'offset': 0, 'chunks': 16,
                                                    'seed': 0, 'seq_len': 2048})
            self.assertIn('pairs_r32', manifest['configs'])
            self.assertIn('default_g64', manifest['configs'])
            self.assertEqual(manifest['default_menu'], list(group_rd.DEFAULT_MENU))

    def test_analyze_end_to_end_on_synthetic_keys_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = offline.plan(out, chunks=2)
            keys = Path(manifest['keys_dir'])
            keys.mkdir()
            for chunk in range(2):
                layers = torch.cat([correlated_keys(heads=2, length=130, noise=.2 + .3 * layer, seed=chunk * 7 + layer)
                                    for layer in range(2)])
                values = torch.cat([correlated_keys(heads=2, length=130, noise=.5, seed=100 + chunk * 7 + layer)
                                    for layer in range(2)]) * 2
                torch.save({'keys': layers.to(torch.bfloat16), 'values': values.to(torch.bfloat16), 'chunk': chunk},
                           keys / f'chunk_{chunk:05d}.pt')
            torch.save({'weights': torch.rand(2, 2, 128) + .1}, manifest['query_weights'])
            Path(manifest['rope']).write_text(json.dumps({'rope_theta': 10000., 'head_dim': 128,
                                                          'inv_freq': None, 'attention_scaling': 1.}))
            offline.analyze(manifest)
            report = json.loads((out / 'stage_a.json').read_text())
            self.assertEqual(set(report['top_menus']), {'k', 'v'})
            for menus in report['top_menus'].values():
                self.assertEqual({k: len(v) for k, v in menus.items()}, {'6': 6, '8': 8})
                self.assertTrue(all(menu[0] == 'D' for menu in menus.values()))
            self.assertEqual(set(report['layers']), {'k/cosine', 'k/query', 'v/cosine', 'v/squared'})
            better = 0
            for kind in report['layers']:
                for row in report['layers'][kind].values():
                    for i, target in enumerate(report['targets']):
                        adaptive, fixed = row['full_g4']['distortion'][i], row['best_fixed']['distortion'][i]
                        if adaptive is not None and fixed is not None:
                            # The full menu contains every fixed menu; it only pays a wider mode code
                            # (6 bits per block instead of 1 per group), so it is never much worse.
                            self.assertLessEqual(adaptive, fixed * 1.05 + 1e-9)
                            better += adaptive < fixed * .99
                        small = row['default_g4']['distortion'][i]
                        if adaptive is not None and small is not None:
                            self.assertLessEqual(adaptive, small * 1.05 + 1e-9)  # default menu is a subset
                        coarse = row['full_g64']['distortion'][i]
                        if adaptive is not None and coarse is not None:
                            self.assertLessEqual(adaptive, coarse * 1.05 + 1e-9)
            self.assertGreater(better, 0)
            text = (out / 'stage_a.md').read_text()
            self.assertTrue(text.startswith('# group_rd stage A'))
            self.assertIn('## values, distortion squared', text)



class TaskStudyTests(unittest.TestCase):
    def test_plan_arms_differ_only_in_key_residual_selection(self):
        from compression_topics.spatial.scripts import group_rd_tasks_study as study
        with tempfile.TemporaryDirectory() as tmp:
            manifest = study.plan(Path(tmp))
        configs = manifest['run']['configurations']
        self.assertEqual(len(configs), 2 + 2 * 3 * 2)
        by_name = {c['name']: c for c in configs}
        for budget in (30, 40, 50):
            plain = by_name[f'plain_b{budget}_ceval']['kv']['pipeline']
            attention = by_name[f'attention_b{budget}_ceval']['kv']['pipeline']
            parse_kv_spec({'pipeline': plain})
            self.assertEqual(plain[0]['select_by'], 'deviation')
            self.assertEqual(attention[0]['select_by'], 'query')
            self.assertEqual({**plain[0], 'select_by': None}, {**attention[0], 'select_by': None})
            self.assertEqual(plain[1], attention[1])
            self.assertEqual((plain[0]['k_layers'], plain[0]['v_layers']), ('all', []))
            self.assertEqual((plain[1]['k_layers'], plain[1]['v_layers']), ([], 'all'))
            self.assertEqual(plain[1]['distortion'], 'squared')
            self.assertAlmostEqual(plain[0]['saving'], budget / 100)



class ImportanceTests(unittest.TestCase):
    def test_attention_received_matches_dense_softmax_and_query_split(self):
        torch.manual_seed(12)
        length = 101
        q, k = torch.randn(1, 8, length, 16), torch.randn(1, 2, length, 16)
        got = offline.attention_received(q, k, block=32)
        scores = q[0] @ k[0].repeat_interleave(4, 0).transpose(-1, -2) / 4
        probs = scores.masked_fill(torch.triu(torch.ones(length, length, dtype=torch.bool), 1), float('-inf')).softmax(-1)
        visible = torch.tril(torch.ones(length, length))
        for part, rows in enumerate((torch.ones(length, dtype=torch.bool), torch.arange(length) % 2 == 0,
                                     torch.arange(length) % 2 == 1)):
            expected = (probs[:, rows].sum(1) / visible[rows].sum(0).clamp_min(1)).reshape(2, 4, length).sum(1)
            torch.testing.assert_close(got[part], expected, atol=1e-5, rtol=1e-5)

    def test_token_weights_scale_each_tokens_error(self):
        x = correlated_keys(length=64, seed=13)
        weights = torch.rand(1, 2, 64) + .5
        plain = group_rd.build_menu(x, rope_tables=None, distortion='squared', menu=['D', 'Q8', 'P8+8'])
        weighted = group_rd.build_menu(x, rope_tables=None, distortion='squared', menu=['D', 'Q8', 'P8+8'],
                                       token_weights=weights)
        ones = group_rd.build_menu(x, rope_tables=None, distortion='squared', menu=['D', 'Q8', 'P8+8'],
                                   token_weights=torch.ones(1, 2, 64))
        torch.testing.assert_close(ones.distortion, plain.distortion)
        self.assertFalse(torch.allclose(weighted.distortion, plain.distortion))
        self.assertTrue(torch.equal(weighted.distortion[..., 0], plain.distortion[..., 0]))  # dense stays exact



class OrphanTests(unittest.TestCase):
    def test_child_exits_when_its_launcher_dies(self):
        import os
        import signal
        import subprocess
        import sys
        import time
        from engine.eval_runner.workers import PARENT_ENV
        child = ("import os,sys,time;sys.path.insert(0,os.getcwd());"
                 "from engine.eval_runner.workers import exit_with_parent;"
                 "exit_with_parent(.1);time.sleep(30)")
        launcher = ("import os,subprocess,sys,time;"
                    f"p=subprocess.Popen([sys.executable,'-c',{child!r}],env=dict(os.environ,{PARENT_ENV}=str(os.getpid())));"
                    "print(p.pid,flush=True);time.sleep(30)")
        parent = subprocess.Popen([sys.executable, '-c', launcher], stdout=subprocess.PIPE, text=True,
                                  cwd=str(Path(__file__).resolve().parents[1]))
        pid = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGKILL)  # like pkill on a test runner
        parent.wait()
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.1)
        else:
            os.kill(pid, signal.SIGKILL)
            self.fail("orphaned child kept running")


if __name__ == '__main__':
    unittest.main()
