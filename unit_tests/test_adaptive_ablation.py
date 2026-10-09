"""Controlled policy differences and small captured-vector proxy accounting."""
import tempfile
import unittest
from pathlib import Path
import torch
from compression_topics.spatial.algorithms import group_rd
from compression_topics.spatial.scripts import adaptive_ablation_study as study
from compression_topics.spatial.scripts import adaptive_ablation_quick as quick

class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)

    def test_policies_only_change_format_freedom_and_importance(self):
        p = study.study_plan(Path('/private/tmp/ablation-unit'), layers=2)
        for target in ('k', 'v'):
            a = study.policy(p, 0, target, .4, 'adaptive', 'all', 'digest')
            b = study.policy(p, 0, target, .4, 'importance', 'all', 'digest')
            self.assertEqual({k:v for k,v in a.items() if k != 'importance'},
                             {k:v for k,v in b.items() if k != 'importance'})
            self.assertEqual(a['importance'], 'none')
            self.assertEqual(b['importance'], 'prefill_attention')
            self.assertFalse(a['decode'])
            for family in study.FAMILIES:
                f = study.policy(p, 0, target, .4, 'fixed', family, 'digest')
                self.assertEqual({k:v for k,v in a.items() if k != 'menu'},
                                 {k:v for k,v in f.items() if k != 'menu'})
                self.assertTrue(set(f['menu']).issubset(a['menu']))
        self.assertEqual(len(study.MENU), 8)
        self.assertEqual(study.sampling(p, 'validate')['split'], 'validation')
        self.assertEqual(study.sampling(p, 'test')['split'], 'test')
        self.assertEqual(study.sampling(p, 'test')['scoring_prefix'], 1536)
        self.assertNotIn('scoring_prefix', study.sampling(p, 'weights'))
        screen, refine = (study.sampling(p, s) for s in ('screen', 'refine'))
        self.assertEqual(screen['offset'] + screen['chunks'], refine['offset'])
        for row in study.candidates(p, 'digest'):
            self.assertLess(row['budget'], study.format_ceiling(row['kv']['pipeline'][0]['menu']))

    def test_proxy_uses_separate_calibration_and_scores_all_choices(self):
        cal = torch.randn(1, 2, 64, 128)
        heldout = torch.randn_like(cal)
        opts = dict(rope_tables=None, residuals=(0,8,16,32), menu=study.MENU, distortion='squared')
        calibration = group_rd.build_menu(cal, **opts)
        plain = group_rd.build_menu(heldout, **opts)
        seen_weights = torch.ones(1, 2, 64)
        seen_weights[..., :16] = 10
        unseen_weights = seen_weights.flip(-1)
        seen = group_rd.build_menu(heldout, token_weights=seen_weights, **opts)
        score = group_rd.build_menu(heldout, token_weights=unseen_weights, **opts)
        rows = quick.test_slot(calibration, plain, score, seen, [.3,.4,.5])
        self.assertEqual(len(rows), 9)
        self.assertEqual({r['arm'] for r in rows}, set(study.ARMS))
        self.assertTrue(all(r['measured_saving'] >= r['budget'] for r in rows))
        for row in rows:
            self.assertGreaterEqual(row['attention_weighted_error'], 0)
            self.assertEqual(sum(row['mode_counts'].values()), 32)
            if row['arm'] == 'fixed':
                self.assertEqual(set(row['mode_counts']), set(study.FAMILIES[row['fixed_family']]))
        # An oracle using identical decision/scoring weights is not accidentally substituted.
        self.assertFalse(torch.equal(seen.distortion, score.distortion))
        enriched = [{**r, 'target':'v', 'layer':0} for r in rows]
        self.assertEqual(len(quick.summarize(enriched)), 9)

    def test_live_three_arm_continuation_has_no_suffix_leakage(self):
        import hashlib
        from transformers import LlamaConfig, LlamaForCausalLM
        from engine.eval_runner.chunks import evaluate_chunks
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=128, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, head_dim=128,
            max_position_embeddings=128)).eval()
        model.config._attn_implementation = 'eager'
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            weights=out/'query_weights.pt'
            torch.save({'weights':torch.ones(1,1,128)}, weights)
            digest=hashlib.sha256(weights.read_bytes()).hexdigest()
            p=study.study_plan(out,layers=1)
            ids=torch.arange(32)%64
            changed=ids.clone();changed[24:]=(changed[24:]+13)%64
            for arm in study.ARMS:
                spec={'pipeline':[study.policy(p,0,t,.4,arm,'quad16',digest) for t in ('k','v')]}
                first=evaluate_chunks(model,[ids],spec,[0],scoring_prefix=24)
                second=evaluate_chunks(model,[changed],spec,[0],scoring_prefix=24)
                self.assertEqual(first['storage']['targets']['kv']['compression'],
                                 second['storage']['targets']['kv']['compression'])
                self.assertGreaterEqual(first['storage']['targets']['kv']['compression'],.4)
                self.assertEqual(first['chunk_scores'][0]['tokens'],8)

if __name__ == '__main__':
    unittest.main()
