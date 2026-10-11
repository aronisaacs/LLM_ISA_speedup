"""Query-aware decisions retain absolute key-error scale."""
import unittest
import torch
from compression_topics.spatial.algorithms.scoring import reconstruction_error
from compression_topics.spatial.algorithms import hierarchy

class QueryAbsoluteTests(unittest.TestCase):
    def test_exact_formula_and_scale_dependence(self):
        source=torch.tensor([[[[[2.,4.]]]]]);restored=source+torch.tensor([1.,2.])
        energy=torch.tensor([3.,5.])
        absolute=reconstruction_error(source,restored,distortion='query',query_energy=energy)
        self.assertEqual(absolute.item(),23.)
        doubled=reconstruction_error(2*source,2*restored,distortion='query',query_energy=energy)
        self.assertEqual(doubled.item(),92.)
        relative=reconstruction_error(source,restored)
        self.assertEqual(relative.item(),.25)
        torch.testing.assert_close(relative,reconstruction_error(2*source,2*restored))
    def test_merge_candidate_error_has_no_key_norm_denominator(self):
        torch.manual_seed(17);x=torch.randn(1,2,8,64);energy=torch.rand(1,2,1,1,64)+.1
        table,recon=hierarchy.build_table(x,target='v',selection='query',distortion='query',query_energy=energy,return_reconstruction=True)
        expected=((x.float()-recon[(2,16)].float()).square()*energy.reshape(1,2,1,64)).sum(-1).reshape(1,2,4,2).mean((0,3))*.25
        for head in range(2):
            for start in range(0,8,2):
                self.assertAlmostEqual(table.rows[0][head][(2,start)][1],expected[head,start//2].item(),places=3)
    def test_missing_weights_fail(self):
        with self.assertRaisesRegex(ValueError,'query energy'):
            reconstruction_error(torch.ones(1,2),torch.zeros(1,2),distortion='query')

if __name__=='__main__':unittest.main()
