import unittest
import tempfile
from pathlib import Path
import numpy as np
from experiment import uncertainty, outcomes, repetition_split, select, policy_error, correlation, discrimination, load_cache

class DiagnosticTests(unittest.TestCase):
    def test_covariance_trace(self):
        p=np.array([[[1.,0.]],[[0.,1.]]])
        self.assertAlmostEqual(uncertainty(p)['sampling_variance'][0],1.)
        self.assertAlmostEqual(uncertainty(p)['entropy'][0],np.log(2))
        self.assertAlmostEqual(uncertainty(p)['variation_ratio'][0],.5)
        self.assertEqual(uncertainty(p)['negative_margin'][0],0.)
        self.assertTrue(np.all(uncertainty(np.repeat(p[:1],3,axis=0))['sampling_variance']==0))
    def test_outcomes(self):
        p=np.array([[[.8,.2],[.6,.4]],[[.3,.7],[.1,.9]]])
        o=outcomes(p,np.array([0,1]))
        np.testing.assert_allclose(o['error'],[.5,.5])
        np.testing.assert_allclose(o['nll'],[-np.log(.8*.3)/2,-np.log(.4*.9)/2])
        np.testing.assert_allclose(o['brier'],[(.08+.98)/2,(.72+.02)/2])
    def test_independence(self):
        a,b=repetition_split(50)
        self.assertEqual((len(a),len(b)),(25,25))
        self.assertFalse(set(a)&set(b))
        self.assertEqual(sorted([*a,*b]),list(range(50)))
        a,b=repetition_split(7)
        self.assertEqual((len(a),len(b)),(3,4))
        with self.assertRaises(ValueError): repetition_split(3)
    def test_policy(self):
        selected=select(np.array([0.,2.,2.,1.]),np.array([0,2,1,3]))
        np.testing.assert_array_equal(selected,[False,False,True,False])
        low=np.array([0.,.4,.8,.2]); high=np.array([0.,.1,.2,.3])
        np.testing.assert_allclose(policy_error(low,high,selected),[0,.4,.2,.2])
        self.assertAlmostEqual(policy_error(low,high,selected).mean(),.2)
    def test_degenerate_scores(self):
        self.assertTrue(np.isnan(correlation(np.ones(5),np.arange(5))))
        self.assertTrue(np.isnan(discrimination(np.arange(5),np.ones(5),'auroc')))
        self.assertAlmostEqual(correlation(np.array([0,0,1,1]),np.array([0,0,1,1])),1.)
    def test_cache_validation(self):
        fixture=dict(probabilities=np.full((2,4,5,2),.5), labels=np.zeros(5,dtype=int),
                     node_ids=np.arange(5), k_values=np.array([16,128]), checkpoint=np.array('fixture'),
                     repeats=np.array(4), completed_k_index=np.array(1))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cache.npz'
            np.savez(path,**fixture)
            self.assertEqual(load_cache(path)[0].shape,(2,4,5,2))
            fixture['probabilities'][0,0,0,0]=np.nan
            np.savez(path,**fixture)
            with self.assertRaises(ValueError): load_cache(path)
            fixture['probabilities'][:]=.8
            np.savez(path,**fixture)
            with self.assertRaises(ValueError): load_cache(path)
            fixture['probabilities'][:]=.5
            fixture['node_ids'][1]=0
            np.savez(path,**fixture)
            with self.assertRaises(ValueError): load_cache(path)

    def test_score_label_free(self):
        p=np.array([[[.8,.2]],[[.6,.4]]])
        before=uncertainty(p)
        outcomes(p,np.array([1]))
        for key in before: np.testing.assert_array_equal(before[key],uncertainty(p)[key])

if __name__=='__main__': unittest.main()
