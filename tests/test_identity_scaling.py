"""The follow-up must not amplify rare identity indicators when scaling."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks/leonard'))
from identity_features import CAT_KEYS
from run_identity_scaling_followup import make_followup

class IdentityScalingTests(unittest.TestCase):
    def test_only_baseline_columns_are_standardized(self):
        X=pd.DataFrame([{'base':b,'identity_numeric':{'rare':r},**dict.fromkeys(CAT_KEYS,'same')} for b,r in [(-10.,0.),(10.,1.)]])
        pipeline=make_followup(['base'],.1)
        encoded=pipeline.named_steps['encoding'].fit_transform(X)
        scaled=pipeline.named_steps['scaling'].fit_transform(encoded)
        np.testing.assert_allclose(scaled[:,0],[-1.,1.])
        np.testing.assert_array_equal(scaled[:,1:],encoded.iloc[:,1:].to_numpy())

if __name__=='__main__':unittest.main()
