"""Selection must be fold-local and preserve a readable feature mapping."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'notebooks/leonard'))
from identity_features import CAT_KEYS
from run_feature_selection_experiment import make_selection_model,semantic_feature_names

class FeatureSelectionTests(unittest.TestCase):
    def test_kbest_and_rfe_use_training_vocabulary_and_keep_requested_count(self):
        rng=np.random.default_rng(42)
        y=np.tile([0,1],40)
        X=pd.DataFrame([{'signal':float(label)+float(rng.normal(scale=.03)),'noise':float(rng.normal()),'identity_numeric':{'ordinary':float(i%3)},
                         **dict.fromkeys(CAT_KEYS,'same')} for i,label in enumerate(y)])
        val=X.iloc[60:].copy();val['identity_numeric']=[{'ordinary':1.,'validation_only':100.} for _ in range(len(val))]
        for method in ('kbest','rfe'):
            with self.subTest(method=method):
                p=make_selection_model(['signal','noise'],method,2)
                p.fit(X.iloc[:60],y[:60])
                self.assertEqual(p.named_steps['feature_selection'].get_support().sum(),2)
                self.assertEqual(len(semantic_feature_names(p,['signal','noise'])),2)
                self.assertIn('signal',semantic_feature_names(p,['signal','noise']))
                self.assertFalse(any('validation_only' in f for f in p.named_steps['encoding'].vectorizer_.feature_names_))
                self.assertEqual(p.predict(val).shape,(20,))

if __name__=='__main__':unittest.main()
