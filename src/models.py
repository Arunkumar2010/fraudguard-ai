import numpy as np

class HybridModel:
    def __init__(self, rf, xgb, meta):
        self.rf = rf
        self.xgb = xgb
        self.meta = meta
    
    def predict(self, X):
        rf_p = self.rf.predict_proba(X)[:,1]
        xgb_p = self.xgb.predict_proba(X)[:,1]
        meta_X = np.column_stack([rf_p, xgb_p])
        return self.meta.predict(meta_X)
    
    def predict_proba(self, X):
        rf_p = self.rf.predict_proba(X)[:,1]
        xgb_p = self.xgb.predict_proba(X)[:,1]
        meta_X = np.column_stack([rf_p, xgb_p])
        return self.meta.predict_proba(meta_X)
