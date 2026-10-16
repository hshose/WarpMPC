"""Padding bounds must not change sparse LDL arithmetic or solve scheduling."""
import importlib.util
import unittest
import jax
import numpy as np
import scipy.sparse as sp
import qdldl
from warpmpc.jax_qdldl.core import build_qdldl_plan, compile_qdldl_variant


@unittest.skipUnless(importlib.util.find_spec('warp'), 'optional Warp dependency')
class WarpPaddingTests(unittest.TestCase):
    def test_uneven_columns_against_independent_qdldl(self):
        jax.config.update('jax_enable_x64', True)
        rng=np.random.default_rng(163)
        r=rng.normal(size=(16,16));s=rng.normal(size=(8,8))
        matrix=sp.block_diag((sp.csc_matrix(r.T@r+np.eye(16)),s.T@s+np.eye(8),np.diag(np.arange(1.,9.))),format='csc')
        plan=build_qdldl_plan(matrix)
        rhs=rng.normal(size=(3,32))
        values=np.stack([plan.a_data*factor for factor in (1.,.7,1.3)])
        for dtype in (np.float32,np.float64):
            for segmented,level,threshold in ((False,False,1),(True,True,1),(True,True,4),(True,True,32)):
                with self.subTest(dtype=dtype,segmented=segmented,level=level,threshold=threshold):
                    compiled=compile_qdldl_variant(plan,dtype=dtype,backend='warp',transpose_work=True,
                        segmented=segmented,segment_budget=8,segment_strategy='optimal',
                        level_scheduled_solve=level,level_scheduled_solve_threshold=threshold)
                    tolerance=2e-5 if dtype==np.float32 else 1e-10
                    x,lx,d=map(lambda value:np.asarray(value).T,compiled.factor_and_solve(values.astype(dtype).T,rhs.astype(dtype).T))
                    for i in range(3):
                        matrix=sp.csc_matrix((values[i].astype(dtype),plan.a_indices,plan.a_indptr),shape=(32,32))
                        reference=qdldl.Solver(matrix,upper=True);lref,dref,pref=reference.factors()
                        np.testing.assert_array_equal(pref,plan.p)
                        np.testing.assert_allclose(lx[i],lref.data,rtol=tolerance,atol=tolerance)
                        np.testing.assert_allclose(d[i],dref,rtol=tolerance,atol=tolerance)
                        np.testing.assert_allclose(x[i],reference.solve(rhs[i].astype(dtype)),rtol=tolerance,atol=tolerance)


if __name__=='__main__':unittest.main()
