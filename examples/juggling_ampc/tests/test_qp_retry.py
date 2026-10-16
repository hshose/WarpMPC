"""A rejected QP must retain projected ADMM z as well as x and dual y."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
from qp import NativeQP


class RetryTests(unittest.TestCase):
    def test_two_retries_equal_one_longer_solve_in_both_coordinate_systems(self):
        # Small deliberately unfinished QP: continuing the projected auxiliary
        # variable differs from restarting it as A*x on the second call.
        P=sp.csc_matrix([[4.,1.],[0.,2.]])
        A=sp.csc_matrix([[1.,1.],[1.,0.],[0.,1.]])
        with jax.enable_x64(True):
            for incremental in ('0','1'):
                with self.subTest(incremental=incremental),patch.dict(os.environ,{
                    'JUGGLING_QP_INCREMENTAL':incremental,
                    'JUGGLING_QP_COMPENSATED_RESIDUAL':'0',
                    'JUGGLING_QP_COMPENSATED_PRODUCTS':'0'}):
                    short=NativeQP(P,A,rho=.001,iterations=3,backend='jax',dtype=np.float64)
                    long=NativeQP(P,A,rho=.001,iterations=6,backend='jax',dtype=np.float64)
                    p=jnp.asarray(P.data[None]);a=jnp.asarray(A.data[None])
                    q=jnp.array([[-2.,-3.]])
                    l=jnp.array([[.2,-.1,-.1]]);u=jnp.array([[.2,.4,.5]])
                    x=jnp.array([[.15,.05]]);y=jnp.zeros((1,3))
                    options={}
                    if incremental=='1':
                        symmetric=P.toarray()+P.toarray().T-np.diag(P.diagonal())
                        options=dict(gradient=q+x@symmetric.T,
                            relative_lower=l-x@A.toarray().T,relative_upper=u-x@A.toarray().T)
                    _,_,state=short.solve(p,a,q,l,u,x,y,return_state=True,**options)
                    actual=short.solve(p,a,q,l,u,x,y,retry_state=state,retry_mask=jnp.array([True]),**options)
                    expected=long.solve(p,a,q,l,u,x,y,**options)
                    for v,w in zip(actual,expected):np.testing.assert_allclose(v,w,atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()
