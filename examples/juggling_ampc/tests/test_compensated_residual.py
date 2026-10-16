import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp
from compensated_residual import compile_residual

class ResidualTests(unittest.TestCase):
    def test_rectangular_constraint_product(self):
        matrix=sp.csc_matrix(np.array([[1e8,1.,-1e8],[.1,.2,.3]],np.float32))
        x=np.ones((2,3),np.float32);rhs=np.zeros((2,2),np.float32)
        values=np.tile(matrix.data,(2,1))
        actual=np.asarray(compile_residual(matrix,symmetric=False)(jnp.array(values),jnp.array(x),jnp.array(rhs)))
        expected=-(matrix.astype(float)@x.astype(float).T).T
        np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-7)
        self.assertEqual(actual[0,0],-1.)

    def test_product_roundoff_is_retained(self):
        upper=sp.csc_matrix(np.array([[.1]],np.float32))
        x=np.array([[.3]],np.float32);rhs=upper.data.reshape(1,1)*x
        actual=np.asarray(compile_residual(upper)(jnp.array(upper.data[None]),jnp.array(x),jnp.array(rhs)))
        expected=rhs.astype(float)-upper.data.astype(float).reshape(1,1)*x.astype(float)
        self.assertNotEqual(float(expected[0,0]),0.)
        np.testing.assert_array_equal(actual,expected.astype(np.float32))

    def test_cancellation_preserves_small_residual_in_float32(self):
        upper=sp.csc_matrix(np.array([[1e8,1.,-1e8],[0.,.001,3.],[0.,0.,1e8]],np.float32))
        values=np.stack([upper.data,upper.data*.5])
        x=np.ones((2,3),np.float32);rhs=np.zeros_like(x)
        actual=np.asarray(compile_residual(upper)(jnp.array(values),jnp.array(x),jnp.array(rhs)))
        expected=[]
        for v in values:
            u=sp.csc_matrix((v.astype(float),upper.indices,upper.indptr),shape=(3,3));full=u+u.T-sp.diags(u.diagonal());expected.append(-full@x[0].astype(float))
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)
        self.assertEqual(actual.dtype,np.float32)
        self.assertEqual(actual[0,0],-1.)

if __name__=='__main__':unittest.main()
