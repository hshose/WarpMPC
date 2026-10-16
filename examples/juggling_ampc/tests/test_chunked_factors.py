import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import casadi as ca
import jax
import numpy as np
from chunked_factors import convert_chunked

class ChunkedTests(unittest.TestCase):
    def test_reused_temporaries_multiple_outputs_and_batch(self):
        x=ca.SX.sym('x',4);u=ca.SX.sym('u',2)
        z=x
        for i in range(12):z=ca.sin(z+u[i%2])+.2*z
        f=ca.Function('chunk_fixture',[x,u],[z,ca.densify(ca.jacobian(z,x))])
        kernel=jax.jit(jax.vmap(convert_chunked(f,13)))
        rng=np.random.default_rng(5);xs=rng.normal(size=(7,4)).astype('float32');us=rng.normal(size=(7,2)).astype('float32')
        actual=kernel(xs,us)
        expected=[f(x,u) for x,u in zip(xs,us)]
        for i in range(2):np.testing.assert_allclose(actual[i],np.array([v[i] for v in expected]),atol=2e-5,rtol=2e-5)
        transposed=jax.jit(convert_chunked(f,13,batched=True))(xs,us)
        for i in range(2):np.testing.assert_allclose(transposed[i],actual[i],atol=2e-6,rtol=2e-6)

    def test_transposed_matrix_inputs_preserve_casadi_column_order(self):
        a=ca.SX.sym('a',2,3);b=ca.SX.sym('b',3,2)
        f=ca.Function('chunk_matrix_fixture',[a,b],[a@b,ca.sum1(a)])
        rng=np.random.default_rng(11);av=rng.normal(size=(5,2,3)).astype('float32');bv=rng.normal(size=(5,3,2)).astype('float32')
        actual=jax.jit(convert_chunked(f,7,batched=True))(av,bv)
        expected=[f(x,y) for x,y in zip(av,bv)]
        for i in range(2):np.testing.assert_allclose(actual[i],np.array([v[i] for v in expected]),atol=2e-6,rtol=2e-6)

if __name__=='__main__':unittest.main()
