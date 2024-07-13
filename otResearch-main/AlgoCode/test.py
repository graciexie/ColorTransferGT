import numpy as np
import algos

# Initialize variables for input
np.random.seed(123)
n = 100
reg = 1e-1
iterMax = 30000
eps = 2e-2
rng = np.random.RandomState(0)

# x is n by 2 matrix where entries are sampled from N(0,1) dist.
x = rng.randn(n,2)

# r is a random simplex vector and our source measure
r = np.random.exponential(scale=1.0, size=n)
r = r / sum(r)

# c is the uniform simplex vector
c = np.random.exponential(scale=1.0, size=n)
c = c / sum(c)

# M is a distance metric matrix
xp2 = np.sum(np.square(x), 1)
M = xp2.reshape((-1,1)) + xp2.reshape((1,-1)) - 2 * np.dot(x, x.T)
M /= M.max()

class TestAlgos():
    def __init__(self):
        # Calls each test
        self.test_sinkhorn()
        self.test_greenkhorn()
        self.test_stochSinkhorn()
        self.test_sag()
        self.test_apdagd()
        self.test_aam()
        self.test_apdrcd()

    def test_sinkhorn(self):
        P, t, o = algos.sinkhorn(r, c, M, reg, eps, iterMax)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        print(f"test_sinkhorn PASSED")
        print(f"Sinkhorn completed in {t}s and {o} operations.")

    def test_greenkhorn(self):
        P, t, o = algos.greenkhorn(r, c, M, reg, eps)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        print(f"test_greenkhorn PASSED")
        print(f"Greenkhorn completed in {t}s and {o} operations.")

    def test_stochSinkhorn(self):
        P, t, o = algos.stochSinkhorn(r, c, M, reg, eps)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        print(f"test_stochSinkhorn PASSED")
        print(f"Stochastic Sinkhorn completed in {t}s and {o} operations.")

    def test_sag(self):
        """
        Test to verify validity of our sag algorithm.
        """
        P, t, o = algos.sag(r, c, M, reg, eps)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        print(f"test_sag PASSED")
        print(f"SAG completed in {t}s and {o} operations.")

    def test_apdagd(self):
        P, t, o = algos.apdagd(r, c, M, eps)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        print(f"test_apdagd PASSED")
        print(f"APDAGD completed in {t}s and {o} operations.")

    def test_aam(self):
        P, t, o = algos.aam(r, c, M, eps)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        
        print(f"test_aam PASSED")
        print(f"AAM completed in {t}s and {o} operations.")

    def test_apdrcd(self):
        P, t, o = algos.apdrcd(r, c, M, eps)
        np.testing.assert_allclose(
            r, P.sum(1), atol=eps
        )
        np.testing.assert_allclose(
            c, P.sum(0), atol=eps
        )
        print(f"test_apdrcd PASSED")
        print(f"APDRCD completed in {t}s and {o} operations.")

if __name__ == '__main__':
    test = TestAlgos()