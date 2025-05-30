from FBSNN import FBSNN
import tensorflow as tf

class HamiltonJacobiBellman(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)

    def phi_tf(self, t, X, Y, Z):
        return tf.reduce_sum(Z**2, 1, keepdims=True)

    def g_tf(self, X):
        return tf.math.log(0.5 + 0.5*tf.reduce_sum(X**2, 1, keepdims=True))

    def sigma_tf(self, t, X, Y):
        return tf.sqrt(2.0) * super().sigma_tf(t, X, Y)