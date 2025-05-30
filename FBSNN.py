import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

np.random.seed(42)
tf.random.set_seed(42)

def sin_activation(x):
    return tf.sin(x)

class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers):
        self.Xi = Xi
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.layers = layers
        self.model = self.build_model(layers)
        self.optimizer = tf.keras.optimizers.Adam()
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )

    def build_model(self, layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            layers[1],
            input_dim=layers[0],
            activation=sin_activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ))
        for i in range(2, len(layers)-1):
            model.add(tf.keras.layers.Dense(
                layers[i],
                activation=sin_activation,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'
            ))
        model.add(tf.keras.layers.Dense(
            layers[-1],
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ))
        return model

    def net_u(self, t, X):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            inputs = tf.concat([t, X], 1)
            u = self.model(inputs, training=True)
        Du = tape.gradient(u, X)
        del tape
        if Du is None:
            Du = tf.zeros_like(X)
        return u, Du

    def Dg_tf(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        Dg = tape.gradient(g, X)
        if Dg is None:
            Dg = tf.zeros_like(X)
        return Dg

    @tf.function
    def loss_function(self, t, W, Xi):
        dt = self.T / self.N
        loss = 0.0
        t_curr, W_curr = t[:, 0, :], W[:, 0, :]
        X_curr = tf.tile(Xi, [self.M, 1])
        Y_curr, Z_curr = self.net_u(t_curr, X_curr)
        X_list = [X_curr]
        Y_list = [Y_curr]
        
        for n in range(self.N):
            t_next, W_next = t[:, n+1, :], W[:, n+1, :]
            dW = W_next - W_curr
            mu_term = self.mu_tf(t_curr, X_curr, Y_curr, Z_curr) * dt
            sigma_matrix = self.sigma_tf(t_curr, X_curr, Y_curr)
            sigma_term = tf.linalg.matvec(sigma_matrix, dW)
            X_next = X_curr + mu_term + sigma_term
            phi_term = self.phi_tf(t_curr, X_curr, Y_curr, Z_curr) * dt
            Z_sigma_term = tf.reduce_sum(Z_curr * sigma_term, axis=1, keepdims=True)
            Y_tilde = Y_curr + phi_term + Z_sigma_term
            Y_next, Z_next = self.net_u(t_next, X_next)
            loss += tf.reduce_mean(tf.square(Y_next - Y_tilde))
            t_curr, W_curr, X_curr, Y_curr, Z_curr = t_next, W_next, X_next, Y_next, Z_next
            X_list.append(X_curr)
            Y_list.append(Y_curr)
        
        loss += tf.reduce_mean(tf.square(Y_curr - self.g_tf(X_curr)))
        loss += tf.reduce_mean(tf.square(Z_curr - self.Dg_tf(X_curr)))
        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)
        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):
        T, M, N, D = self.T, self.M, self.N, self.D
        dt = T / N
        dt_array = np.zeros((M, N+1, 1), dtype=np.float32)
        dt_array[:,1:,:] = dt
        DW = np.zeros((M, N+1, D), dtype=np.float32)
        DW[:,1:,:] = np.sqrt(dt) * np.random.normal(size=(M, N, D)).astype(np.float32)
        t = np.cumsum(dt_array, axis=1)
        W = np.cumsum(DW, axis=1)
        return t, W

    @tf.function
    def train_step(self, t_batch, W_batch, Xi):
        with tf.GradientTape() as tape:
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, Xi)
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.model.trainable_weights])
            total_loss = loss + 1e-5 * l2_loss
        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss, Y0_pred, grad_norm

    def train(self, N_Iter, learning_rate):
        self.optimizer.learning_rate.assign(learning_rate)
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        for it in range(N_Iter):
            t_batch, W_batch = self.fetch_minibatch()
            t_batch = tf.constant(t_batch, dtype=tf.float32)
            W_batch = tf.constant(W_batch, dtype=tf.float32)
            Xi = tf.constant(self.Xi, dtype=tf.float32)
            loss_value, Y0_value, grad_norm = self.train_step(t_batch, W_batch, Xi)
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            if it % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Grad Norm: %.3e, Time: %.2f, LR: %.3e' %
                      (it, loss_value.numpy(), Y0_value.numpy(), grad_norm.numpy(),
                       elapsed, learning_rate))
                start_time = time.time()
            if it > 0 and it % 5000 == 0:
                current_lr = self.optimizer.learning_rate.numpy()
                new_lr = current_lr * 0.8
                self.optimizer.learning_rate.assign(new_lr)
                print(f"Reduced learning rate to {new_lr:.3e}")

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = tf.constant(Xi_star, dtype=tf.float32)
        t_star = tf.constant(t_star, dtype=tf.float32)
        W_star = tf.constant(W_star, dtype=tf.float32)
        loss, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        return X_star.numpy(), Y_star.numpy()

    ###########################################################################
    ############################# Abstract Methods ###########################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = self.M
        D = self.D
        return tf.zeros([M,D], dtype=tf.float32)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return tf.eye(D, batch_shape=[M], dtype=tf.float32)