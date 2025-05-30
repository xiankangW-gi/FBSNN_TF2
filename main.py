from HJB import HamiltonJacobiBellman
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("Starting improved 100-dimensional Hamilton-Jacobi-Bellman equation solver...")
    print("Using TensorFlow", tf.__version__)

    
    M = 100 
    N = 50 
    D = 100 
    layers = [D+1] + 4*[256] + [1]  
    Xi = np.zeros([1,D])
    T = 1.0

    
    print("\nInitializing model...")
    model = HamiltonJacobiBellman(Xi, T, M, N, D, layers)

    total_params = sum([tf.size(var).numpy() for var in model.model.trainable_weights])
    print(f"Model architecture: {layers}")
    print(f"Total parameters: {total_params}")

    
    print("\nPhase 1: Initial training (LR: 1e-3)")
    model.train(N_Iter=5000, learning_rate=1e-3)

    print("\nPhase 2: Fine-tuning (LR: 5e-4)")
    model.train(N_Iter=5000, learning_rate=5e-4)

    print("\nPhase 3: Final refinement (LR: 1e-4)")
    model.train(N_Iter=5000, learning_rate=1e-4)

    print("\nTraining completed! Generating test results...")

  
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    
    def u_exact(t, X):
        MC = 10000  
        NC = t.shape[0]
        W = np.random.normal(size=(MC, NC, D))
        X_expanded = np.expand_dims(X, 0)
        t_expanded = np.expand_dims(t.squeeze(), 0)
        sqrt_term = np.sqrt(2.0 * np.abs(T - t_expanded))[:, :, np.newaxis]
        X_terminal = X_expanded + sqrt_term * W
        X_squared = np.sum(X_terminal**2, axis=2, keepdims=True)
        g_values = np.log(0.5 + 0.5 * X_squared)
        u_values = -np.log(np.mean(np.exp(-g_values), axis=0))
        return u_values.squeeze()

    print("Computing exact solution for comparison...")
    Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:])
    Y_test_terminal = np.log(0.5 + 0.5*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))

    
    print(f"\nResults after training:")
    print(f"Initial value Y0 (learned): {Y_pred[0,0,0]:.6f}")
    print(f"Initial value Y0 (exact): {Y_test[0]:.6f}")
    initial_error = abs(Y_pred[0,0,0] - Y_test[0])
    relative_error = initial_error / abs(Y_test[0])
    print(f"Absolute error: {initial_error:.6f}")
    print(f"Relative error: {relative_error:.6f} ({relative_error*100:.3f}%)")

   
    plt.figure(figsize=(12, 8))
    plt.plot(t_test[0:1,:,0].T, Y_pred[0:1,:,0].T, 'b-', linewidth=2, label='Learned $u(t,X_t)$')
    plt.plot(t_test[0,:,0].T, Y_test, 'r--', linewidth=2, label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1,-1,0], Y_test_terminal[0:1,0], 'ks', markersize=8, label='$Y_T = u(T,X_T)$')
    plt.plot([0], Y_test[0], 'ko', markersize=8, label='$Y_0 = u(0,X_0)')
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$Y_t = u(t,X_t)$', fontsize=14)
    plt.title('100-dimensional Hamilton-Jacobi-Bellman Equation Solution', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

  
    errors = np.abs(Y_test - Y_pred[0,:,0]) / np.abs(Y_test)
    plt.figure(figsize=(12, 6))
    plt.plot(t_test[0,:,0], errors, 'b-', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('Relative Error', fontsize=14)
    plt.title('100-dimensional Hamilton-Jacobi-Bellman: Relative Error Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    print(f"Max relative error: {np.max(errors):.6f}")
    print(f"Mean relative error: {np.mean(errors):.6f}")