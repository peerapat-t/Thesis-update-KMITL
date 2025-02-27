import numpy as np
import pandas as pd

def sgd_mf(Y, K, epochs):
    user, item = Y.shape
    P = np.random.rand(user, K)
    Q = np.random.rand(item, K)
    for epoch in range(epochs):
        sum_sq_error = 0
        for u in range(user):
            for i in range(item):
                # Compute predicted rating y_hat_ui
                p_u = P[u, :]
                q_i = Q[i, :]
                y_hat_ui = p_u @ q_i
                y_ui = Y[u, i]
                if y_ui > 0:
                    # Error of p_u
                    e_pu_1 = (y_ui - y_hat_ui) * q_i
                    e_pu_6 = 0.01 * (p_u)
                    # Error of q_i
                    e_qi_1 = (y_ui - y_hat_ui) * p_u
                    e_qi_6 = 0.01 * (q_i)
                    # Update using GD
                    P[u, :] += 0.0001 * (e_pu_1 - e_pu_6)
                    Q[i, :] += 0.0001 * (e_qi_1 - e_qi_6)
                    # Compute squared error
                    sum_sq_error += (y_ui - y_hat_ui) ** 2
        print(f"Epoch {epoch+1}/{epochs}, SSE: {sum_sq_error:.8f}", end="\r")
    return np.dot(P,Q.T)

def sgd_mmf(Y, MU, K, epochs):
    user, item = Y.shape
    P = np.random.rand(user, K)
    Q = np.random.rand(item, K)
    for epoch in range(epochs):
        sum_sq_error = 0
        for u in range(user):
            for i in range(item):
                # Compute predicted rating y_hat_ui
                p_u = P[u, :]
                q_i = Q[i, :]
                y_hat_ui = p_u @ q_i
                y_ui = Y[u, i]
                if y_ui > 0:
                    # Error of p_u
                    e_pu_1 = (y_ui - y_hat_ui) * q_i
                    e_pu_4 = 1.0 * (0.5 * q_i * MU[i])
                    e_pu_6 = 0.01 * (p_u)
                    # Error of q_i
                    e_qi_1 = (y_ui - y_hat_ui) * p_u
                    e_qi_4 = 1.0 * (0.5 * p_u * MU[i])
                    e_qi_6 = 0.01 * (q_i)
                    # Update using GD
                    P[u, :] += 0.0001 * (e_pu_1 + e_pu_4 - e_pu_6)
                    Q[i, :] += 0.0001 * (e_qi_1 + e_pu_4 - e_qi_6)
                    # Compute squared error
                    sum_sq_error += (y_ui - y_hat_ui) ** 2
        print(f"Epoch {epoch+1}/{epochs}, SSE: {sum_sq_error:.8f}", end="\r")
    return np.dot(P,Q.T)

def sgd_wmf(Y, K, epochs):
    user, item = Y.shape
    P = np.random.rand(user, K)
    Q = np.random.rand(item, K)
    C = 1 + (40 * Y)
    for epoch in range(epochs):
        sum_sq_error = 0
        for u in range(user):
            for i in range(item):
                # Compute predicted rating y_hat_ui
                p_u = P[u, :]
                q_i = Q[i, :]
                y_hat_ui = p_u @ q_i
                y_ui = Y[u, i]
                c_ui = C[u, i]
                # Error of p_u
                e_pu_1 = c_ui * (y_ui - y_hat_ui) * q_i
                e_pu_6 = 0.01 * (p_u)
                # Error of q_i
                e_qi_1 = c_ui * (y_ui - y_hat_ui) * p_u
                e_qi_6 = 0.01 * (q_i)
                # Update using GD
                P[u, :] += 0.0001 * (e_pu_1 - e_pu_6)
                Q[i, :] += 0.0001 * (e_qi_1 - e_qi_6)
                # Compute squared error
                sum_sq_error += (y_ui - y_hat_ui) ** 2
        print(f"Epoch {epoch+1}/{epochs}, SSE: {sum_sq_error:.8f}", end="\r")
    return np.dot(P,Q.T)

def sgd_mwmf(Y, MU, K, epochs):
    user, item = Y.shape
    P = np.random.rand(user, K)
    Q = np.random.rand(item, K)
    C = 1 + (40 * Y)
    for epoch in range(epochs):
        sum_sq_error = 0
        for u in range(user):
            for i in range(item):
                # Compute predicted rating y_hat_ui
                p_u = P[u, :]
                q_i = Q[i, :]
                y_hat_ui = p_u @ q_i
                y_ui = Y[u, i]
                c_ui = C[u, i]
                # Error of p_u
                e_pu_1 = c_ui * (y_ui - y_hat_ui) * q_i
                e_pu_4 = 1.0 * (0.5 * q_i * MU[i])
                e_pu_6 = 0.01 * (p_u)
                # Error of q_i
                e_qi_1 = c_ui * (y_ui - y_hat_ui) * p_u
                e_qi_4 = 1.0 * (0.5 * p_u * MU[i])
                e_qi_6 = 0.01 * (q_i)
                # Update using GD
                P[u, :] += 0.0001 * (e_pu_1 + e_pu_4 - e_pu_6)
                Q[i, :] += 0.0001 * (e_qi_1 + e_pu_4 - e_qi_6)
                # Compute squared error
                sum_sq_error += (y_ui - y_hat_ui) ** 2
        print(f"Epoch {epoch+1}/{epochs}, SSE: {sum_sq_error:.8f}", end="\r")
    return np.dot(P,Q.T)