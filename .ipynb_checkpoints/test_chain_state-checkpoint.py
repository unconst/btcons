import matplotlib.pyplot as plt
from scipy.sparse import random
from sklearn.preprocessing import normalize
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def random_weight_matrix( n, density ):
    rows = []
    for _ in range( n ):
        row_i = random(1, n, density=density).A # Each row has density_A
        row_i = normalize(row_i, axis=1, norm='l1') # Each row sums to 1.
        rows.append(row_i)
    assert np.shape(row_i)[1] == n
    assert np.isclose(np.sum(row_i), 1.0, 0.001)
    
    W = np.concatenate(rows)
    assert np.shape(W)[0] == n and np.shape(W)[1] == n
    return W

def random_stake_vector( n ):
    S = np.random.uniform(0, 1, size=(n, 1))
    S = (S / S.sum())
    S = np.reshape(S, [1, n])
    S = np.transpose(S)
    return S

def sigmoid(x):
    return 1 / (1 + np.exp( -x ))

def get_gradient( n, W, R, S, A, C ):
    gradient = []
    for i in range(n):
        grad_i = 0.0
        for k in range(n):
            inner = 0.0
            for j in range(n):
                inner += A[j] * S[j] * C[j, k]
            grad_i += C[i, k] * R[k] * np.exp(inner) * sigmoid(inner) * sigmoid(inner) + W[i, k] * (sigmoid(inner) - 0.5)
        gradient.append(grad_i)
    return np.concatenate(gradient)

def cabal_decay(
        nA:int = 5,
        nB:int = 100,
        n_blocks:int = 2000,
        tau:float = 0.01,
        learning_rate:float = 0.05
    ):
    r""" Measures the proportion of stake held by a disjoint sub-graph ’cabal’ as a function of steps.
    Args:
        nA (int):
            Number of peers in the honest graph
        nB (int):
            Number of peers in the disjoint graph
        n_blocks (int):
            Number of blocks to run.
        temperature (float):
            temperature of the sigmoid activation function.
        tau (float):
            stake inflation rate.
        learning_rate (float):
            scaling term learning rate.
    Returns:
        history (list(float)):
            Dishonest graph proportion.
    """
    n = nA + nB
    
    # Randomized Matrix of weights. We create the subgraphs by concatenating two disjoint
    # and random square matrices.
    W = np.concatenate((
    np.concatenate((random_weight_matrix(nA, 0.9), np.zeros((nA,nB)) ), axis=1),
    np.concatenate((np.zeros((nB,nA)), random_weight_matrix(nB, 0.9)), axis=1)), axis=0) ; print ('W \n', W)
    
    # Randomized Vector of stake.
    # Subgraph A gets 0.51 distributed across values 1->n_a.
    # Subgraph B gets 0.49 distributed across values 1->n_b
    S = np.concatenate((
        random_stake_vector(nA)*0.51,
        random_stake_vector(nB)*0.49),
        axis=0
    ); print ('S \n', np.transpose(S))
    
    # Scaling vector. A is multiplied by S before each iteration
    # to attain our scaled stake.
    A = np.ones(n) ; print ('A \n', A)
    
    # Matrix of connectivity. We use the true absorbing markov chain calculation
    # In practice this is too difficult to compute and opt for a depth d cut off.
    Wdiag = np.zeros((n,n))
    Wdiag[np.diag_indices_from(W)] = np.diag(W)
    Q = W - Wdiag
    C = np.maximum(np.linalg.pinv(np.identity(n) - Q), np.zeros((n,n)))
    C = np.where(C > 0, 1, -1) ; print ('C \n', C)
    
    # Iterate through blocks
    history = []
    for block in range(n_blocks):
        # Scale the stake against our vector A.
        S_scaled = S * np.reshape(A, (n,1))#; print (’S * A \n’, S_scaled)
        # Compute the ranks W^t S.
        R = np.matmul(np.transpose(W), S_scaled) / np.sum(S_scaled) #; print (’R \n’, R)
        # Compute our trust scores sigmoid( (C^T S) * temperature )
        T = 1 /( 1 + np.exp(-np.matmul(np.transpose(C), S_scaled))) #; print (’T \n’, T)
        # Compute our chain loss.
        loss = -np.dot(np.transpose(R), (T - 0.5) ) # ; print (’loss \n’, loss[0][0])
        # Our loss is negative, so we emit more stake.
        if loss < 0.0:
            # Compute our incentive function.
            I = R * T
            # Distribute more stake.
            S = S + tau * (I/np.sum(I))
            # Measure the size of our cabal.
            S_Honest = np.sum(S[:nA])
            S_Cabal = np.sum(S[nB:])
            ratio = S_Cabal / (S_Honest + S_Cabal)
            history.append(ratio)
        # Our loss is positive, so we update our scaling terms.
        else:
            # Compute the gradient of our scaling terms.
            grad = get_gradient( n, W, R, S, A, C ) #; print (’Grad \n’, grad)
            # Update our scaling terms, being sure not to have negative values.
            A = np.minimum(np.maximum( A + learning_rate * grad, np.zeros_like(A) ), 1)
        
    return history

plt.plot(cabal_decay())
