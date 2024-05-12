import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

def GMRES(apply_A, b, n, tol = 1e-15):
    """
    Performs Generalized Minimal Residues to find x that approximates the solution to Ax=b. 
    
    Iterative method that at step n uses Arnoldi iteration to find an orthogonal basis for Krylov subspace

    Parameters:
    A : m x m complex (possibly) Non-Hermitian matrix 
    b: m dim vector
    x0: initial guess
    n: maximum number of iterations for the algorithm (m being the largest possible)
    tol: threshold value for convergence

    Returns:
    x: m dim vector that minimizes the norm of Ax-b that lies in the Krylov subspace of dimension n (<m)
    e: error of each iteration, where the error is calculated as ||r_k||/||b|| where r_k is (b-A*x_k), the residue at 
    the k iteration.
    """

    #Residual vector from which to build Krylov subspace of A (Ar, A^2 r,.., A^n r)
    # if (dX0 == np.zeros_like(dX0)).all():
    #     r = b
    # else:
    #     r = b - apply_A(dX0, 0.)
    # #Define norms of vectors
    # b_norm = np.linalg.norm(b)

    # #Compute initial error and save in list
    # error = r_norm/b_norm

    r = b
    r_norm = b_norm = np.linalg.norm(r)
    e = [1.] #r_norm/b_norm

    #Initialize sine and cosine Givens 1d vectors. This allows the algorithm to be O(k) instead of O(k^2)
    sn = np.zeros(n)
    cs = np.zeros(n)

    #Unitary base of Krylov subspace (maximum number of cols: n)
    Q = np.zeros((len(r), n))
    Q[:,0] = r/np.linalg.norm(r) #Normalize the input vector

    #Hessenberg matrix
    H = np.zeros((n+1,n))

    #First canonical vector:
    e1 = np.zeros(n+1)
    e1[0] = 1

    #Beta vector to be multiplied by Givens matrices.
    beta = e1 * r_norm

    with open('error_gmres.txt', 'a') as file:
        file.write('GMRes: \n')
    #In each iteration a new column of Q and H is computed.
    #The H column is then modified using Givens matrices so that H becomes a triangular matrix R
    for k in range(1,n):
        Q[:,k], H[:k+1,k-1] = arnoldi_step(apply_A, Q, k) #Perform Arnoldi iteration to add column to Q (m entries) and to H (k entries)  
        
        H[:k+1,k-1], cs[k-1],sn[k-1] = apply_givens_rotation(H[:k+1,k-1],cs,sn,k) #eliminate the last element in H ith row and update the rotation matrix

        #update residual vector
        beta[k] = -sn[k-1] * beta[k-1]
        beta[k-1] = cs[k-1] * beta[k-1]
        
        #||r_k|| can be obtained with the last element of beta because of the Givens algorithm
        error = abs(beta[k])/b_norm

        #save the error
        e.append(error)
        with open('error_gmres.txt', 'a') as file:
            file.write(f'error({k}) = {error} \n')

        if error<tol:
            break
    #calculate result by solving a triangular system of equations H*y=beta
    y = back_substitution(H[:k,:k], beta[:k])
    # x = x0 + Q[:,:k]@y
    x = Q[:,:k]@y
    return x[:-1], x[-1], e


def arnoldi_step(apply_A, Q, k):
    """Performs k_th Arnoldi iteration of Krylov subspace spanned by <r, Ar, A^2 r,.., A^(k-1) r> 
    """
    v = apply_A(Q[:-1,k-1],Q[-1,k-1]) #generate candidate vector
    h = np.zeros(k+1)
    for j in range(k):#substract projections of previous vectors
        h[j] = np.dot(Q[:,j], v)
        v -= h[j] * Q[:,j]
    h[k] = np.linalg.norm(v, 2)
    return (v/h[k], h) #Returns k_th column of Q (python indexing) and k-1 column of H
    # TODO: Check if i should verify if norm(v)!=0, i.e. the matrix A is not full rank
    # if h[k] > tol:
    #     return (v/h[k], h) #Returns k_th column of Q (python indexing) and k-1 column of H
    # else:
    #     print('Matrix A not full rank')


def apply_givens_rotation(h, cs, sn, k):
    #Premultiply the last H column by the previous k-1 Givens matrices
    for i in range(k-1):
        temp = cs[i]*h[i] + sn[i]*h[i+1]
        h[i+1] = -sn[i]*h[i] + cs[i]*h[i+1]
        h[i] = temp        
    hip = np.sqrt(np.abs(h[k-1])**2+np.abs(h[k])**2)
    cs_k, sn_k = h[k-1]/hip, h[k]/hip
    #Update the last H entries and eliminate H[k,k-1] to obtain a triangular matrix R
    h[k-1] = cs_k*h[k-1] + sn_k*h[k]
    h[k] = 0
    return h, cs_k, sn_k


def back_substitution(R, b):
    """
    Solves the equation Rx = b, where R is a square right triangular matrix
    """
    n = len(b)
    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        aux = 0
        for j in range(i+1, n):
            aux += x[j] * R[i,j]
        x[i] = (b[i]-aux)/R[i,i]
    return x
    


def initialize_vectors(rN, n):
    r_norm = np.linalg.norm(rN, 2)
    #Initialize sine and cosine Givens 1d vectors:
    sn = np.zeros(n)
    cs = np.zeros(n)

    #Unitary base of Krylov subspace (maximum number of cols: n)
    Q = np.zeros((len(rN), n))
    Q[:,0] = rN/np.linalg.norm(rN, 2) #Normalize the input vector

    #Hessenberg matrix
    H = np.zeros((n+1,n))

    #First canonical vector:
    e1 = np.zeros(n+1)
    e1[0] = 1

    #Beta vector to be multiplied by Givens matrices
    beta = e1 * r_norm
    return Q, H, beta, cs, sn


if __name__ == '__main__':

    print('')
    m = 100
    np.random.seed(0)

    A = np.zeros((m,m))
    A = np.random.randn(m,m)
    A = 2*np.eye(m) + 0.5*A/np.sqrt(m)

    # A = np.random.rand(m,m) + 0.5j*np.random.rand(m,m)

    b = np.random.rand(m)
 
    x0 = np.zeros_like(b)
    max_iterations = m+1
    threshold = 1e-10

    # record start time
    time_start = time.perf_counter()
    # call benchmark code
    # max_iterations = 4
    x, e = GMRES(A, b, x0, max_iterations, threshold, restart = 0)
    x_norm = np.linalg.norm(x, 2)

    # record end time
    time_end = time.perf_counter()
    
    
    print("Result:", x)
    print("Errors:", e)

    print('Performance GMRES:', time_end-time_start)

    plt.figure()
    plt.plot(e, lw = 0, marker = 'o', markersize = 3)
    plt.yscale('log')
    plt.ylabel('Error')
    plt.xlabel('Iteracion')
    plt.show()

    ###### GMRES(N) #####
    # # record start time
    # time_start = time.perf_counter()
    # # call benchmark code
    # x_rest, e_rest = GMRES(A, b, x0, max_iterations, threshold, restart = m//10)
    # x_rest_norm = np.linalg.norm(x_rest, 2)
    # # record end time
    # time_end = time.perf_counter()
    
    # print("Result:", x_rest)
    # print("Errors:", e_rest)

    # print('Performance GMRES(N):', time_end-time_start)

    # plt.figure()
    # plt.plot(e_rest, lw = 0, marker = 'o', markersize = 3)
    # plt.ylabel('Error')
    # plt.xlabel('Iteracion')
    # plt.show()
    ###### End GMRES(N) #####

    # record start time
    time_start = time.perf_counter()
    # call benchmark code
    
    x_scipy = scipy.linalg.solve(A, b)
    x_scipy_norm = np.linalg.norm(x_scipy, 2)

    # record end time
    time_end = time.perf_counter()

    print('Scipy solver: x = ', x_scipy)
    print('Performance Scipy:', time_end-time_start)
    	
    # print('Norm GMRES(N) - Norm GMRES = ', abs(x_norm-x_rest_norm))
    print('Norm scipy - norm GMRES = ', abs(x_norm-x_scipy_norm))
