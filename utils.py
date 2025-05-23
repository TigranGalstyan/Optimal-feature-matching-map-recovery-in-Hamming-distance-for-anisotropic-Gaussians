import numpy as np
import cvxpy as cp

from scipy.optimize import linear_sum_assignment


def solve_sdp_optimization(X, X_diese):
    """
    Solve the SDP optimization problem:

    minimize log(det(M))
    subject to:
        sum_{i,j} (X_i - X_diese_j)(X_i - X_diesej)^T P{ij} <= M
        P_{ij} in [0, 1]
        sumj P{ij} = 1 for all i (each row of P sums to 1)
        sumi P{ij} <= 1 for all j (each column of P sums to at most 1)
        M is positive semidefinite

    Args:
        X: List of n vectors X_i, each of dimension d
        X_diese: List of m vectors X_diese_j, each of dimension d

    Returns:
        P: Optimal n x m matrix
        M: Optimal d x d positive semidefinite matrix
        optimal_value: The optimal value of the objective function
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.array(X)
    X_diese = np.array(X_diese)

    # Dimensions
    n = X.shape[0]      # Number of X_i vectors
    m = X_diese.shape[0]  # Number of X_diese_j vectors
    d = X.shape[1]      # Dimension of the vectors

    print(n, m, d)

    # Define variables
    P = cp.Variable((n, m), nonneg=True)  # P must be non-negative
    M_inv = cp.Variable((d, d), symmetric=True)  # M⁻¹ must be symmetric

    # Objective function: maximize log(det(M⁻¹)), which is equivalent to minimizing log(det(M))
    objective = cp.log_det(M_inv)

    # Constraint 1: Row sums of P = 1
    row_sum_constraints = [cp.sum(P, axis=1) == 1]  # Each row sums to 1

    # Constraint 2: Column sums of P <= 1
    col_sum_constraints = [cp.sum(P, axis=0) <= 1]  # Each column sums to at most 1

    # Constraint 3: Entries of P in [0, 1]
    bound_constraints = [P <= 1]  # P is already constrained to be non-negative

    # Constraint 4: M_inv is positive definite
    psd_constraint =  [] #[M_inv >> 0]  # M⁻¹ is PSD

    # Constraint 5: The matrix inequality constraint using Schur complement
    lhs = cp.Variable((d, d), symmetric=True)
    for i in range(n):
        for j in range(m):
            diff = X[i] - X_diese[j]
            outer_product = np.outer(diff, diff)
            lhs += cp.multiply(P[i, j], outer_product)

    # Create a block matrix for the Schur complement formulation
    # schur_matrix = cp.bmat([
    #     [M_inv, cp.Variable((d, d))],
    #     [cp.Variable((d, d)), lhs]
    # ])

    # # Create the Schur complement matrix
    # [M⁻¹, I]
    # [I, sum_{i,j} (X_i - X_diese_j)(X_i - X_diesej)^T P{ij}]
    identity = np.eye(d)
    schur_matrix = cp.bmat([
        [M_inv, identity],
        [identity, lhs]
    ])

    matrix_constraint = [schur_matrix >> 0]

    # Combine all constraints
    constraints = row_sum_constraints + col_sum_constraints + bound_constraints + psd_constraint + matrix_constraint

    # Define and solve the problem
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=cp.SCS)

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        print(f"Problem status: {problem.status}")
        return None, None, None

    # Recover M from M_inv
    M_value = np.linalg.inv(M_inv.value)

    return P.value, M_value, -problem.value, schur_matrix.value  # Negate the objective value to get log(det(M))


def recover_permutation(P):
    """
    Convert the P matrix to a permutation using the Hungarian algorithm.

    Args:
        P: n x m matrix with values in [0, 1]

    Returns:
        perm: List where perm[i] is the index j that i maps to
        cost_matrix: The cost matrix used for the assignment problem
    """
    # Use negative P as cost matrix (Hungarian algorithm minimizes cost)
    cost_matrix = -P

    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the permutation mapping
    perm = col_ind.tolist()

    return perm, cost_matrix
