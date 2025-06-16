def resolver_diagonal(
    menor,
    diagonal,
    mayor,
    b,
):
    """
    Resuelve la matriz Ax=b cuando A es una diagonal.
    """
    from scipy.sparse import linalg, diags

    N = len(diagonal)
    A = diags(
        diagonals=[
            menor,
            diagonal,
            mayor,
        ],
        offsets=[-1, 0, 1],
        shape=(N, N),
        format="csr",
    )

    return linalg.spsolve(A, b)
