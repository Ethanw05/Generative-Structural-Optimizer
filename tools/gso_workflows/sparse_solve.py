"""Scoped sparse factorizations for repeated right-hand sides on an unchanged matrix."""
from contextlib import contextmanager
import numpy as np
from scipy.sparse.linalg import splu, spsolve


class PardisoLinearSolver:
    def __init__(self):
        from pypardiso import PyPardisoSolver
        self.single = PyPardisoSolver(mtype=11)

    def __call__(self, matrix, rhs):
        return self.single.solve(matrix, rhs)

    @contextmanager
    def fixed_matrix(self, matrix):
        # A nested solve (e.g. stress-field projection) may use another matrix.
        # Keep its numerical factors independent of the enclosing Ritz solve.
        from pypardiso import PyPardisoSolver
        solver = PyPardisoSolver(mtype=11)
        try:
            solver.factorize(matrix)
            yield lambda rhs: solver.solve(matrix, np.asfortranarray(rhs))
        finally:
            solver.free_memory(everything=True)


@contextmanager
def fixed_matrix_solver(matrix, solve):
    if hasattr(solve, "fixed_matrix"):
        with solve.fixed_matrix(matrix) as apply:
            yield apply
    elif solve is spsolve:
        factors = splu(matrix.tocsc())
        yield factors.solve
    else:
        # Existing injected solvers may accept only a single force vector.
        def apply(rhs):
            if rhs.ndim == 1:
                return solve(matrix, rhs)
            return np.column_stack([solve(matrix, np.ascontiguousarray(f)) for f in rhs.T])
        yield apply
