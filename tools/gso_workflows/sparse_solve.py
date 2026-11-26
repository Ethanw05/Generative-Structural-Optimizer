"""Scoped sparse factorizations for repeated right-hand sides on an unchanged matrix."""
from contextlib import contextmanager
import numpy as np
from scipy.sparse.linalg import splu, spsolve


class PardisoLinearSolver:
    def __init__(self,factor_storage=None):
        self.factor_storage=factor_storage
        from pypardiso import PyPardisoSolver
        self.single = PyPardisoSolver(mtype=11)

    def __call__(self, matrix, rhs):
        return self.single.solve(matrix, rhs)

    def fixed_stiffness(self,matrix):
        if self.factor_storage is None:
            return self.fixed_matrix(matrix)
        from .disk_factors import disk_backed_stiffness
        return disk_backed_stiffness(matrix,*self.factor_storage)

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
def fixed_matrix_solver(matrix, solve, *, positive_definite=False):
    if positive_definite and hasattr(solve,"fixed_stiffness"):
        with solve.fixed_stiffness(matrix) as apply:
            yield apply
    elif hasattr(solve, "fixed_matrix"):
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
