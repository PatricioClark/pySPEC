from .newton import NewtonSolver

class UPO(NewtonSolver):
    def __init__(self, pm, solver):
        super().__init__(pm, solver)
        self.fields = solver.fields
        self.grid = solver.grid
        self.evolve = solver.evolve

    def update_A(self):
        pass

    def apply_A(self):
        pass
