from gpkit import Variable, Model, ConstraintSet
import numpy as np

class m(Model):
    def setup(self):
        a1 = Variable('a1', np.linspace(1, 2, 2), '-')
        a2 = Variable('a2', np.linspace(2, 5, 2), '-')
        a3 = Variable('a3', np.linspace(2, 5, 2), '-')
        def f(c):
            return np.exp(c[a1]*c[a2]*c[a3])
        b = Variable('b', f)
        d = Variable('d', 1)
        x = Variable('x')
        self.cost = x
        return [b*x >= d]

m = m()
print m.solve().table()

