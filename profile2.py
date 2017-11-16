from GPModels import mike_solar_model, Model
from time import time


@profile
def test():
    model = mike_solar_model()
    cost = model.cost
    cons = [i <= 1 for i in model.as_posyslt1()]

    def first():
        Model(cost, cons)

    first()

    # def second():
    #     Model(cost, cons)
    #
    # second()


test()
