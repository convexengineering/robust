from GPModels import mike_solar_model
from time import time

model = mike_solar_model()

def test():
         st = time()
         model.solve("mosek")
         print time() - st
         
test()
