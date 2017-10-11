import gassolar.solar.solar as solar_mike
model = solar_mike.Mission(latitude=25)
model.cost = model["W_{total}"]

free_vars = [var for var in model.varkeys if var.key not in model.substitutions.keys()]
subs_vars = model.substitutions.keys()


file = open("free_variables.txt", "w")

for var in free_vars:
    file.write(var.key.name + ": " + var.key.label + ": %s:  %s \n" % (var.key.value, var.key.models))

file.close()

file = open("substitutions_variables.txt", "w")

for var in subs_vars:
    file.write(var.key.name + ": " + var.key.label + ": %s:  %s \n" % (var.key.value, var.key.models))

file.close()
