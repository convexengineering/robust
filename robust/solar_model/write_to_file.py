import gassolar.solar.solar as solar_mike

model = solar_mike.Mission(latitude=11)
model.cost = model["W_{total}"]

solution = model.solve()

free_vars = solution['freevariables']
subs_vars = solution['constants']

model_freevars = {}
model_subsvars = {}

for var in free_vars:
    string = ''
    for item in var.key.models:
        string = string + item + "-"
    if string not in model_freevars.keys():
        model_freevars[string] = {var: free_vars[var]}
    else:
        model_freevars[string].update({var: free_vars[var]})

for var in subs_vars:
    string = ''
    for item in var.key.models:
        string = string + item + "-"
    if string not in model_subsvars.keys():
        model_subsvars[string] = {var: subs_vars[var]}
    else:
        model_subsvars[string].update({var: subs_vars[var]})

data_file = open("data/free_variables.txt", "w")
for model in model_freevars:
    data_file.write(model + ": \n")
    for var in model_freevars[model]:
        data_file.write(' '*len(model) + var.key.name + ": " + var.key.label + "\n")
    data_file.write("\n")
data_file.close()

data_file = open("data/substitutions_variables.txt", "w")
for model in model_subsvars:
    data_file.write(model + ": \n")
    for var in model_subsvars[model]:
        data_file.write(' '*len(model) + var.key.name + ": " + var.key.label + ": %s ---- %s \n" % (model_subsvars[model][var], var.key.value))
    data_file.write("\n")
data_file.close()
