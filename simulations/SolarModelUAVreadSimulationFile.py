import matplotlib.pyplot as plt
import numpy as np
import GPModels as models

gamma = []
pwl = []
gamma_data = [[], [], [], [], [], [], [], []]
pwl_data = [[], [], [], [], [], []]

the_file = open("SolarModelUAVsim.txt", "r")

line = the_file.readline()
while line != "":
    if 'gamma' in line:
        gamma.append(float(line.split(" = ")[-1]))
        for j in xrange(8):
            line = the_file.readline()
            line = line.replace('<Quantity(', '')
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace('force_pound', '')
            line = line.replace('>', '')
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            data = line.split(',')
            data = [float(i) for i in data if i != "''"]
            gamma_data[j].append(data)
    elif 'number of linear sections' in line:
        pwl.append(float(line.split(" = ")[-1]))
        for j in xrange(6):
            line = the_file.readline()
            line = line.replace('<Quantity(', '')
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace('force_pound', '')
            line = line.replace('>', '')
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            data = line.split(',')
            data = [float(i) for i in data if i != "''"]
            pwl_data[j].append(data)
    line = the_file.readline()

nominal_solution = gamma_data[2][0][2]
model = models.mike_solar_model(20)
solve_time = model.solve(verbosity=0)['soltime']

simple_box_obj = [gamma_data[2][i][2] / nominal_solution for i in xrange(len(gamma))]
iter_ell_obj = [gamma_data[4][i][2] / nominal_solution for i in xrange(len(gamma))]
boyd_ell_obj = [gamma_data[7][i][2] / nominal_solution for i in xrange(len(gamma))]
"""
plt.figure()
plt.plot(gamma[11:13], iter_ell_obj[11:13], 'r--', label='Uncertain Exponents')
plt.plot(gamma[11:13], simple_box_obj[11:13], 'b--', label='Margins')
plt.plot(gamma[11:13], boyd_ell_obj[11:13], 'g--', label='Boyd')
# plt.plot(the_gamma, boyd_box_obj, 'ro', label='State of Art')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Objective Function")
plt.title("The Average Performance: Box Uncertainty Set")
plt.legend(loc=0)
plt.show()

plt.figure()
x = np.arange(3)
plt.bar(x + [0.3, 0.3, 0.3],
        [pwl_data[3][len(pwl) - 1][2] / nominal_solution, gamma_data[7][len(gamma) - 1][2] / nominal_solution,
         gamma_data[2][len(gamma) - 1][2] / nominal_solution], [0.4, 0.4, 0.4], color=['r', 'g', 'b'])
plt.xticks(x + .5, ['Uncertain Exponents', 'Boyd', 'Margins'])
plt.ylabel("Weight/nominal Weight")
plt.title("The Expected Weight")
plt.show()

plt.figure()
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

x = np.arange(4)
ax1.bar(x + [0.2, 0.2, 0.2, 0.2],
        [pwl_data[3][len(pwl) - 1][2] / nominal_solution, pwl_data[4][len(pwl) - 1][2] / nominal_solution,
         gamma_data[7][len(gamma) - 1][2] / nominal_solution, gamma_data[2][len(gamma) - 1][2] / nominal_solution],
        [0.2, 0.2, 0.2, 0.2], color='b', label='Optimal Relative Weight')
ax2.bar(x + [0.45, 0.45, 0.45, 0.45],
        [pwl_data[3][len(pwl) - 1][5] / 2698.0, pwl_data[4][len(pwl) - 1][5] / 2698.0,
         gamma_data[7][len(gamma) - 1][5] / 2698.0, gamma_data[2][len(gamma) - 1][5] / 2698.0],
        [0.2, 0.2, 0.2, 0.2], color='g', label='Relative Number of constraints')
ax1.set_ylabel("Weight/Nominal Weight")
ax1.set_ylim((1.5, 2.2))
ax2.set_ylabel("Constraints/Nominal Constraints")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd', 'Margins'])
plt.title('Comparing Different Methods for Solving the Solar Model Under Uncertainty')
plt.show()

plt.figure()
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

x = np.arange(4)
ax1.bar(x + [0.2, 0.2, 0.2, 0.2],
        [pwl_data[3][len(pwl) - 1][1] / nominal_solution, pwl_data[4][len(pwl) - 1][1] / nominal_solution,
         gamma_data[7][len(gamma) - 1][1] / nominal_solution, gamma_data[2][len(gamma) - 1][1] / nominal_solution],
        [0.2, 0.2, 0.2, 0.2], color='b', label='Optimal Relative Weight')
ax2.bar(x + [0.45, 0.45, 0.45, 0.45],
        [pwl_data[3][len(pwl) - 1][5] / 2698.0, pwl_data[4][len(pwl) - 1][5] / 2698.0,
         gamma_data[7][len(gamma) - 1][5] / 2698.0, gamma_data[2][len(gamma) - 1][5] / 2698.0],
        [0.2, 0.2, 0.2, 0.2], color='g', label='Relative Number of constraints')
ax1.set_ylabel("Weight/Nominal Weight")
ax1.set_ylim((1.5, 2.2))
ax2.set_ylabel("Constraints/Nominal Constraints")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd', 'Margins'])
plt.title('Comparing Different Methods for Solving the Solar Model Under Uncertainty')
plt.show()

iter_ell_prob_of_failure = [gamma_data[4][i][0] for i in xrange(len(gamma))]

plt.figure()
plt.plot(gamma, iter_ell_prob_of_failure, 'r--')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Probability of Failure")
plt.title("The Probability of Failure of the Solar Model as a function of the size of the Uncertainty Set")
plt.legend(loc=0)
plt.figure()
"""
x = np.arange(4)
plt.bar(x + [0.2, 0.2, 0.2, 0.2],
        [pwl_data[3][len(pwl) - 1][2] / nominal_solution, pwl_data[4][len(pwl) - 1][2] / nominal_solution,
         gamma_data[7][len(gamma) - 1][2] / nominal_solution, gamma_data[2][len(gamma) - 1][2] / nominal_solution],
        [0.5, 0.5, 0.5, 0.5], color='b', label='Optimal Relative Weight')
plt.ylabel("Scaled Weight")
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd', 'Traditional'])
plt.title('Worst-case Optimal Weight')
plt.ylim(1.6, 2.1)
plt.show()

plt.figure()
x = np.arange(4)
plt.bar(x + [0.2, 0.2, 0.2, 0.2],
        [pwl_data[3][len(pwl) - 1][1] / nominal_solution, pwl_data[4][len(pwl) - 1][1] / nominal_solution,
         gamma_data[7][len(gamma) - 1][1] / nominal_solution, gamma_data[2][len(gamma) - 1][1] / nominal_solution],
        [0.5, 0.5, 0.5, 0.5], color='b', label='Optimal Relative Weight')
plt.ylabel("Scaled Weight")
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd', 'Traditional'])
plt.title('Average Optimal Weight')
plt.ylim(1.6, 2.1)
plt.show()

plt.figure()
x = np.arange(4)
plt.bar(x + [0.2, 0.2, 0.2, 0.2],
        [pwl_data[3][len(pwl) - 1][5] / 2698.0, pwl_data[4][len(pwl) - 1][5] / 2698.0,
         gamma_data[7][len(gamma) - 1][5] / 2698.0, gamma_data[2][len(gamma) - 1][5] / 2698.0],
        [0.5, 0.5, 0.5, 0.5], color='b', label='Complexity')
plt.ylabel("Scaled Complexity")
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd', 'Traditional'])
plt.title('Robust Formulation Complexity')
plt.show()

plt.figure()
x = np.arange(3)
plt.bar(x + [0.2, 0.2, 0.2],
        [(gamma_data[4][len(gamma) - 2][3]) / solve_time,
         (gamma_data[5][len(gamma) - 2][3]) / solve_time,
         (gamma_data[7][len(gamma) - 2][3]) / solve_time],
        [0.5, 0.5, 0.5], color='b', label='Complexity')
plt.ylabel("Scaled Setup Time")
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd'])
plt.title('Robust Formulation Setup Time')
plt.show()

plt.figure()
x = np.arange(4)
plt.bar(x + [0.2, 0.2, 0.2, 0.2],
        [(pwl_data[3][len(pwl) - 1][4]) / solve_time,
         (pwl_data[4][len(pwl) - 1][4]) / solve_time,
         (gamma_data[7][len(gamma) - 2][4]) / solve_time,
         solve_time / solve_time],
        [0.5, 0.5, 0.5, 0.5], color='b', label='Complexity')
plt.ylabel("Scaled Solve Time")
plt.xticks(x + .45, ['Uncertain Exps', 'Uncertain Coeffs', 'Boyd', 'Traditional'])
plt.title('Robust Formulation Solve Time')
plt.show()

iter_ell_prob_of_failure = [gamma_data[4][i][0] for i in xrange(len(gamma))]

plt.figure()
plt.plot(gamma, iter_ell_prob_of_failure, 'b')
plt.xlabel("Uncertainty Set Scaling Factor Gamma")
plt.ylabel("Probability of Failure")
plt.title("Robust Design Probability of Failure")
plt.legend(loc=0)
plt.show()
