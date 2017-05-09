from gpkit import Variable, Model, SignomialsEnabled, VarKey, units
import numpy as np
import RobustGP as RGP
import EquivalentModels as EM
import copy

def simpleWing():
    # Env. constants
    g = Variable("g", 9.81, "m/s^2", "gravitational acceleration")
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air", pr=4.225352)
    rho = Variable("\\rho", 1.23, "kg/m^3", "density of air")
    rho_f = Variable("\\rho_f", 817, "kg/m^3", "density of fuel")
    u = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air", pr=4.225352)

    # Non-dimensional constants
    C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down", pr=25)
    e = Variable("e", 0.92, "-", "Oswald efficiency factor", pr=7.6086956)
    k = Variable("k", 1.17, "-", "form factor", pr=11.111111)
    N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor", pr=33.333333)
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio", pr=3.6144578)
    tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio", pr=33.333333)
    W_W_coeff1 = Variable("W_{W_{coeff1}}", 12e-5, "1/m",
                          "Wing Weight Coefficent 1", pr=66.666666)
    W_W_coeff2 = Variable("W_{W_{coeff2}}", 60, "Pa",
                          "Wing Weight Coefficent 2", pr=66.666666)

    # Dimensional constants
    CDA0 = Variable("(CDA0)", 0.035, "m^2", "fuselage drag area", pr=42.857142)
    Range = Variable("Range",3000, "km", "aircraft range")
    toz = Variable("toz", 1, "-", pr=15)
    TSFC = Variable("TSFC", 0.6, "1/hr", "thrust specific fuel consumption")
    V_min = Variable("V_{min}", 25, "m/s", "takeoff speed", pr=20)
    W_0 = Variable("W_0", 6250, "N", "aircraft weight excluding wing", pr=60)

    # Free Variables
    D = Variable("D", "N", "total drag force")
    A = Variable("A", "-", "aspect ratio")
    S = Variable("S", "m^2", "total wing area")
    V = Variable("V", "m/s", "cruising speed")
    W = Variable("W", "N", "total aircraft weight")
    Re = Variable("Re", "-", "Reynold's number")
    C_D = Variable("C_D", "-", "Drag coefficient of wing")
    C_L = Variable("C_L", "-", "Lift coefficent of wing")
    C_f = Variable("C_f", "-", "skin friction coefficient")
    W_w = Variable("W_w", "N", "wing weight")
    W_f = Variable("W_f", "N", "fuel weight")
    V_f = Variable("V_f", "m^3", "fuel volume")
    T_flight = Variable("T_{flight}", "hr", "flight time")

    constraints = []

    # Drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (np.pi * A * e)
    constraints += [C_D >= C_D_fuse * toz + C_D_wpar / toz + C_D_ind * toz]

    # Wing weight model
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    constraints += [W_w >= W_w_surf + W_w_strc]

    # and the rest of the models

    constraints += [D >= 0.5 * rho * S * C_D * V ** 2,
                    Re <= (rho / mu) * V * (S / A) ** 0.5,
                    C_f >= 0.074 / Re ** 0.2,
                    T_flight == Range / V,
                    W_0 + W_w + 0.5 * W_f <= 0.5 * rho * S * C_L * V ** 2,
                    W <= 0.5 * rho * S * C_Lmax * V_min ** 2,
                    W >= W_0 + W_w + W_f,
                    V_f == W_f / g / rho_f,
                    W_f >= TSFC * T_flight * D]

    return Model(D, constraints)
    # return Model(W_f, constraints)
    # return Model(W, constraints)
    # return Model(W_f*T_flight,constraints)
    # return Model(W_f + 10*T_flight*units('N/min'),constraints)

def simpleWingTwoDimensionalUncertainty():
    k = Variable("k", 1.17, "-", "form factor", pr=11.111111)
    e = Variable("e", 0.92, "-", "Oswald efficiency factor")
    mu = Variable("\\mu", 1.775e-5, "kg/m/s", "viscosity of air")
    # pi = Variable("\\pi", np.pi, "-", "half of the circle constant", pr= 0)
    rho = Variable("\\rho", 1.23, "kg/m^3", "density of air")
    tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
    N_ult = Variable("N_{ult}", 3.3, "-", "ultimate load factor")
    V_min = Variable("V_{min}", 25, "m/s", "takeoff speed")
    C_Lmax = Variable("C_{L,max}", 1.6, "-", "max CL with flaps down")
    S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.075, "-", "wetted area ratio")
    W_W_coeff1 = Variable("W_{W_{coeff1}}", 12e-5, "1/m",
                          "Wing Weight Coefficent 1")
    W_W_coeff2 = Variable("W_{W_{coeff2}}", 60, "Pa",
                          "Wing Weight Coefficent 2")
    CDA0 = Variable("(CDA0)", 0.035, "m^2", "fuselage drag area")
    W_0 = Variable("W_0", 6250, "N", "aircraft weight excluding wing")
    toz = Variable("toz", 1, "-", pr=15)

    # Free Variables
    D = Variable("D", "N", "total drag force")
    A = Variable("A", "-", "aspect ratio")
    S = Variable("S", "m^2", "total wing area")
    V = Variable("V", "m/s", "cruising speed")
    W = Variable("W", "N", "total aircraft weight")
    Re = Variable("Re", "-", "Reynold's number")
    C_D = Variable("C_D", "-", "Drag coefficient of wing")
    C_L = Variable("C_L", "-", "Lift coefficent of wing")
    C_f = Variable("C_f", "-", "skin friction coefficient")
    W_w = Variable("W_w", "N", "wing weight")

    constraints = []

    # Drag model
    C_D_fuse = CDA0 / S
    C_D_wpar = k * C_f * S_wetratio
    C_D_ind = C_L ** 2 / (np.pi * A * e)
    constraints += [C_D >= C_D_fuse / toz + C_D_wpar / toz + C_D_ind / toz]

    # Wing weight model
    # W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_strc = W_W_coeff1 * (N_ult * A ** 1.5 * (W_0 * W * S) ** 0.5) / tau
    W_w_surf = W_W_coeff2 * S
    constraints += [W_w >= W_w_surf + W_w_strc]

    # and the rest of the models
    constraints += [D >= 0.5 * rho * S * C_D * V ** 2,
                    Re <= (rho / mu) * V * (S / A) ** 0.5,
                    C_f >= 0.074 / Re ** 0.2,
                    W <= 0.5 * rho * S * C_L * V ** 2,
                    W <= 0.5 * rho * S * C_Lmax * V_min ** 2,
                    W >= W_0 + W_w]

    return Model(D, constraints)


# class uncertainModel(Model):
#    def setup(self, model, pfail, distr):
#        for vk in model.varkeys:
#            
#        return constraints

def testModel():
    x = Variable('x')
    y = Variable('y')

    a = Variable('a', 1.1, pr = 10)
    b = Variable('b', 1, pr = 10)
    
    constraints = [a*b*x + a*b*y <= 1,
                    b*x/y + b*x*y + b*x**2 <= 1]
    return Model((x*y)**-1, constraints)

def exampleSP():
    x = Variable('x')
    y = Variable('y')
    a = Variable('a',1,pr = 10)
    b = Variable('b',1, pr = 10)
    constraints = []
    with SignomialsEnabled():
        constraints = constraints + [x >= 1 - a*y, b*y <= 0.1]
    return Model(x, constraints)


def MikeSolarModel():
    import gassolar.solar.solar as solarMike
    model = solarMike.Mission(latitude=25)
    model.cost = model["W_{total}"]
    uncertainVarDic = {"W_{pay}":5,"B_{PM}":0,"\\eta":0.0000001,"\\eta_{charge}":1,"\\eta_{discharge}":1,
                                "h_{batt}":2,"W_{total}":2,"W_{cent}":2,"W_{wing}":2,"\\tau":2,"C_M":0,"e":2,
                                "(E/\\mathcal{V})":1,"C_{L_{max}}":1,"m_w":1,}
    keys = uncertainVarDic.keys()
    for i in xrange(len(uncertainVarDic)):
        #limits = uncertainVarDic.get(keys[i])
        #value = sum(limits)/2.0
        for j in xrange(len(model.variables_byname(keys[i]))):
            print(keys[i],uncertainVarDic.get(keys[i]),len(model.variables_byname(keys[i])))
            copy_key = VarKey(**model.variables_byname(keys[i])[j].key.descr)
            copy_key.key.descr["pr"] = uncertainVarDic.get(keys[i])#((value - limits[0])/(value + 0.0))*100
            model.subinplace({model.variables_byname(keys[i])[j].key:copy_key})
            #model.substitutions[copy_key] = sum(limits)/2
    return model


def solveModel(model, *args):
    initialGuess = {}
    if args:
        initialGuess = args[0]
        # del initialGuess["S"]
    try:
        sol = model.solve(verbosity=0)
    except:
        sol  = model.localsolve(verbosity=0,x0 = initialGuess)
    #print (sol.summary())
    return sol
    
def evaluateRandomModel(oldModel,solutionBox, solutionEll):
    modelBox = EM.sameModel(oldModel)
    modelEll = EM.sameModel(oldModel)
    freeVars = [var for var in modelBox.varkeys.keys()
    if var not in modelBox.substitutions.keys()]
    uncertainVars = [var for var in modelBox.substitutions.keys()
                    if "pr" in var.key.descr]             
    for key in freeVars:
        if key.descr['name'] == "A" or key.descr['name'] == "S":
            try:
                modelBox.substitutions[key] = solutionBox.get(key).m
                modelEll.substitutions[key] = solutionEll.get(key).m
            except:
                modelBox.substitutions[key] = solutionBox.get(key)
                modelEll.substitutions[key] = solutionEll.get(key)
    for key in uncertainVars:
        val = modelBox[key].key.descr["value"]
        pr = modelBox[key].key.descr["pr"]
        if pr != 0:
            sigma = pr*val/300.0
            #newVal = np.random.normal(val,sigma)
            newVal = np.random.uniform(val-pr*val/100.0,val+pr*val/100.0)
            modelBox.substitutions[key] = newVal
            modelEll.substitutions[key] = newVal
    return modelBox, modelEll
    
def failOrSuccess(model):
    try:
       #model.as_posyslt1(model.substitutions)
       sol = solveModel(model);
       vars = sol.get('variables')
       return True, vars.get("D")
    except:
        return False,0

def probabilityOfFailure(model,numberOfIterations):
    probBox = []
    costBox = []
    probEll = []
    costEll = []
    for Gamma in xrange(9):
        failureBox = 0
        successBox = 0
        failureEll = 0
        successEll = 0
        robModelBox = RGP.robustModelBoxUncertainty(model,(Gamma)/6.0)
        robModelEll = RGP.robustModelEllipticalUncertainty(model, (Gamma)/6.0)
        #print('Done Creating Robust Models')
        solBox = solveModel(robModelBox[0])
        solEll = solveModel(robModelEll[0])
        #print('Done Solving')
        solutionBox = solBox.get('variables')
        solutionEll = solEll.get('variables')
        del solutionBox["D"]
        del solutionEll["D"]
        sumCostBox = 0
        sumCostEll = 0
        for i in xrange(numberOfIterations):
            print('Gamma = %s'%Gamma, 'iteration: %s' %i)
            newModelBox, newModelEll = evaluateRandomModel(model,solutionBox, solutionEll)
            FSBox,cost = failOrSuccess(newModelBox)
            sumCostBox = sumCostBox + cost
            if FSBox:
                successBox = successBox + 1
            else:
                failureBox = failureBox + 1
            FSEll,cost = failOrSuccess(newModelEll)
            sumCostEll = sumCostEll + cost
            if FSEll:
                successEll = successEll + 1
            else:
                failureEll = failureEll + 1
        costBox.append(sumCostBox/(successBox+0.0))
        costEll.append(sumCostEll/(successEll+0.0))
        probBox.append(failureBox/(failureBox+successBox+0.0))
        probEll.append(failureEll/(failureEll+successEll+0.0))
    return probBox, probEll,costBox,costEll


if __name__ == '__main__':
    m = simpleWing()
    sol = m.solve()
