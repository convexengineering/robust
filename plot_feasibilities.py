import numpy as np
from gpkit import Model, Variable, ConstraintSet, GPCOLORS, GPBLU, Vectorize, Monomial
from gpkit.small_scripts import mag


def plot_feasibilities(x, y, m, rm=None, rmtype=None):
    posynomials = m.as_posyslt1()
    interesting_vars = [x, y]
    old = []
    while set(old) != set(interesting_vars):
        old = interesting_vars
        for p in posynomials:
            if set(interesting_vars) & set(p.varkeys.keys()):
                interesting_vars = list(set(interesting_vars) | set(p.varkeys.keys()))

    class FeasCircle(Model):
        "SKIP VERIFICATION"
        def setup(self, m, sol):
            r = 4
            additional_constraints = []
            slacks = []
            thetas = []
            for count in xrange((len(interesting_vars)-1)):
                th = Variable("\\theta_%s" % count, np.linspace(0, 2*np.pi, 120), "-")
                thetas += [th]
            for i_set in xrange(len(interesting_vars)):
                x_i = Monomial(1)*Variable("x_%s" % i_set, 4, interesting_vars[i_set].unitstr())
                for j in xrange(i_set):
                    x_i *= Variable("x_%s^%s" % (i_set, j), lambda c, i_set=i_set: sol(interesting_vars[i_set])*np.exp(np.cos(c[thetas[j]])), '-')
                if i_set != len(interesting_vars) - 1:
                    x_i *= Variable("x_%s^%s" % (i_set, i_set), lambda c, i_set=i_set: sol(interesting_vars[i_set])*np.exp(np.sin(c[thetas[i_set]])), '-')
                print x_i
                s_i = Variable("s_%s" % i_set)
                slacks += [s_i]
                additional_constraints += [s_i >= 1, m[interesting_vars[i_set]]/s_i <= x_i, x_i <= m[interesting_vars[i_set]]*s_i]

            self.cost = sum([sl**2 for sl in slacks])
            feas_slack = ConstraintSet(additional_constraints)

            return [m, feas_slack], {k: v for k, v in sol["freevariables"].items()
                                     if k in m.varkeys}

    # plot original feasibility set
    # plot boundary of uncertainty set
    sol = None
    if rm:
        fc = FeasCircle(m, rm.solution)
        for interesting_var in interesting_vars:
            del fc.substitutions[interesting_var]
        sol = fc.solve()
    ofc = FeasCircle(m, m.solution)
    for interesting_var in interesting_vars:
        del ofc.substitutions[interesting_var]
    for p in m.as_posyslt1():
        print p.unitstr()
    origfeas = ofc.solve()

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2)

    def plot_uncertainty_set(ax):
        xo, yo = map(mag, map(m.solution, [x, y]))
        ax.plot(xo, yo, "k.")
        x_center = None
        y_center = None
        if rm:
            eta_max_x = np.log(1 + x.key.pr / 100.0)
            eta_min_x = np.log(1 - x.key.pr / 100.0)
            center_x = (eta_min_x + eta_max_x) / 2.0
            eta_max_y = np.log(1 + y.key.pr / 100.0)
            eta_min_y = np.log(1 - y.key.pr / 100.0)
            center_y = (eta_min_y + eta_max_y) / 2.0
            x_center = np.log(xo) + center_x
            y_center = np.log(yo) + center_y
            ax.plot(np.exp(x_center), np.exp(y_center), "kx")
        if rmtype == "elliptical":
            th = np.linspace(0, 2*np.pi, 50)
            ax.plot(np.exp(x_center)*np.exp(np.cos(th))**(np.log(xo) + np.log((1 + x.key.pr/100.0)) - x_center),
                    np.exp(y_center)*np.exp(np.sin(th))**(np.log(yo) + np.log((1 + y.key.pr/100.0)) - y_center), "k",
                    linewidth=1)
        elif rmtype:
            p = Polygon(np.array([[xo*(1 - x.key.pr/100.0)]+[xo*(1 + x.key.pr/100.0)]*2+[xo*(1 - x.key.pr/100.0)],
                                  [yo*(1 - y.key.pr/100.0)]*2 + [yo*(1 + y.key.pr/100.0)]*2]).T,
                        True, edgecolor="black", facecolor="none", linestyle="dashed")
            ax.add_patch(p)

    orig_a, orig_b = map(mag, map(origfeas, [x, y]))
    a_i, b_i, a, b = [None]*4
    if rm:
        a_i, b_i, a, b = map(mag, map(sol, ["x_0^0", "x_1^0", x, y]))
        for i in range(len(a)):
            axes[0].loglog([a_i[i], a[i]], [b_i[i], b[i]], color=GPCOLORS[1], linewidth=0.2)
    else:
        axes[0].loglog([orig_a[0]], [orig_b[0]], "k-")

    from matplotlib.patches import Polygon
    # from matplotlib.collections import PatchCollection

    perimeter = np.array([orig_a, orig_b]).T
    p = Polygon(perimeter, True, color=GPBLU, linewidth=0)
    axes[0].add_patch(p)
    if rm:
        perimeter = np.array([a, b]).T
        p = Polygon(perimeter, True, color=GPCOLORS[1], alpha=0.5, linewidth=0)
        axes[0].add_patch(p)
    plot_uncertainty_set(axes[0])
    axes[0].axis("equal")
    # axes[0].set_ylim([0.1, 1])
    axes[0].set_ylabel(y)

    perimeter = np.array([orig_a, orig_b]).T
    p = Polygon(perimeter, True, color=GPBLU, linewidth=0)
    axes[1].add_patch(p)
    if rm:
        perimeter = np.array([a, b]).T
        p = Polygon(perimeter, True, color=GPCOLORS[1], alpha=0.5, linewidth=0)
        axes[1].add_patch(p)
    plot_uncertainty_set(axes[1])
    # axes[1].set_xlim([0, 6])
    # axes[1].set_ylim([0, 1])
    axes[1].set_xlabel(x)
    axes[1].set_ylabel(y)

    fig.suptitle("%s vs %s feasibility space" % (x, y))
    plt.show()
