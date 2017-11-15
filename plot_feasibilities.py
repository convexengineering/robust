import numpy as np
from gpkit import Model, Variable, ConstraintSet, GPCOLORS, GPBLU
from gpkit.small_scripts import mag


def plot_feasibilities(x, y, m, rm=None, rmtype=None):
    class FeasCircle(Model):
        "SKIP VERIFICATION"
        def setup(self, m, sol):
            r = 4
            th = Variable("\\theta", np.linspace(0, 2*np.pi, 120), "-")
            x_i = Variable("x_i", lambda c: sol(x)*np.exp(r*np.cos(c[th])), x.unitstr(), "starting point in x")
            y_i = Variable("y_i", lambda c: sol(y)*np.exp(r*np.sin(c[th])), y.unitstr(), "starting point in y")
            s_x = Variable("s_x", "-", "slack in x")
            s_y = Variable("s_y", "-", "slack in y")

            self.cost = s_x**2 + s_y**2
            feas_slack = ConstraintSet(
                [s_x >= 1, s_y >= 1,
                 m[x]/s_x <= x_i, x_i <= m[x]*s_x,
                 m[y]/s_y <= y_i, y_i <= m[y]*s_y])

            return [m, feas_slack], {k: v for k, v in sol["freevariables"].items()
                                     if k in m.varkeys}

    # plot original feasibility set
    # plot boundary of uncertainty set
    if rm:
        fc = FeasCircle(m, rm.solution)
        del fc.substitutions[x]
        del fc.substitutions[y]
        sol = fc.solve()
    ofc = FeasCircle(m, m.solution)
    del ofc.substitutions[x]
    del ofc.substitutions[y]
    origfeas = ofc.solve()

    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2)

    def plot_uncertainty_set(ax):
        xo, yo = map(mag, map(m.solution, [x, y]))
        ax.plot(xo, yo, "k.")
        if rm:
            eta_max_x = np.log(1 + x.key.pr / 100.0)
            eta_min_x = np.log(1 - x.key.pr / 100.0)
            center_x = (eta_min_x + eta_max_x) / 2.0
            eta_max_y = np.log(1 + y.key.pr / 100.0)
            eta_min_y = np.log(1 - y.key.pr / 100.0)
            center_y = (eta_min_y + eta_max_y) / 2.0
            x_center = xo*np.exp(center_x)
            y_center = yo*np.exp(center_y)
            ax.plot(x_center, y_center, "kx")
        if rmtype == "elliptical":
            th = np.linspace(0, 2*np.pi, 50)
            ax.plot(xo*np.exp(center_x)*np.exp(np.cos(th))**np.log((1 + x.key.pr/100.0)),
                    yo*np.exp(center_y)*np.exp(np.sin(th))**np.log((1 + y.key.pr/100.0)), "k--",
                    linewidth=1)
        elif rmtype:
            p = Polygon(np.array([[xo*np.exp(center_x)/(1 + x.key.pr/100.0)]+[xo*np.exp(center_x)*(1 + x.key.pr/100.0)]*2+[xo*np.exp(center_x)/(1 + x.key.pr/100.0)],
                                  [yo*np.exp(center_y)/(1 + y.key.pr/100.0)]*2 + [yo*np.exp(center_y)*(1 + y.key.pr/100.0)]*2]).T,
                        True, edgecolor="black", facecolor="none", linestyle="dashed")
            ax.add_patch(p)

    orig_a, orig_b = map(mag, map(origfeas, [x, y]))
    if rm:
        a_i, b_i, a, b = map(mag, map(sol, ["x_i", "y_i", x, y]))
        for i in range(len(a)):
            axes[0].loglog([a_i[i], a[i]], [b_i[i], b[i]], color=GPCOLORS[1], linewidth=0.2)
    else:
        axes[0].loglog([orig_a[0]], [orig_b[0]], "k-")

    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

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
