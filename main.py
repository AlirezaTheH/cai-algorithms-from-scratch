import numpy as np

from scipy.optimize import fmin_bfgs
from model import LinearRegression, LogisticRegression
from plot import PlotLinearRegression, PlotLogisticRegression
from preprocessing import normalize_data, map_feature


def run():
    np.seterr(all='ignore')

    # file fields
    name = 'name'
    converters = 'converter'
    delimiter = 'delimiter'
    is_regular = 'is_regular'
    labels = 'labels'
    plot_domain = 'plot_domain'

    # moods
    u = 'unnormal'
    n = 'normal'

    def load_data(f):
        x = dict()
        data = np.loadtxt('data/%s.txt' % f[name], converters=f[converters], delimiter=f[delimiter])
        x[u] = data[:, 0:-1]
        x[n] = np.c_[np.ones(data.shape[0]), normalize_data(data[:, 0:-1])]
        y = np.c_[data[:, -1]]

        return x, y

    lin_reg_files = [{name: 'lin_reg_data1',
                      converters: None,
                      delimiter: ',',
                      labels: ('x', 'y'),
                      plot_domain: (-10, 10, 50)},
                     {name: 'lin_reg_data2',
                      converters: {0: int, 1: int, 2: int},
                      delimiter: None},
                     {name: 'lin_reg_data3',
                      converters: {5: int, 6: int, 7: int, 8: int, 9: int, 10: int},
                      delimiter: None}]

    logistic_files = [{name: 'logistic_data1',
                       converters: {2: int},
                       delimiter: ',',
                       is_regular: True,
                       labels: ('Exam 1 score',
                                'Exam 2 score',
                                'Admitted',
                                'Not admitted')},
                      {name: 'logistic_data2',
                       converters: {2: int},
                       delimiter: ',',
                       is_regular: False,
                       labels: ('Microchip test 1',
                                'Microchip test 2',
                                'Passed',
                                'Not Passed')}]

    print '======================= Linear Regression ======================='
    for lrf in lin_reg_files:
        x, y = load_data(lrf)
        linear = LinearRegression
        theta, theta_seq, j_seq = linear.gradient_descent(x[n], y)

        print '-----------------------------------------------------------------'
        print 'file name: %s.txt' % lrf[name]
        print 'theta values that minimize cost function:'
        print '\n\t\t%s' % \
              '\n\t\t'.join(['theta_%s = %s' % (i, t) for i, t in zip(range(theta.size), theta.T[0])])
        print '-----------------------------------------------------------------'

        plot_linear = PlotLinearRegression
        if x[u].shape[1] == 1:
            plot_linear.plot_data(x[u], y, *lrf[labels])
            plot_linear.plot_cost_function(x[n], y, theta_seq, j_seq, lrf[plot_domain])

        plot_linear.plot_convergence(j_seq)
        plot_linear.show()
    print '================================================================='

    print '====================== Logistic Regression ======================'
    for lf in logistic_files:
        x, y = load_data(lf)
        plot = PlotLogisticRegression
        plot.plot_data(x[u], y, *lf[labels])
        plot.show()

        logistic_regression = LogisticRegression
        x_mapped = x[n]
        map_feature_degree = None
        if not lf[is_regular]:
            x_mapped = map_feature(x[n][:, 1:3], 6)
            map_feature_degree = 6
        theta = np.zeros(x_mapped.shape[1])
        theta = fmin_bfgs(logistic_regression.compute_cost,
                          theta,
                          logistic_regression.gradient,
                          (x_mapped, y),
                          maxiter=400,
                          disp=True)

        print '-----------------------------------------------------------------'
        print 'file name: %s.txt' % lf[name]
        print 'theta values that minimize cost function:'
        print '\n\t\t%s' % \
              '\n\t\t'.join(['theta_%s = %s' % (i, t) for i, t in zip(range(theta.size), theta)])
        print '-----------------------------------------------------------------'

        plot.plot_learned_model(x[n][:, 1::], y, theta, map_feature_degree, lf[labels])
        plot.show()
    print '================================================================='


if __name__ == '__main__':
    run()
