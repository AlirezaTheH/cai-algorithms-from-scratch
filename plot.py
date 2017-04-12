import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from model import LinearRegression, LogisticRegression
from preprocessing import map_feature


class PlotLinearRegression:

    def __init__(self):
        pass

    @staticmethod
    def plot_data(x, y, x_label, y_label, scatter_label='Unnormalized', title='Input Data'):
        fig, ax = plt.subplots()

        # Scattering
        ax.scatter(x, y, label=scatter_label)

        # Scatter setting
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True)

    @staticmethod
    def plot_convergence(j_seq, x_label='Iterations', y_label='J',
                         line_label='Normalized', title='Gradient Descent Convergence'):

        # figure setting
        fig, ax = plt.subplots()

        # plot line
        ax.plot(j_seq, label=line_label)

        # Plot setting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True)

    @staticmethod
    def plot_cost_function(x, y, theta_seq, j_seq, plot_domain, title='Cost Function', domain_title='an Arbitrary'):

        # Create grid coordinates that don't need to specify mood
        spaced = np.linspace(*plot_domain)
        xx, yy = np.meshgrid(spaced, spaced, indexing='xy')

        # Create grid coordinates for plotting in arbitrary period
        z = np.zeros((plot_domain[2],) * 2)

        # Calculate z-values (cost) based on grid of coefficients
        for (i, j), v in np.ndenumerate(z):
            z[i, j] = LinearRegression.compute_cost(x, y, [[xx[i, j]], [yy[i, j]]])

        # Figure setting
        fig = plt.figure()
        fig.set_size_inches(fig.get_size_inches()*np.array([2,1]))

        # 1 x 2 grid, first subplot
        position = 121

        # Create grid coordinates for plotting in convergence period
        theta = np.meshgrid(theta_seq[:, 0], theta_seq[:, 1], indexing='xy')

        axes = [fig.add_subplot(position+i, projection='3d') for i in range(2)]

        # plot surfaces
        axes[0].plot_surface(theta[0], theta[1], j_seq, alpha=0.6, cmap=plt.cm.jet)
        axes[1].plot_surface(xx, yy, z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)

        # First plot setting
        axes[0].set_xlim(theta[0].min(), theta[0].max())
        axes[0].set_ylim(theta[1].min(), theta[1].max())
        axes[0].set_zlim(j_seq.min(), j_seq.max())
        axes[0].set_title('%s in Convergence Domain' % title)

        # Second plot setting
        axes[1].set_zlim(z.min(), z.max())
        axes[1].view_init(elev=15, azim=230)
        axes[1].set_title('%s in %s Domain' % (title, domain_title))

        # Settings common to both plots
        for ax in axes:
            ax.set_xlabel(r'$\theta_0$')
            ax.set_ylabel(r'$\theta_1$')
            ax.set_zlabel(r'J($\theta$)')

    @staticmethod
    def show():
        plt.show()


class PlotLogisticRegression:

    def __init__(self):
        pass

    @staticmethod
    def plot_data(x, y, label_x, label_y, label_pos, label_neg, title='Input Data',ax=None):
        # Get indexes for class 0 and class 1
        neg = y[:, 0] == 0
        pos = y[:, 0] == 1

        # If no specific axes object has been passed, create a new one
        if ax is None:
            fig, ax = plt.subplots()

        # Scattering
        ax.scatter(x[pos][:, 0], x[pos][:, 1], marker='+', s=60, linewidth=2, label=label_pos)
        ax.scatter(x[neg][:, 0], x[neg][:, 1], label=label_neg)

        # Scatter setting
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True)

    @classmethod
    def plot_learned_model(cls, x, y, theta, map_feature_degree, labels, title='Learned Model'):
        fig, ax = plt.subplots()
        cls.plot_data(x, y, *labels, title=title, ax=ax)
        xx1, xx2 = np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max()), np.linspace(x[:, 1].min(), x[:, 1].max()))
        sigmoid = LogisticRegression.sigmoid
        if map_feature_degree is None:
            h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
        else:
            h = sigmoid(map_feature(np.c_[xx1.ravel(), xx2.ravel()], map_feature_degree).dot(theta))
        h = h.reshape(xx1.shape)
        contour = ax.contour(xx1, xx2, h, [0.5])
        contour.collections[0].set_label('Decision Boundary')
        ax.legend(frameon=True, fancybox=True)

    @staticmethod
    def show():
        plt.show()