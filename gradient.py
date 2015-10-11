from abc import ABCMeta, abstractmethod
import numpy as np
import time
import matplotlib.pyplot as plt


def calculate_gradient(func, x, dx=0.00000001):
    """
    Calculates gradient of a function
    :param func: function
    :param x: x as numpy.array
    :param dx: controls level of precision
    :return: gradient as numpy.array
    """
    grad = np.zeros(len(x))
    for i in range(len(x)):
        delta = np.zeros(len(x))
        delta[i] = dx
        grad[i] = (func(x + delta) - func(x - delta)) / (2 * dx)
    return grad


class AbstractGradient(object):
    """
    This is an abstract class for various gradient descent classes. Other
    ones should inherit from this one.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def iterate(self):
        """
        Makes an iteration of gradient descent
        """
        raise NotImplementedError("You forgot to implement the 'iterate' method")

    def initialize_plot(self):
        plt.ion()
        self.figure, self.f = plt.subplots(2 + len(self.x), sharex=True)
        self.lines = [0] * (len(self.x) + 2)
        self.lines[0], = self.f[0].plot(self.iter_count, self.func_value, 'k')
        self.f[0].set_title('f(x)')
        self.lines[1], = self.f[1].plot(self.iter_count, self.grad_norm, 'k')
        self.f[1].set_title('Gradient norm')

        for i in range(len(self.x)):
            self.lines[2 + i], = self.f[2 + i].plot(self.iter_count, self.x[i])
            self.f[2 + i].set_title("x%s" % (i + 1))
            self.f[2 + i].set_autoscalex_on(True)
            self.f[2 + i].set_autoscaley_on(True)

        self.f[0].set_autoscalex_on(True)
        self.f[0].set_autoscaley_on(True)
        self.f[1].set_autoscalex_on(True)
        self.f[1].set_autoscaley_on(True)
        time.sleep(0.5)

    def update_plot(self):
        self.lines[0].set_xdata(np.append(self.lines[0].get_xdata(), self.iter_count))
        self.lines[0].set_ydata(np.append(self.lines[0].get_ydata(), self.func_value))
        self.lines[1].set_xdata(np.append(self.lines[1].get_xdata(), self.iter_count))
        self.lines[1].set_ydata(np.append(self.lines[1].get_ydata(), self.grad_norm))

        for i in range(len(self.x)):
            self.lines[2 + i].set_xdata(np.append(self.lines[2 + i].get_xdata(), self.iter_count))
            self.lines[2 + i].set_ydata(np.append(self.lines[2 + i].get_ydata(), self.x[i]))

        for i in range(len(self.x) + 2):
            self.f[i].relim()
            self.f[i].autoscale_view()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def run(self):
        """
        Runs iterations of gradient descent until a stopping condition is
        fulfilled
        """
        if self.plot:
            self.initialize_plot()
        while (self.grad_norm > self.precision) and (self.iter_count < self.max_iterations):
            self.iterate()
            if self.plot and (self.iter_count % self.show_every == 0):
                self.update_plot()

        if self.iter_count == self.max_iterations:
            print("Max iterations limit reached")
            print(self)
        else:
            print("Iterations finished successfully!")
            print(self)

        if self.plot:
            self.show_plot()

    @staticmethod
    def show_plot():
        plt.ioff()
        plt.show()


class SimpleGradient(AbstractGradient):
    """
    Implements simple gradient descent
    """

    def __init__(self, func, x_start, gamma=0.5, precision=0.00001,
                 max_iterations=500, plot=False, show_every=20):
        self.func = func
        self.func_value = func(x_start)
        self.x = x_start
        self.gamma = gamma
        self.precision = precision
        self.grad = calculate_gradient(func, x_start)
        self.grad_norm = np.linalg.norm(self.grad)
        self.iter_count = 0
        self.max_iterations = max_iterations
        self.plot = plot
        self.show_every = show_every

    def iterate(self):
        """
        Makes an iteration of gradient descent
        """
        self.x = self.x - self.gamma * self.grad
        self.func_value = self.func(self.x)
        self.grad = calculate_gradient(self.func, self.x)
        self.grad_norm = np.linalg.norm(self.grad)
        self.iter_count += 1

    def __str__(self):
        return "Method: Simple gradient\nIterations made: %s\nCurrent values:\nx: %s\
                \ngradient: %s\nf: %s" \
               % (self.iter_count, self.x, self.grad, self.func_value)


class HardBall(AbstractGradient):
    """
    Implements hard ball method gradient descent
    """

    def __init__(self, func, x_start, alpha=0.5, beta=0.5, precision=0.00001,
                 max_iterations=500, plot=False, show_every=20):
        self.func = func
        self.func_value = func(x_start)
        self.x = x_start
        self.x_prev = x_start
        self.delta_x = 0
        self.alpha = alpha
        self.beta = beta
        self.precision = precision
        self.grad = calculate_gradient(func, x_start)
        self.grad_norm = np.linalg.norm(self.grad)
        self.iter_count = 0
        self.max_iterations = max_iterations
        self.plot = plot
        self.show_every = show_every

    def iterate(self):
        """
        Makes an iteraion of gradient descent
        """
        self.x_prev = self.x
        self.x = self.x - self.alpha * self.grad + self.beta * self.delta_x
        self.delta_x = self.x - self.x_prev
        self.func_value = self.func(self.x)
        self.grad = calculate_gradient(self.func, self.x)
        self.grad_norm = np.linalg.norm(self.grad)

        self.iter_count += 1

    def __str__(self):
        return "Method: Hard Ball\nIterations made: %s\nCurrent values:\nx: %s\
                \ngradient: %s\nf: %s" \
               % (self.iter_count, self.x, self.grad, self.func_value)


def f(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


x = np.array([-1, 1])
trial = HardBall(f, x, alpha=0.001, plot=True, show_every=200, max_iterations=100000)
trial.run()
