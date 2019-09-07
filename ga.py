from utils import Stopwatch, roulette_wheel_select
import crossovers as crx
import random


def will_occur(prob):
    return random.random() < prob


class GA:
    def __init__(self, params):
        self.params = params

        # TODO Parameter checks.

    def __get_best(self):
        return self.params['best_getter'].get_best(self.population)

    def __crossover(self, parents):
        return roulette_wheel_select(self.params['crossovers']).cross(parents)

    def __mutate(self, child):
        roulette_wheel_select(self.params['mutators']).mutate(child)

    def _initialize(self):
        self.population = []
        for _ in range(self.params['population_size']):
            g =  self.params['genotype_template'].copy()
            self.params['initializer'].initialize(g)
            self.params['evaluator'].evaluate(g)
            self.population.append(g)

    def _run_step(self):
        pop_size = len(self.population)
        new_pop = []
        if self.params['elitism']:  # Save the queen.
            new_pop.append(__get_best().clone())
            pop_size -= 1

        for i in range(pop_size):
            parents = self.params['selector'](self.population, [g.fitness for g in self.population], 2)
            children = __crossover(parents)
            for c in children:
                if will_occur(self.params['mutation_prob']):
                    __mutate(c)

    def run(self):
        self.iteration = 0
        self.best_unit = None
        self.evaluations = 0
        self.elapsed_time = 0
        stopwatch = Stopwatch()

        stop_cond = self.params['stop_condition']

        print('===> Initializing population!')
        stopwatch.start()
        self._initialize()
        best_unit = __get_best().copy()
        print('===> Done! ({})'.format(format_stopwatch(stopwatch)))

        print('===> Starting algorithm with population of {} units!'.format(len(self.population)))
        while not stop_cond.is_satisfied(self):
            self.iteration += 1
            self._run_step()
        print('===> Done! ({})'.format(format_stopwatch(stopwatch)))
        print(stop_cond.report(self))


class Initializer:
    def initialize(self, g):
        g.tree = None
        raise Exception('Missing implementation!')


class Evaluator:
    def evaluate(self, g):
        g.fitness = None
        raise Exception('Missing implementation!')


class StopCondition:
    def __init__(self, params):
        self.max_iterations = params.get('max_iterations', None)
        self.max_evaluations = params.get('max_evaluations', None)
        self.max_time = params.get('max_time', None)
        self.min_fitness = params.get('min_fitness', None)
        self.max_fitness = params.get('max_fitness', None)

    def is_satisfied(self, algo):
        return (self.max_iterations and self.max_iterations <= algo.iteration) or \
               (self.min_fitness and self.min_fitness >= algo.best_unit.fitness) or \
               (self.max_fitness and self.max_fitness <= algo.best_unit.fitness) or \
               (self.max_evaluations and self.max_evaluations <= algo.evaluations) or \
               (self.max_time and self.max_time <= algo.elapsed_time)

    def report(self, algo):
        s = ''
        if self.max_iterations and self.max_iterations <= algo.iteration:
            s += 'Max iterations achieved!\n'
        if self.min_fitness and self.min_fitness >= algo.best_unit.fitness:
            s += 'Min fitness achieved!\n'
        if self.max_fitness and self.max_fitness <= algo.best_unit.fitness:
            s += 'Max fitness achieved!\n'
        if self.max_evaluations and self.max_evaluations <= algo.evaluations:
            s += 'Max evaluations achieved!\n'
        if self.max_time and self.max_time <= algo.elapsed_time:
            s += 'Max time achieved!\n'
        if not s:
            return 'No condition satisfied!'
        return s


class Genotype:
    def __init__(self):
        self.fitness = None
        self.tree = None

    def __init__(self, tree):
        self.fitness = None
        self.tree = tree

    def get(self, index):
        return self.tree.get(index)

    def set(self, index, value):
        n = self.tree.get(index)
        n.swap(value)

    def size(self):
        return self.tree.size()

    def clone(self):
        return Genotype(self.tree)


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf

    # Target function.
    func = lambda x: np.sin(x**2)

    # Generate samples.
    X = np.linspace(-5, 5, 1000).reshape(-1)
    Y = np.array([func(x) for x in X]).reshape(-1)

    # Define evaluator (for tree).
    def evaluator(tree):
        # Build tf graph.
        tf_truth = tf.placeholder(tf.float32, [None, 1])
        tf_inp = tf.placeholder(tf.float32, [None, 1])
        tf_out = tree.build(tf_inp)

        tf_loss = tf.reduce_sum((tf_truth - tf_out) ** 2)
        tf_optimi = tf.train.GradientDescentOptimizer(0.01).minimize(tf_loss)
        tf_sess = tf.Session()
        tf_sess.run(tf.initialize_all_variables())

        # Train learnable nodes.
        ls_prev = -1
        for i in range(1,101):
            ls, _ = tf_sess.run([tf_loss, tf_optimi], {tf_inp: x, tf_truth: y})
            print('Iter', i, 'has loss:', ls)
            if abs(ls_prev - ls) < 1e-12:
                break
            ls_prev = ls
        tree.update(tf_sess)
