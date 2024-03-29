import tensorflow as tf

import re


class Tree:
    def __init__(self, string):
        string = string.strip()
        if string.find('[') < 0:  # Leaf root
            self.root = parse_node(string).copy()
        else:
            self.root = None
            lst = re.split('[\[,\]]+', string)[:-1]
            for l in lst:
                if not self.root:
                    self.root = parse_node(l)
                    continue
                self.__add_node(self.root, parse_node(l))

    def get(self, index):
        i = [index]
        return self.root.get(i)

    def build(self, input):
        return self.root.build(input)

    def update(self, session):
        self.root.update(session)

    def size(self):
        return self.root.size()

    def __add_node(self, parent, node):
        for i in range(parent.children_num):
            if not parent.children[i]:
                parent.children[i] = node
                return True
            if self.__add_node(parent.children[i], node):  # Recurse into children.
                return True
        return False

    def to_string(self):
        return self.root.to_string()


class Node:
    def __init__(self, name, children_num, build_method):
        self.name = name
        self.children_num = children_num
        self.children = [None]*children_num
        self.build_method = build_method

    def copy(self):
        n = Node(self.name, self.children_num, self.build_method)
        n.children = []
        for c in self.children:
            if c:
                n.children.append(c.copy())
            else:
                n.children.append(None)
        return n

    def get(self, index):
        if index[0] == 0:
            return self
        index[0] -= 1
        for c in self.children:
            v = c.get(index)
            if v:
                return v
        return None

    def swap(self, node):
        self.name, node.name = node.name, self.name
        self.children_num, node.children_num = node.children_num, self.children_num
        self.children, node.children = node.children, self.children
        self.build_method, node.build_method = node.build_method, self.build_method

    def build(self, input):
        return self.build_method(self.children, input)

    def update(self, session):
        for c in self.children:
            c.update(session)

    def size(self):
        s = 1
        for c in self.children:
            s += c.size()
        return s

    def to_string(self):
        if len(self.children) == 0:
            return self.name

        s = self.name + "["
        t = False
        for c in self.children:
            if t:
                s += ','
            s += c.to_string()
            t = True
        return s + "]"

class _ConstNode(Node):
    def __init__(self, value):
        super().__init__("const", 0, lambda c, input: tf.constant(value))
        self.value = value

    def swap(self, node):
        super().swap(node)
        self.value, node.value = node.value, self.value

    def to_string(self):
        return str(self.value)

class _LearnableNode(Node):
    __VARIABLE_INDEX = 0

    def __init__(self, value):
        self.var_index = _LearnableNode.__VARIABLE_INDEX
        self.value = value

        super().__init__('l'+str(value), 0, lambda c, input: tf.get_variable('acti_var' + str(self.var_index), shape=input.get_shape()[-1],
                                         initializer=tf.constant_initializer(value), collections=[tf.GraphKeys.TRAINABLE_VARIABLES], dtype=tf.float32))
        _LearnableNode.__VARIABLE_INDEX += 1

    def swap(self, node):
        super().swap(node)
        self.var_index, node.var_index = node.var_index, self.var_index
        self.value, node.value = node.value, self.value

    def update(self, session):
        self.value = tensorflow.contrib.framework.get_variables('acti_var'+str(self.var_index))[0].eval(session=session)

    def to_string(self):
        return 'l'+str(self.value)


def parse_node(s):  # String to node
    try:
        float(s)  # Constant.
        return _ConstNode(float(s))
    except ValueError:
        pass
    if re.match('l[-]*[0-9.]+', s):  # Learnable node
        try:
            float(s[1:])  # Constant.
            return _LearnableNode(float(s[1:]))
        except ValueError:
            return None
    if s in nodes:
        return nodes[s].copy()
    else:
        raise KeyError('Unknown key: '+s)

nodes = {
    ## Binary
    '+': Node('+', 2, lambda c, input: tf.add(c[0].build(input), c[1].build(input))),
    '-': Node('-', 2, lambda c, input: tf.subtract(c[0].build(input), c[1].build(input))),
    '*': Node('*', 2, lambda c, input: tf.multiply(c[0].build(input), c[1].build(input))),
    '/': Node('/', 2, lambda c, input: tf.divide(c[0].build(input), c[1].build(input))),
    'min': Node('min', 2, lambda c, input: tf.minimum(c[0].build(input), c[1].build(input))),
    'max': Node('max', 2, lambda c, input: tf.maximum(c[0].build(input), c[1].build(input))),
    'pow': Node('pow', 2, lambda c, input: tf.pow(c[0].build(input), c[1].build(input))),
    ## Unary
    'abs': Node('abs', 1, lambda c, input: tf.abs(c[0].build(input))),
    'neg': Node('neg', 1, lambda c, input: tf.negate(c[0].build(input))),
    'sin': Node('sin', 1, lambda c, input: tf.sin(c[0].build(input))),
    'cos': Node('cos', 1, lambda c, input: tf.cos(c[0].build(input))),
    'tan': Node('tan', 1, lambda c, input: tf.tan(c[0].build(input))),
    'exp': Node('exp', 1, lambda c, input: tf.exp(c[0].build(input))),
    'pow2': Node('pow2', 1, lambda c, input: tf.square(c[0].build(input))),
    'pow3': Node('pow3', 1, lambda c, input: tf.pow(c[0].build(input), tf.constant(3.))),
    'log': Node('log', 1, lambda c, input: tf.log(c[0].build(input))),
    'gauss': Node('gauss', 1, lambda c, input: tf.exp(tf.negate(tf.square(c[0].build(input))))),
    'sigmoid': Node('sigmoid', 1, lambda c, input: tf.sigmoid(c[0].build(input))),
    'swish': Node('swish', 1, lambda c, input: tf.nn.swish(c[0].build(input))),
    # Relus
    'relu': Node('relu', 1, lambda c, input: tf.nn.relu(c[0].build(input))),
    'relu6': Node('relu6', 1, lambda c, input: tf.nn.relu6(c[0].build(input))),  #
    'lrelu': Node('lrelu', 1, lambda c, input: tf.nn.leaky_relu(c[0].build(input))),
    'selu': Node('selu', 1, lambda c, input: tf.nn.selu(c[0].build(input))),
    'elu': Node('elu', 1, lambda c, input: tf.nn.elu(c[0].build(input))),
    'prelu': Node('prelu', 1, lambda c, input: prelu(input)),
    # Softies
    'softmax': Node('softmax', 1, lambda c, input: tf.nn.softmax(c[0].build(input))),  #
    'softplus': Node('softplus', 1, lambda c, input: tf.nn.softplus(c[0].build(input))),
    'softsign': Node('softsign', 1, lambda c, input: tf.nn.softsign(c[0].build(input))),
    # Hyperbolic
    'sinh': Node('sinh', 1, lambda c, input: tf.sinh(c[0].build(input))),  #
    'cosh': Node('cosh', 1, lambda c, input: tf.cosh(c[0].build(input))),  #
    'tanh': Node('tanh', 1, lambda c, input: tf.tanh(c[0].build(input))),
    # Input
    'x': Node('x', 0, lambda c, input: input)
}


def prelu(x):
    alpha = parse_node('l0')
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    return pos + neg


class TreeActivationLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(CustomActivation, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_variable("kernel",
                                    shape=[int(input_shape[-1]),
                                           self.num_outputs])

  def call(self, input):
    pass # return tf.function(tree.exeute)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Define target function.
    func_target = lambda x: np.exp(-(x - 3.14) ** 2)
    # Define model function.
    func_model = 'relu[-[1,abs[-[x,l1.0]]]]'

    # Generate data.
    x = np.linspace(-5, 5, 300).reshape([-1, 1])
    y = np.array([func_target(t) for t in x]).reshape([-1, 1])

    # Build tree.
    tree = Tree(func_model)
    tf_truth = tf.placeholder(tf.float32, [None, 1])
    tf_inp = tf.placeholder(tf.float32, [None, 1])
    tf_out = tree.build(tf_inp)

    tf_loss = tf.reduce_sum((tf_truth - tf_out) ** 2)
    tf_optimi = tf.train.GradientDescentOptimizer(0.01).minimize(tf_loss)
    tf_sess = tf.Session()
    tf_sess.run(tf.global_variables_initializer())

    # Train learnable nodes.
    ls_prev = -1
    for i in range(1,101):
        ls, _ = tf_sess.run([tf_loss, tf_optimi], {tf_inp: x, tf_truth: y})
        print('Iter', i, 'has loss:', ls)
        if abs(ls_prev-ls) < 1e-12:
            break
        ls_prev = ls
    tree.update(tf_sess)

    # Collect final predictions.
    p = tf_sess.run([tf_out], {tf_inp: x})
    p = np.array(p).reshape(-1)

    # Plot.
    plt.figure(figsize=(10,4))
    plt.plot(x, y, 'b', label='Original function')
    plt.plot(x, p, 'r', label=tree.to_string())
    plt.legend()
    plt.show()
