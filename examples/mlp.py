import chainer
import chainer.functions as F
import chainer.links as L

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units=10, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, *args):
        x = args[0]
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)
