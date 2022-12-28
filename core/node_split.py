
class NumericAttributeBinaryTest():
    """ Teste a ser feito pelo nó de decisão"""

    def __init__(self, att_idx, att_value, equal_passes_test):
        self._att_idx = att_idx
        self._att_value = att_value
        self._equals_passes_test = equal_passes_test

    def branch_for_instance(self, X):
        """ ... """
        v = X[self._att_idx]
        if v == self._att_value:
            return 0 if self._equals_passes_test else 1
        return 0 if v < self._att_value else 1


class SplitNode():
    """ Nó de decisão da árvore """

    def __init__(self, split_test, stats):
        self._stats = stats  # Estatísticas do nó
        self._split_test = split_test  # Teste associado ao nó
        self._children = {}  # Próximo nível

    def set_child(self, index, node):
        """ ... """
        self._children[index] = node

    def get_child(self, index):
        """ ... """
        return self._children[index]

    def instance_child_index(self, X):
        """ ... """
        return self._split_test.branch_for_instance(X)

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ ... """
        child_index = self.instance_child_index(X)
        child = self.get_child(child_index)
        return child.filter_instance_to_leaf(X, self, child_index)
