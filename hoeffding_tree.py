from .core.node_leaf import LeafNBA
from .core.node_split import SplitNode
from operator import attrgetter
import numpy as np
import copy


class HoeffdingTreeClassifier():

    def __init__(self):
        self._tree_root = None
        self.grace_period = 200
        self.split_confidence = 0.0000001
        self.tie_threshold = 0.05

    def partial_fit(self, X, y, sample_weight=1):
        """ Trains the model on samples X and corresponding targets y. """

        if self._tree_root is None:
            self._tree_root = self._new_learning_node()

        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
        leaf_node = found_node.node

        # Cria uma folha caso não exista uma criada
        if leaf_node is None:
            leaf_node = self._new_learning_node()
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
        leaf_node.learn_one(X, y, weight=sample_weight)

        weight_seen = leaf_node.total_weight()
        weight_diff = weight_seen - leaf_node.last_split_attempt_at

        if weight_diff >= self.grace_period:
            self._attempt_to_split(leaf_node,
                                   found_node.parent,
                                   found_node.parent_branch)
            leaf_node.last_split_attempt_at = weight_seen

    def _new_learning_node(self, initial_class_observations=None):
        """ Create a new learning node. """
        return LeafNBA(initial_class_observations)

    def _attempt_to_split(self, leaf_node, parent: SplitNode, parent_idx: int):
        """ Attempt to split a node."""

        best_split_suggestions = leaf_node.get_best_split_suggestions()
        best_split_suggestions.sort(key=attrgetter('merit'))

        should_split = False
        if len(best_split_suggestions) < 2:
            should_split = len(best_split_suggestions) > 0
        else:
            n_samples = leaf_node.total_weight()
            class_seen = len(list(leaf_node.get_stats().keys()))
            range_val = np.log2(class_seen) if class_seen > 2 else np.log2(2)
            hoeffding_bound = self._hoeffding_bound(
                range_val=range_val,
                confidence=self.split_confidence,
                n=n_samples)
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]

            if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound) or (
                    hoeffding_bound < self.tie_threshold):
                should_split = True

        if should_split:
            split_decision = best_split_suggestions[-1]

            new_split = SplitNode(
                split_decision.split_test, leaf_node.get_stats())

            for i in [0, 1]:
                new_child = self._new_learning_node(
                    split_decision.resulting_stats_from_split(i))
                new_split.set_child(i, new_child)

            if parent is None:
                self._tree_root = new_split
            else:
                parent.set_child(parent_idx, new_split)

    def _hoeffding_bound(self, range_val, confidence, n):
        """ ... """
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))

    def _get_votes_for_instance(self, X):
        """ Get class votes for a single instance. Returns dict (class_value, weight) """

        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            return leaf_node.predict_one(X) \
                if not isinstance(leaf_node, SplitNode) else leaf_node.stats
        else:
            return {}

    def predict(self, X):
        """ Predicts the label of the X instance(s)"""
        y_proba = self.predict_proba(X)
        return int(max(y_proba, key=y_proba.get))

    def predict_proba(self, X):
        vote = self._get_votes_for_instance(X)
        if vote == {}:
            return {0: 1}
        else:
            dictionary = copy.deepcopy(vote)
            soma = sum(vote.values())
            if soma != 0:
                for key, value in dictionary.items():
                    dictionary[key] = value / soma

            return dictionary
