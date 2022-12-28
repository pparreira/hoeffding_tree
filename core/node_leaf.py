from .gaussian_estimator import GaussianEstimator
from .node_split import NumericAttributeBinaryTest
from .statistics import do_naive_bayes_prediction
import numpy as np

from sortedcontainers.sortedlist import SortedList


class FoundLeaf(object):
    """ Ret """

    def __init__(self, node=None, parent=None, parent_branch=None, depth=None):
        self.node = node
        self.parent = parent
        self.parent_branch = parent_branch
        self.depth = depth


class AttributeSplitSuggestion(object):
    """ ... """

    def __init__(self, split_test, resulting_class_distributions, merit):
        """ ... """
        self.split_test = split_test
        self.resulting_class_distributions = resulting_class_distributions
        self.merit = merit

    def num_splits(self):
        """ ... """
        return len(self.resulting_class_distributions)

    def resulting_stats_from_split(self, split_idx):
        """ ... """
        return self.resulting_class_distributions[split_idx]


class NumericAttributeObserverGaussian():
    """ Classe com a distrubição dos atributos númericos por classe"""

    def __init__(self):
        self._min_value_observed_per_class = {}  # Valor mínimo do atributo por classe
        self._max_value_observed_per_class = {}  # Valor máximos do atributo por classe
        self._att_val_dist_per_class = {}  # Distribuição do atributo por classe
        self.num_bin_options = 10  # The number of bins, default 10

    def update(self, att_val, class_val, weight):
        """ Atualiza a distrubição do atributo dada a classe """

        # Nova classe aparaceu
        if class_val not in self._att_val_dist_per_class.keys():
            self._att_val_dist_per_class[class_val] = GaussianEstimator()
            self._min_value_observed_per_class[class_val] = att_val
            self._max_value_observed_per_class[class_val] = att_val

        # Atualizações dos valores maximos e mínimos
        if att_val < self._min_value_observed_per_class[class_val]:
            self._min_value_observed_per_class[class_val] = att_val
        if att_val > self._max_value_observed_per_class[class_val]:
            self._max_value_observed_per_class[class_val] = att_val

        # Atualização do estimador com o valor/peso
        val_dist = self._att_val_dist_per_class[class_val]
        val_dist.add_observation(att_val, weight)

        self._att_val_dist_per_class = dict(
            sorted(self._att_val_dist_per_class.items()))
        self._max_value_observed_per_class = \
            dict(sorted(self._max_value_observed_per_class.items()))
        self._min_value_observed_per_class = \
            dict(sorted(self._min_value_observed_per_class.items()))

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        """ ..."""
        if class_val in self._att_val_dist_per_class:
            obs = self._att_val_dist_per_class[class_val]
            return obs.probability_density(att_val)
        else:
            return 0.0

    def get_split_point_suggestions(self):
        """ Retorna sugestões de pontos para a criação de um nó de decisão """

        suggested_split_values = SortedList()
        min_value = min(self._min_value_observed_per_class.values())
        max_value = max(self._max_value_observed_per_class.values())

        if min_value < np.inf:
            bin_size = (max_value - min_value) / \
                (float(self.num_bin_options) + 1.0)

            for i in range(self.num_bin_options):
                split_value = min_value + (bin_size * (i + 1))
                if min_value < split_value < max_value:
                    suggested_split_values.add(split_value)

        return suggested_split_values

    def get_class_dists_from_binary_split(self, split_value):
        """ Dado o valor sugerido, estima a quantidade
        de instâncias a direia e esquerda por classe """

        left_dist = {}
        right_dist = {}

        for k, estimator in self._att_val_dist_per_class.items():
            if split_value < self._min_value_observed_per_class[k]:
                right_dist[k] = estimator.get_total_weight_observed()
            elif split_value >= self._max_value_observed_per_class[k]:
                left_dist[k] = estimator.get_total_weight_observed()
            else:
                weight_dist = estimator.estimated_weights_in_split(split_value)
                left_dist[k] = weight_dist[0] + \
                    weight_dist[1]  # P(X <= x)*weight
                right_dist[k] = weight_dist[2]  # P(X > x)*weight

        return [left_dist, right_dist]

    def compute_entropy_dict(self, dist):
        """ ... """
        entropy = 0.0
        dis_sums = 0.0
        for _, weight in dist.items():
            if weight > 0.0:
                entropy -= weight * np.log2(weight)
                dis_sums += weight
        return (entropy + dis_sums * np.log2(dis_sums)) / dis_sums if dis_sums > 0.0 else 0.0

    def compute_entropy(self, dists):
        """ ... """
        total_weight = 0.0
        dist_weights = [0.0]*len(dists)
        for i in range(len(dists)):
            dist_weights[i] = sum(dists[i].values())
            total_weight += dist_weights[i]
        entropy = 0.0
        for i in range(len(dists)):
            entropy += dist_weights[i] * self.compute_entropy_dict(dists[i])
        return entropy / total_weight

    def get_infogain_split(self, pre_split_dist, post_split_dist):
        """ ... """
        return self.compute_entropy([pre_split_dist]) - self.compute_entropy(post_split_dist)

    def get_best_evaluated_split_suggestion(self, pre_split_dist, att_idx):
        """ Obtém a melhor sugestão de split """
        point_suggestions = self.get_split_point_suggestions()

        sugestions_split = {}
        for point_suggestion in point_suggestions:
            class_dists = self.get_class_dists_from_binary_split(
                point_suggestion)
            sugestions_split[point_suggestion] = self.get_infogain_split(
                pre_split_dist, class_dists)

        if len(sugestions_split) == 0:
            return None

        better_split = max(sugestions_split, key=sugestions_split.get)
        if sugestions_split[better_split] is not None:
            post_split_dist = self.get_class_dists_from_binary_split(
                better_split)
            return AttributeSplitSuggestion(
                split_test=NumericAttributeBinaryTest(
                    att_idx, better_split, True),
                resulting_class_distributions=post_split_dist,
                merit=sugestions_split[better_split])
        else:
            return None


class LeafNBA():
    """ Folha contendo o modelo NBA"""

    def __init__(self, initial_stats=None):
        if initial_stats is None:
            self._stats = {}
        else:
            self._stats = initial_stats  # Estatísticas Iniciais das classes
        self._mc_correct_weight = 0.0  # Acertos usando classe majoritário
        self._nb_correct_weight = 0.0  # Acertos usando Naive Bayes
        self.last_split_attempt_at = self.total_weight()  # Última tentativa de divisão
        self._attribute_observers = {}

    def get_stats(self):
        """ ... """
        return self._stats

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ ... """
        return FoundLeaf(self, parent, parent_branch)

    def total_weight(self):
        """ Calculate the total weight seen by the node. """
        return sum(self._stats.values())

    def learn_one(self, X, y, weight=1.0):
        """ Update the node with the provided instance. """
        if self._stats == {}:
            if y == 0:
                self._mc_correct_weight += weight
        elif max(self._stats, key=self._stats.get) == y:
            self._mc_correct_weight += weight
        if y not in self._stats.keys():
            self._stats[y] = 0
        self._stats[y] += weight
        self._stats = dict(sorted(self._stats.items()))

        nb_prediction = do_naive_bayes_prediction(
            X, self._stats, self._attribute_observers)
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight
        self.update_attribute_observers(X, y, weight)

    def update_attribute_observers(self, X, y, weight):
        """ ... """
        def create_observer():
            """ Futuro: verificar nominal"""
            return NumericAttributeObserverGaussian()

        for idx, x_value in enumerate(X):
            if idx not in self._attribute_observers.keys():
                self._attribute_observers[idx] = create_observer()
            self._attribute_observers[idx].update(x_value, y, weight)

    def predict_one(self, X):
        """ Get the votes per class for a given instance. """
        if self._mc_correct_weight > self._nb_correct_weight:
            return self._stats
        return do_naive_bayes_prediction(X, self._stats, self._attribute_observers)

    def get_best_split_suggestions(self):
        """ Find possible split candidates. """

        best_suggestions = []
        pre_split_dist = self._stats

        for i, obs in self._attribute_observers.items():
            suggestion = obs.get_best_evaluated_split_suggestion(
                pre_split_dist, i)
            if suggestion is not None:
                best_suggestions.append(suggestion)

        return best_suggestions
