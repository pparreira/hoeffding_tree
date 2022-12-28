import math
from .statistics import normal_probability


class GaussianEstimator(object):
    """ Estimador Gaussiano usado como distribuições dos atributos númericos"""

    def __init__(self):
        self._weight_sum = 0.0  # Soma dos pesos
        self._mean = 0.0  # Média
        self._variance_sum = 0.0  # Soma da variância
        self._NORMAL_CONSTANT = math.sqrt(2 * math.pi)

    def add_observation(self, value, weight):
        """ Atualiza os parâmetros com  a observação"""
        if value is None or math.isinf(value):
            raise Exception("Estimador Gaussiano - Valor nulo ou infnito")
        if self._weight_sum > 0.0:
            self._weight_sum += weight
            last_mean = self._mean
            self._mean += weight * (value - last_mean) / self._weight_sum
            self._variance_sum += weight * \
                (value - last_mean) * (value - self._mean)
        else:
            self._mean = value
            self._weight_sum = weight

    def get_total_weight_observed(self):
        """ ... """
        return self._weight_sum

    def get_mean(self):
        """ ... """
        return self._mean

    def get_std_dev(self):
        """ ... """
        return math.sqrt(self.get_variance())

    def get_variance(self):
        """ ... """
        return self._variance_sum / (self._weight_sum - 1.0) if self._weight_sum > 1.0 else 0.0

    def probability_density(self, value):
        """ função densidade de probabilidade (FDP): Retorna P(X = x) """

        if self._weight_sum > 0.0:
            std_dev = self.get_std_dev()
            mean = self.get_mean()
            if std_dev > 0.0:
                diff = value - mean
                return ((1.0 / (self._NORMAL_CONSTANT * std_dev))
                        * math.exp(-(diff * diff / (2.0 * std_dev * std_dev))))
            if value == mean:
                return 1.0
        return 0.0

    def estimated_weights_in_split(self, value):
        """ Retorna [P(X < x), P(X=x), P(X>x)] * Weight"""

        equal_weight = self.probability_density(value) * self._weight_sum
        std_dev = self.get_std_dev()
        mean = self.get_mean()
        if std_dev > 0.0:
            # Função normal returna P(-inf < X < z), onde z = (x-mi)/std
            less_weight = normal_probability(
                (value - mean) / std_dev) * self._weight_sum - equal_weight
        else:
            if value < mean:
                less_weight = self._weight_sum - equal_weight
            else:
                less_weight = 0.0

        greater_weight = self._weight_sum - equal_weight - less_weight
        if greater_weight < 0.0:
            greater_weight = 0.0

        return [less_weight, equal_weight, greater_weight]
