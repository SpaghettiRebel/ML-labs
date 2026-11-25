import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    return _split_scores(feature_vector, target_vector, 1)


def _split_scores(feature_vector, target_vector, min_leaf):
    order = np.argsort(feature_vector)
    values = np.asarray(feature_vector, dtype=float)[order]
    left = np.flatnonzero(values[:-1] < values[1:]) + 1
    n = len(values)
    left = left[(left >= min_leaf) & (n - left >= min_leaf)]
    if left.size == 0:
        return np.array([]), np.array([]), None, None

    positives = np.cumsum(np.asarray(target_vector)[order])
    pos_left = positives[left - 1]
    pos_right = positives[-1] - pos_left
    right = n - left
    thresholds = (values[left - 1] + values[left]) / 2
    ginis = -2 / n * (pos_left * (left - pos_left) / left
                      + pos_right * (right - pos_right) / right)
    best = np.argmax(ginis)
    return thresholds, ginis, thresholds[best], ginis[best]


class DecisionTree(ClassifierMixin, BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        if len(self.feature_types) != self.n_features_in_:
            raise ValueError("Specify one type per feature")
        if any(kind not in ("real", "categorical") for kind in self.feature_types):
            raise ValueError("Unknown feature type")
        self.classes_, labels = np.unique(y, return_inverse=True)
        if len(self.classes_) > 2:
            raise ValueError("DecisionTree supports binary classification")

        def sample_limit(value, default):
            if value is None:
                return default
            if isinstance(value, (float, np.floating)):
                return int(np.ceil(value * len(X)))
            return value

        self._min_leaf = sample_limit(self.min_samples_leaf, 1)
        self._min_split = max(2, sample_limit(self.min_samples_split, 2))
        self.depth_ = 0
        self._tree = {}
        self._fit_node(X, labels, self._tree, 0)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth):
        self.depth_ = max(self.depth_, depth)
        counts = np.bincount(sub_y)
        node["type"] = "terminal"
        node["class"] = self.classes_[np.argmax(counts)]
        if (counts.max() == len(sub_y) or depth == self.max_depth
                or len(sub_y) < max(self._min_split, 2 * self._min_leaf)):
            return

        best_score = -np.inf
        for feature, kind in enumerate(self.feature_types):
            column = sub_X[:, feature]
            if kind == "categorical":
                categories, indices, sizes = np.unique(column, return_inverse=True, return_counts=True)
                rates = np.bincount(indices, weights=sub_y) / sizes
                order = np.argsort(rates, kind="stable")
                values = np.argsort(order)[indices]
            else:
                values = column.astype(float, copy=False)

            _, _, threshold, score = _split_scores(values, sub_y, self._min_leaf)
            if threshold is None or score <= best_score:
                continue
            best_score = score
            mask = values < threshold
            if kind == "real":
                rule = {"threshold": threshold}
            else:
                rule = {"categories_split": set(categories[order[:int(np.ceil(threshold))]])}
            best_feature = feature

        if best_score > -np.inf:
            node.clear()
            node.update(type="nonterminal", feature_split=best_feature, **rule,
                        left_child={}, right_child={})
            self._fit_node(sub_X[mask], sub_y[mask], node["left_child"], depth + 1)
            self._fit_node(sub_X[~mask], sub_y[~mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        while node["type"] != "terminal":
            value = x[node["feature_split"]]
            if "threshold" in node:
                go_left = float(value) < node["threshold"]
            else:
                go_left = value in node["categories_split"]
            node = node["left_child"] if go_left else node["right_child"]
        return node["class"]

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in np.asarray(X)])
