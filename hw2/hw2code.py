import numpy as np
from collections import Counter

#Определяем лучший порог для разбиения по критерию Джини среди всех возможных вариантов
def find_best_split(feature_vector, target_vector):
    order = np.argsort(feature_vector)
    x = feature_vector[order]
    y = target_vector[order]

    uniq_mask = np.diff(x) != 0
    if not np.any(uniq_mask):
        return np.array([]), np.array([]), None, None

    thresholds = (x[:-1][uniq_mask] + x[1:][uniq_mask]) / 2

    n = len(y)
    y0 = (y == 0).astype(int)
    y1 = (y == 1).astype(int)

    cum0 = np.cumsum(y0)[:-1][uniq_mask]
    cum1 = np.cumsum(y1)[:-1][uniq_mask]

    n_left = cum0 + cum1
    n_right = n - n_left

    mask = (n_left > 0) & (n_right > 0)
    thresholds = thresholds[mask]
    n_left = n_left[mask]
    n_right = n_right[mask]
    cum0 = cum0[mask]
    cum1 = cum1[mask]

    if thresholds.size == 0:
        return np.array([]), np.array([]), None, None

    p0_left = cum0 / n_left
    p1_left = cum1 / n_left

    total0 = np.sum(y0)
    total1 = np.sum(y1)
    p0_right = (total0 - cum0) / n_right
    p1_right = (total1 - cum1) / n_right

    H_left = 1 - p0_left**2 - p1_left**2
    H_right = 1 - p0_right**2 - p1_right**2

    ginis = -(n_left / n) * H_left - (n_right / n) * H_right

    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best




class DecisionTree:
    def __init__(self, feature_types,
                 max_depth=None, min_samples_split=None, min_samples_leaf=None):

        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            elif feature_type == "categorical":
                col = sub_X[:, feature]
                counts = Counter(col)
                clicks = Counter(col[sub_y == 1])
                ratio = {}
                for key, cnt in counts.items():
                    click = clicks.get(key, 0)
                    ratio[key] = click / (cnt + 1e-9)

                sorted_categories = list(
                    map(lambda x: x[0],
                        sorted(ratio.items(), key=lambda x: x[1]))
                )
                categories_map = dict(
                    zip(sorted_categories, range(len(sorted_categories)))
                )

                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], col))
                )
            else:
                raise ValueError

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            if thresholds.size == 0:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(
                        map(lambda x: x[0],
                            filter(lambda x: categories_map[x[0]] < threshold,
                                   categories_map.items()))
                    )
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_leaf is not None:
            n_left = np.sum(split)
            n_right = np.sum(~split)
            if n_left < self._min_samples_leaf or n_right < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            threshold = node["threshold"]
            if x[feature] < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            categories_left = node["categories_split"]
            if x[feature] in categories_left:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

