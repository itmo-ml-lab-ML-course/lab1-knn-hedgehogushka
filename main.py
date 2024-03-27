import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# constants
kategorial_null_changer = "Неопределен"
number_null_changer = 0
price = "Цена"
category_types = ["Производитель",
                  "Тип механизма секретности",
                  "Тип двери"]
number_types = ["Цена",
                "Вес",
                "Межосевое расстояние",
                "Бэксет (удаление ключевого отверстия)",
                "Вылет ригеля"]
# data parsing
full_data = pd.read_csv("data.csv")
full_data = full_data.drop(columns=["Название товара", "Оптовая цена"])
full_data[category_types[2]] = ["Деревянная"] * 817 + ["Металлическая"] * 352

for type in category_types:
    full_data[type] = full_data[type].fillna(kategorial_null_changer)

for type in number_types:
    full_data[type] = full_data[type].fillna(number_null_changer)
    if (type == "Вес"):
        list_data = full_data[type].to_list()
        for i in range(len(list_data)):
            if list_data[i] >= 100:
                list_data[i] = list_data[i] / 1000
        full_data[type] = list_data
    sorted_data = full_data.sort_values(by=[type])
    plt.title(type)
    plt.plot(sorted_data[type], sorted_data[price])
    plt.show()

type_1 = "Вес"
for type_2 in number_types:
    if type_1 != type_2:
        sorted_data = full_data.sort_values(by=[type_1])
        sorted_data[price] = tuple(map(lambda x: int(x),
                                       sorted_data[price]))
        plt.scatter(sorted_data[type_1],
                    sorted_data[type_2],
                    c=sorted_data[price])
        plt.title(type_1 + " / " + type_2)
        plt.show()

full_data["Доп. параметр"] = full_data[type_1]/full_data[price]
scaler = StandardScaler()
number_types.append("Доп. параметр")
full_data[number_types] = scaler.fit_transform(full_data[number_types])

onehotencoder = OneHotEncoder(sparse_output=False)
data_new = onehotencoder.fit_transform(full_data[category_types])
categories = onehotencoder.categories_
full_data = full_data.drop(columns=category_types)
full_data = pd.concat([full_data,
                       pd.DataFrame(data_new,
                                    columns=np.concatenate(
                                        [*categories]))], axis=1)


X = full_data.drop(columns=["Цена"])
Y = full_data["Цена"]
coefs = [0] * 50
for i in range(50):
    coefs[i] = i + 1

x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,train_size=0.8)

error_train = [0] * len(coefs)
error_val = [0] * len(coefs)

it = 0
for coef in coefs:
    knn = KNeighborsRegressor(n_neighbors=coef)
    knn.fit(x_train, y_train)
    error_train[it] = mean_squared_error(knn.predict(x_train), y_train)
    error_val[it] = mean_squared_error(knn.predict(x_val), y_val)
    it += 1

plt.plot(coefs, error_train, label='train')
plt.plot(coefs, error_val, label='valid')
plt.legend()
plt.show()



class KNN:
    def __init__(self, K: int = 3, window_size: str = 'fixed') -> None:
        self.K = K
        self.window_size = window_size  # 'fixed' or 'variable'
        self.X_train = np.array([])
        self.y_train = np.array([])

    # START DISTANCE
    def minkowski(self, x: np.array, p: int) -> np.array:
        return np.power(np.sum(np.power(np.abs(self.X_train - x), p), axis=1), 1 / p)

    def cosine_similarity(self, x: np.array) -> np.array:
        dot = np.sum(self.X_train * x, axis=1)
        return 1 - dot / (np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x))

    def manhattan(self, x: np.array) -> np.array:
        return np.sum(np.abs(self.X_train - x), axis=1)

    def distances(self, X: np.array, metric: str) -> np.array:
        if metric == 'minkowski':
            return np.apply_along_axis(lambda x: self.minkowski(x, p=2), 1, X)
        elif metric == 'cosine':
            return np.apply_along_axis(self.cosine_similarity, 1, X)
        elif metric == 'manhattan':
            return np.apply_along_axis(self.manhattan, 1, X)

    # END DISTANCE
    # START KERNEL

    def uniform_kernel(self, x: np.array) -> np.array:
        return np.ones(len(x))

    def gaussian_kernel(self, x: np.array) -> np.array:
        return np.exp(-0.5 * (x ** 2))

    def triangular_kernel(self, x: np.array) -> np.array:
        return np.maximum(0, 1 - np.abs(x))

    def epanechnikov_kernel(self, x: np.array) -> np.array:
        return np.maximum(0, 3 / 4 * (1 - x ** 2))

    # END KERNEL

    def fit(self, X: np.array, y: np.array) -> None:
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)

    def generate_predictions(self, idx_neighbours: np.array, kernel_func, weights=None) -> np.array:
        if weights is None:
            weights = kernel_func(idx_neighbours[:, -1])
        y_pred = np.average(self.y_train[idx_neighbours], weights=weights, axis=1)
        return y_pred

    def predict(self, X: np.array, metric: str, kernel: str, weights: np.array = None,
                window_size: int = -1) -> np.array:
        if self.window_size == 'fixed' or window_size < 0:
            idx_neighbours = self.distances(X, metric).argsort()[:, :self.K]
        else:
            idx_neighbours = self.distances(X, metric).argsort()[:, :window_size]

        kernel_func = self.uniform_kernel
        if kernel == 'gaussian':
            kernel_func = self.gaussian_kernel
        elif kernel == 'triangular':
            kernel_func = self.triangular_kernel
        elif kernel == 'epanechnikov':
            kernel_func = self.epanechnikov_kernel

        y_pred = self.generate_predictions(idx_neighbours, kernel_func, weights)
        return y_pred