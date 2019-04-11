import pandas as pd
import numpy as np


class MultivariateClassifier:
    ## Class Variables ##
    X = C_n = mean = cov = prior = None

    ## Constructors ##
    def __init__(self):
        self.C_n = {}
        self.mean = {}
        self.cov = {}
        self.prior = {}

    ## Methods ##
    def parse_data(self, file_path, label_col, sep=','):
        df = pd.read_csv(file_path, sep=sep, header=None)
        self.X = {
            "label": np.array(df.pop(label_col))
            , "data": np.array(df)
        }

        self._init_cn()
        self._load_prior()
        self._load_mean()
        self._load_cov()

    def _init_cn(self):
        temp = {}
        C_i = 0

        for i, row in enumerate(self.X["data"]):
            if self.X["label"][i] not in temp:
                temp[self.X["label"][i]] = C_i
                self.C_n[C_i] = []
                C_i += 1

            self.C_n[temp[self.X["label"][i]]].append(row)

        for i in range(len(self.C_n.keys())):
            self.C_n[i] = np.array(self.C_n[i])

    def _load_prior(self):
        for n in self.C_n.keys():
            self.prior[n] = len(self.C_n[n]) / len(self.X["label"])

    def _load_mean(self):
        self.mean["all"] = np.array([np.mean(x_i) for x_i in self.X["data"].T])

        for n in self.C_n.keys():
            self.mean[n] = np.array([np.mean(x_i) for x_i in self.C_n[n].T])

    def _load_cov(self):
        self.cov["all"] = np.cov(self.X["data"].T)

        for n in self.C_n.keys():
            self.cov[n] = np.cov(self.C_n[n].T)

    def _likelihood(self, x_i, m_n, cov_n):
        exponent = lambda x, m, cov: -0.5 * (x - m).dot(np.linalg.inv(cov)).dot((x - m).T)
        return (1 / ((2 * np.pi) ** (len(self.C_n.keys()) / 2) * np.linalg.det(cov_n) ** 0.5)) \
               * np.exp(exponent(x_i, m_n, cov_n))

    def _evidence(self, x_i):
        exponent = lambda x: -0.5 * (x - self.mean["all"]).dot(np.linalg.inv(self.cov["all"])).dot((x - self.mean["all"]).T)
        return (1 / ((2 * np.pi) ** (len(self.C_n.keys()) / 2) * np.linalg.det(self.cov["all"]) ** 0.5)) \
               * np.exp(exponent(x_i))

    def classify(self, test_x):
        hmap = {}
        for n in self.C_n.keys():
            post = self._likelihood(np.array(test_x), self.mean[n], self.cov[n]) * self.prior[n] \
                   / self._evidence(test_x)

            if post not in hmap:
                hmap[post] = []

            hmap[post].append(n)

        class_list = hmap[np.max(list(hmap))]
        return class_list[np.random.randint(0, len(class_list))]
