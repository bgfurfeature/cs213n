# coding=utf-8
"""
 smo 迭代优化svm 算法 实现
"""
import math
import random

class SMO(object):
    def __init__(self):
        self.alpha = []
        self.C = 0
        self.kn = 0
        self.sigma = 1

    def kernel(self, x1, x2):
        n = len(x2) - 1
        s = 0
        if self.kn == 0:
            for i in range(n):
                s += x1[i] * x2[i]
            return s
        for i in range(n):
            s += (x1[i] - x2[i]) ** 2
        k = math.exp( -s / 2 * self.sigma ** 2)
        return k

    def predict(self, d, data):
        m = len(data) # sample number
        y = 0
        for i in range(m):
            y += self.alpha[i] * data[i][-1] * self.kernel(d, data[i])
        y += b
        return y

    def predict_finish(self, d, data, ga):
        m = len(ga)  # number of no-zero params
        y = 0
        for i in range(m):
            j = ga[i]
            y += self.alpha[j] * data[j][-1] * self.kernel(d, data[j])
        y += b
        return y

    def is_same(self, v, s):
        if v == s:
            return 1
        else:
            return 0

    def select_first(self, data):
        m = len(data)
        for i in range(m):
            if 0 < self.alpha[i] < self.C:
                p = self.predict(data[i], data) * data[i][-1]
                if not self.is_same(p,1):
                    return i

        for i in range(m):
            if self.alpha[i] == 0:
                p = self.predict(data[i],data) * data[i][-1]
                if p < 1:
                    return i
            elif self.alpha[i] == self.C:
                p = self.predict(data[i],data) * data[i][-1]
                if p > 1:
                    return i
        return -1

    def select_second(self,i, m):
        j = i
        while j == i:
            j = random.randint(0, m-1)
        return j

    def update(self, i, j, data):
        low = 0
        high = self.C
        if data[i][-1] == data[j][-1]:
            low = max(0, self.alpha[i] + self.alpha[j] - self.C)
            high = min(self.C, self.alpha[i] + self.alpha[j])
        else:
             low = max(0, self.alpha[j] -  self.alpha[j])
             high = min(self.C, self.alpha[j] -  self.alpha[j] + self.C)

        if low == high:
            return False
        eta = self.kernel(data[i], data[i]) + self.kernel(data[j], data[j]) - 2 * self.kernel(data[i], data[j])

        if self.is_same(eta, 0):
            return False
        ei = self.predict(data[i], data) - data[i][-1]
        ej = self.predict(data[j],data) - data[j][-1]
        alpha_j = self.alpha[j] + data[j][-1] * (ei - ej ) / eta
        if alpha_j == self.alpha[j]:
            return False
        if alpha_j > high:
            alpha_j = high
        if alpha_j < low:
            alpha_j = low
        self.alpha[i] += (self.alpha[j] - alpha_j) * data[i][-1] * data[j][-1]
        self.alpha[j] = alpha_j
        return True

    def update_b(self,i, j, data):
        global  b
        bi = b + data[i][-1]  - self.predict(data[i], data)
        bj = b + data[j][-1]  - self.predict(data[j], data)
        if self.C > self.alpha[i] > 0:
            return  bi
        elif self.C > self.alpha[j] > 0:
            return bj
        return (bi + bj) / 2

    def smo(self, data):
        m = len(data)
        global b
        for time in range(5000):
            no_change = 0
            i = self.select_first(data)
            if i == -1:
                break
            j = self.select_second(i, m)
            if not self.update(i, j, data):
                no_change += 1
                continue
            b = self.update_b(i,j,data)
            print time,b
            if no_change > 100:
                break

