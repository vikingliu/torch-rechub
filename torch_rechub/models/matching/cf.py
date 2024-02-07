# coding=utf-8
import math
from collections import defaultdict


class CollaborativeFiltering(object):
    def __init__(self, data, algo='i2i'):
        """
            传统item-cf背后的直觉是，如果大量用户同时喜欢两个物品，那么这两个物品之间应该有比较高的关联（相似度）。
            s(i, j) = |Ui ∩ Uj| / (sqrt(|Ui| * sqrt(|Uj|)))

            传统user-cf背后的直觉是，如果两个用户喜欢的物品重合度很高，那么这两个用户的兴趣爱好比较相似（相似度）。
            s(i, j) = |Ii ∩ Ij| / (sqrt(|Ii| * sqrt(|Ij|)))
             Args:
                 data: [item_id, user_id]

             """
        self.__statistics(data, algo)

    def __statistics(self, data, algo):
        self.idx = defaultdict(set)
        for item_id, user_id in data:
            if algo == 'i2i':
                self.idx[item_id].add(user_id)
            elif algo == 'u2u':
                self.idx[user_id].add(item_id)

        # 这里 list 可以定义成一个 小根堆，减少存储
        self.cf = defaultdict(list)
        ids = list(self.idx.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                similarity = self.__cal_similarity(ids[i], ids[j])
                self.cf[ids[i]].append((similarity, ids[j]))
                self.cf[ids[j]].append((similarity, ids[i]))
            self.cf[ids[i]].sort(key=lambda x: x[0], reverse=True)

    def __cal_similarity(self, id_i, id_j):
        if id_i not in self.idx or id_j not in self.idx:
            return 0
        Ui_and_Uj = len(self.idx[id_i] & self.idx[id_j])
        Ui = len(self.idx[id_i])
        Uj = len(self.idx[id_j])
        return Ui_and_Uj / math.sqrt(Ui * Uj)

    def get_i2i(self, id):
        if id not in self.cf:
            return []
        return self.cf[id]


data = [
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 2],
    [2, 3],
    [2, 7],
    [2, 8],
    [3, 2],
    [3, 3],
    [3, 1],
    [3, 9],
    [3, 10],
]

cf = CollaborativeFiltering(data)
print(cf.get_i2i(1))
print(cf.get_i2i(2))
print(cf.get_i2i(3))
