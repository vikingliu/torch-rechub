# coding=utf-8
"""
阿里swing论文，Large Scale Product Graph Construction for Recommendation in E-commerce
https://arxiv.org/pdf/2010.05525.pdf
"""
from collections import defaultdict


class Swing(object):
    def __init__(self, data, alpha=1):
        """
        swing的直觉来源是，如果大量用户同时喜欢两个物品，且这些用户之间的相关性低，那么这两个物品一定是强关联。
        Args:
            data: [(item_id, user_id)]

        """
        self.__statistics(data, alpha)

    def __statistics(self, data, alpha):
        self.item_dic = defaultdict(set)
        self.user_dic = defaultdict(set)
        for item_id, user_id in data:
            self.item_dic[item_id].add(user_id)
            self.user_dic[user_id].add(item_id)

        # 这里 list 可以定义成一个 小根堆，减少存储
        self.item_i2i = defaultdict(list)
        item_ids = list(self.item_dic.keys())
        for i in range(len(item_ids)):
            for j in range(i + 1, len(item_ids)):
                similarity = self.__cal_similarity(item_ids[i], item_ids[j], alpha)
                self.item_i2i[item_ids[i]].append((similarity, item_ids[j]))
                self.item_i2i[item_ids[j]].append((similarity, item_ids[i]))
            self.item_i2i[item_ids[i]].sort(key=lambda x: x[0], reverse=True)

    def __cal_similarity(self, item_i, item_j, alpha):
        # Ui表示喜欢物品i的用户集合
        Ui = self.item_dic[item_i]
        # Uj表示喜欢物品j的用户集合
        Uj = self.item_dic[item_j]
        U = Ui & Uj
        similarity = 0
        for u in U:
            # Iu表示用户u喜欢的item集合
            lu = self.user_dic[u]
            for v in U:
                lv = self.user_dic[v]
                similarity += 1 / (alpha + len(lu & lv))

        return similarity

    def get_i2i(self, item_id):
        if item_id not in self.item_i2i:
            return []
        return self.item_i2i[item_id]


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

i2i = Swing(data)
print(i2i.get_i2i(1))
print(i2i.get_i2i(2))
print(i2i.get_i2i(3))