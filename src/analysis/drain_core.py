"""Drain 论文机制的标准库复现 v1。

来源 https://pinjiahe.github.io/files/pdf/research/ICWS17.pdf
机制对照 https://github.com/logpai/logparser/blob/main/logparser/Drain/Drain.py
本模块独立实现树路由、token 相似度及模板泛化，不是官方包。
深度定义沿用 LogPAI 实现，depth=4 时按长度及首 token 路由。
"""

from dataclasses import dataclass, field


WILDCARD = "<*>"


@dataclass
class Cluster:
    cluster_id: int
    tokens: list[str]
    count: int = 1


@dataclass
class Node:
    children: dict = field(default_factory=dict)
    cluster_ids: list[int] = field(default_factory=list)


class Drain:
    def __init__(self, depth=4, similarity=0.4, max_children=100):
        if depth < 3 or not 0 <= similarity <= 1 or max_children < 2:
            raise ValueError("Drain 参数无效")
        self.depth = depth
        self.similarity = similarity
        self.max_children = max_children
        self.root = Node()
        self.clusters = {}

    def route(self, tokens, create):
        length = len(tokens)
        if length not in self.root.children:
            if not create:
                return None
            self.root.children[length] = Node()
        node = self.root.children[length]
        for token in tokens[:self.depth - 3]:
            if token in node.children:
                key = token
            elif not create:
                key = WILDCARD
                if key not in node.children:
                    return None
            else:
                numeric = any(character.isdigit() for character in token)
                reserve = WILDCARD not in node.children
                full = len(node.children) >= self.max_children - int(reserve)
                key = WILDCARD if numeric or full else token
                node.children.setdefault(key, Node())
            node = node.children[key]
        return node

    def add(self, message):
        tokens = message.split()
        node = self.route(tokens, create=False)
        candidates = [] if node is None else node.cluster_ids
        best_id = None
        best_key = (-1., -1)
        for cluster_id in candidates:
            template = self.clusters[cluster_id].tokens
            similarity = (
                sum(a == b and a != WILDCARD for a, b in zip(template, tokens)) / len(tokens)
                if tokens else 1.
            )
            key = (similarity, template.count(WILDCARD))
            if key > best_key:
                best_key = key
                best_id = cluster_id
        if best_id is not None and best_key[0] >= self.similarity:
            cluster = self.clusters[best_id]
            cluster.tokens = [a if a == b else WILDCARD for a, b in zip(cluster.tokens, tokens)]
            cluster.count += 1
            return best_id
        cluster_id = len(self.clusters) + 1
        self.clusters[cluster_id] = Cluster(cluster_id, tokens)
        self.route(tokens, create=True).cluster_ids.append(cluster_id)
        return cluster_id

    def templates(self):
        return {key: " ".join(cluster.tokens) for key, cluster in self.clusters.items()}
