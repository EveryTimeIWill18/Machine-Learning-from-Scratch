"""
classification_trees.py
~~~~~~~~~~~~~~~~~~~~~~~
"""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Set, Tuple

inputs = [
 {'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no', 'did_well': False},
 {'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes', 'did_well': False},
 {'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no', 'did_well': True},
 {'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no', 'did_well': True},
 {'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no', 'did_well': True},
 {'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes', 'did_well': False},
 {'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes', 'did_well': True},
 {'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no', 'did_well': False},
 {'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no', 'did_well': True},
 {'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no', 'did_well': True},
 {'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes', 'did_well': True},
 {'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes', 'did_well': True},
 {'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no', 'did_well': True},
 {'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes', 'did_well': False}
]

features = ['level', 'lang', 'tweets', 'phd', 'did_well']


class Node:
    def __init__(self, feature: str,  children=None, value: bool=None, node_partition=None):
        self.feature = feature
        self.children: Dict[str, Any] = children
        self.node_partition = node_partition
        self.value = value

    def _add_children(self, children: List[Any]) -> None:
        if self.children is None:
            self.children = {}
        if len(children) > 0:
            for node in children:
                self.children[node.feature] = node

    def is_leaf(self, counts) -> bool:
        if len(counts) == 1:
            self.value = list(counts.keys())[0]
            return True
        else:
            return False



class ClassificationTree:
    def __init__(self, dataset: List[Dict[str, Any]], target: str, features: List[str]):
        self.dataset = dataset
        self.target = target
        self.target_entropy: float or None = None
        self.features = [f for f in features if f != self.target]
        self.root = None

    def get_unique_feature_values(self, data, feature_name: str) -> Set[str]:
        unique_sub_features = set([d[feature_name] for d in data])
        return unique_sub_features

    def feature_probability(self, feature_data: Dict[Any, int]):
        data_len = sum([value for value in feature_data.values()])

        probability = {k: val/data_len for k, val in feature_data.items()}
        return probability

    def target_probability(self, target_counts):
        data_len = sum([val for val in target_counts.values()])

        probability = {k: val/data_len for k, val in target_counts.items()}
        return probability

    def weighted_entropy(self, feature_counts, feature_entropies):
        data_len = sum([val for val in feature_counts.values()])

        weighted_entropy = sum([(feature_counts[k]/data_len)*feature_entropies[k]
                                for k in feature_counts.keys()])
        return weighted_entropy

    def information_gain(self, weighted_entropy):
        if self.target_entropy is not None:
            return self.target_entropy - weighted_entropy

    def feature_entropy(self, feature_values, is_target=False):
        feature_entropies = {}
        if is_target:
            target_probability = self.target_probability(target_counts=feature_values)
            entropy = -sum([val*np.log2(val) for val  in target_probability.values()])
            self.target_entropy = entropy

        else:
            for feature in feature_values.keys():
                p = self.feature_probability(feature_data=feature_values[feature])
                entropy = -sum([pr*np.log2(pr) for pr in p.values()])
                if entropy == -0 or entropy == -0.0:
                    entropy = 0.0
                feature_entropies[feature] = entropy
            return feature_entropies



    def partition_by(self, data: List[Dict[str, Any]], feature: str) -> Tuple[Dict[str,  Any], Dict[str, Any], Dict[Any, Any]]:
        partition = defaultdict(list)
        counts = defaultdict(int)
        feature_target_values = defaultdict(dict)

        for row in data:
            k = row[feature]
            target_value = row[self.target]
            partition[k].append(row)
            counts[k] += 1
            feature_target_values[k][target_value] = feature_target_values[k].get(target_value, 0) + 1

        return partition, counts, feature_target_values

    def select_best_feature(self, information_gain: Dict[str, float]) -> Tuple[str, float]:
        max_value = max(information_gain.values())
        max_feature = [f for f in information_gain.keys() if information_gain[f] == max_value][0]

        return max_feature, max_value

    def construct_tree(self, current_partition, features: List[str], feature_name: str=None, current_node: Node=None):
        current_partitions = {}
        current_counts = {}
        current_tf = {}
        information_gains = {}

        if len(features) == 0:
            print(f'TREE COMPLETED')
            return current_partitions
        elif self.root is None:
            target_partition, target_counts, target_tf = self.partition_by(data=current_partition, feature=self.target)
            current_partitions[self.target] = target_partition
            current_counts[self.target] = target_counts
            self.feature_entropy(feature_values=target_counts, is_target=True)

            for feature in features:
                partition, counts, tf_counts = self.partition_by(data=current_partition, feature=feature)
                current_partitions[feature] = partition
                current_tf[feature] = tf_counts
                entropy = self.feature_entropy(feature_values=tf_counts, is_target=False)
                weighted_entropy = self.weighted_entropy(feature_counts=counts, feature_entropies=entropy)
                info_gain = self.information_gain(weighted_entropy=weighted_entropy)
                information_gains[feature] = info_gain
            # Create a new node
            best_feature, gain = self.select_best_feature(information_gain=information_gains)
            next_partition = current_partitions[best_feature]
            updated_features = [f for f in features if f != best_feature]

            self.root = Node(feature=best_feature)
            nodes = []
            for sub in next_partition.keys():
                node = Node(feature=sub, node_partition=next_partition[sub])
                nodes.append(node)
                is_leaf = node.is_leaf(counts=current_tf[best_feature][sub])
                if is_leaf:
                    print(f'Node: {best_feature}:{sub} is leaf. Value := {node.value=}')

            self.root._add_children(nodes)
            for child in self.root.children.keys():
                if self.root.children[child].value is None:
                    self.construct_tree(current_partition=self.root.children[child].node_partition,
                                        features=updated_features,
                                        feature_name=child,
                                        current_node=self.root.children[child])
        else:
            for feature in features:
                partition, counts, tf_counts = self.partition_by(data=current_partition, feature=feature)
                current_partitions[feature] = partition
                current_tf[feature] = tf_counts
                entropy = self.feature_entropy(feature_values=tf_counts, is_target=False)
                weighted_entropy = self.weighted_entropy(feature_counts=counts, feature_entropies=entropy)
                info_gain = self.information_gain(weighted_entropy=weighted_entropy)
                information_gains[feature] = info_gain
            # Create a new node
            best_feature, gain = self.select_best_feature(information_gain=information_gains)
            next_partition = current_partitions[best_feature]
            updated_features = [f for f in features if f != best_feature]

            nodes = []
            for sub in next_partition.keys():
                node = Node(feature=sub, node_partition=next_partition[sub])
                nodes.append(node)
                is_leaf = node.is_leaf(counts=current_tf[best_feature][sub])
                if is_leaf:
                    print(f'Node: {best_feature}:{sub} is leaf. Value := {node.value=}')

            current_node._add_children(nodes)
            for child in current_node.children.keys():
                if current_node.children[child].value is None:
                    self.construct_tree(current_partition=current_node.children[child].node_partition,
                                            features=updated_features,
                                            feature_name=child,
                                            current_node=current_node.children[child])
