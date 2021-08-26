# basic neural net that subtracts
import neuralnet as nn
import numpy as np
import utils
import tensorflow as tf
import random
import analysis
import matplotlib.pyplot as plt

symbols = ['-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '10', '11', '12', '13', '14', '15', '16','17', '18', '19',
           '+', '-', '=',
           "init", "tea", "coffee", "water", "stir", "sugar", "cream", "serve_tea", "serve_coffee"]

beverage_seqs = [
    ["init", "tea", "water", "stir", "sugar", "stir", "serve_tea"],
    ["init", "tea", "sugar", "stir", "water", "stir", "serve_tea"],
    ["init", "coffee", "water", "stir", "cream", "stir", "serve_coffee"],
    ["init", "coffee", "cream", "stir", "water", "stir", "serve_coffee"],
]

arithmetic_seqs = []
plusplusseqs = [
[3,4,5,12],[4,3,6,13],[5,4,3,12],[6,3,4,13],
[7,4,2,13],[8,5,4,17],[9,6,3,18],[7,4,3,14],
[3,5,7,15],[4,5,8,17],[5,3,4,12],[6,5,3,14],
[7,5,6,18],[8,4,7,19],[9,2,5,16],[6,8,3,17],
[6,5,7,18]]
for seq in plusplusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '+')
    seq.insert(3, '+')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
plusminusseqs = [
    [6, 9, 8, 7], [4,2,9,-3], [5,3,9,-1],
    [6,5,7,4], [4,7,3,8], [7,5,6,6],
    [8,4,5,7], [5,2,8,-1], [7,5,6,6],
    [4,3,9,-2], [5,2,8,-1], [6,3,4,5],
    [4,7,3,8], [2,3,8,-3], [5,6,3,8],
    [2,3,8,-3], [9,4,7,6],
    ]
for seq in plusminusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '+')
    seq.insert(3, '-')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
minusplusseqs = [
    [9,2,5,12],    [6,7,8,7],    [3,8,4,-1],
    [4,3,7,8],    [8,4,5,9],    [7,2,4,9],
    [8,3,6,11],    [9,4,3,8],    [2,5,7,4],
    [3,9,4,-2],    [4,8,3,-1],    [5,3,6,8],
    [2,9,4,-3],    [3,8,2,-3],    [4,9,3,-2],
    [5,7,9,7],    [9,5,3,7]
]
for seq in minusplusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '-')
    seq.insert(3, '+')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)
minusminusseqs = [
    [9,6,2,1],    [4,9,4,-9],    [6,8,3,-5],    [8,5,4,-1],
    [7,4,5,-2],    [3,4,6,-7],    [4,5,6,-7],    [5,8,3,-6],
    [5,2,8,-5],    [4,3,5,-4],    [7,3,5,-1],    [2,6,3,-7],
    [6,8,5,-7],    [8,2,5,1],    [7,3,6,-2],    [7,3,8,-4],
    [9,5,2,2]
    ]
for seq in minusminusseqs:
    for idx, elem in enumerate(seq):
        seq[idx]=str(elem)
    seq.insert(1, '-')
    seq.insert(3, '-')
    seq.insert(5, '=')
    arithmetic_seqs.append(seq)

for seq in arithmetic_seqs:
    print(seq)

label_seqs_ari = [['num1', '+', 'num2', '+', 'num3', '='],
                    ['num1', '+', 'num2', '-', 'num3', '='],
                    ['num1', '-', 'num2', '+', 'num3', '='],
                    ['num1', '-', 'num2', '-', 'num3', '=']]

label_seqs_bev = []
for seq in beverage_seqs:
    label_seqs_bev.append(seq[1:])

label_seqs_all = []
for b in beverage_seqs:
    for a in label_seqs_ari:
        label_seqs_all.append([b[1], a[0], b[2], a[1], b[3], a[2], b[4], a[3], b[5], a[4], b[6], a[5]])



class Target(object):
    def __init__(self, action):
        self.action_one_hot = action
        self.goal1_one_hot = None
        self.goal2_one_hot = None
