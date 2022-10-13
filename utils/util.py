# Copyright 2022 Intel Corporation.
# SPDX-License-Identifier: MIT


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np

def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)

def predictive_entropy(mc_preds):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return entropy(np.mean(mc_preds, axis=0))

def mutual_information(mc_preds):
    """
    Compute the difference between the entropy of the mean of the
    predictive distribution and the mean of the entropy.
    """
    MI = entropy(np.mean(mc_preds, axis=0)) - np.mean(entropy(mc_preds),
                                                      axis=0)
    return MI

def sim_scores(features_train, features):
    scores = []
    indices = []
    features_train = torch.from_numpy(features_train)
    features = torch.from_numpy(features)
    features_train_norm = features_train / (features_train.norm(dim=1, keepdim=True))
    for feature in features:
        # features_norm = feature / (feature.norm(dim=0, keepdim=True))
        features_norm = feature
        sim_score, index = torch.max((features_norm * features_train_norm).sum(dim=1), -1)
        scores.append(sim_score)
        indices.append(index)
    scores_all = torch.FloatTensor(scores)
    indices_all = torch.IntTensor(indices)
    return scores_all, indices_all

def sim_scores_cos(features_train, features):
    scores = []
    indices = []
    features_train = torch.from_numpy(features_train)
    features = torch.from_numpy(features)
    features_train_norm = features_train / (features_train.norm(dim=1, keepdim=True))
    for feature in features:
        a = feature
        b = features_train_norm
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=0).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (2 * a_norm * b_norm)
        sim_score, index = torch.min(cos, dim=-1) 
        scores.append(sim_score)
        indices.append(index)
    scores_all = torch.FloatTensor(scores)
    indices_all = torch.IntTensor(indices)
    return scores_all, indices_all

def sim_scores_new(train_features, no_fault_features, fault_features, predictions, fault_predictions, gnd_label):
    labels = list(train_features.keys())
    NUM_CLASSES = np.unique(labels)
    # scores_nofault = np.array(fault_predictions.shape)
    # no_fault_features = np.tile(no_fault_features, int(fault_features.shape[0]/no_fault_features.shape[0]))
    scores_nofault = np.empty(np.array(predictions).shape)
    scores_fault   = np.empty(np.array(fault_predictions).shape)

    for label in NUM_CLASSES:
        # sim_score_nofault = np.array(sim_scores(train_features[str(label)], no_fault_features[np.where(np.array(gnd_label) == int(label))[0]])[0], dtype=float)
        # sim_score_fault = np.array(sim_scores(train_features[str(label)], fault_features[np.where(np.array(gnd_label) == int(label))[0]])[0], dtype=float)
        sim_score_nofault = np.array(sim_scores(train_features[str(label)], no_fault_features[np.where(np.array(gnd_label) == int(label))[0]])[0])
        sim_score_fault = np.array(sim_scores(train_features[str(label)], fault_features[np.where(np.array(gnd_label) == int(label))[0]])[0])

        # scores_nofault[str(label)] = sim_score_nofault
        # scores_fault[str(label)]   = sim_score_fault
        scores_nofault[np.where(np.array(gnd_label) == int(label))[0]] = sim_score_nofault
        scores_fault[np.where(np.array(gnd_label) == int(label))[0]] = sim_score_fault
    # scores_nofault = np.tile(scores_nofault, int(scores_fault.shape[0]/scores_nofault.shape[0]))
    return scores_nofault, scores_fault

def sim_scores_new_cos(train_features, no_fault_features, fault_features, predictions, fault_predictions, gnd_label):
    labels = list(train_features.keys())
    NUM_CLASSES = np.unique(labels)
    # scores_nofault = np.array(fault_predictions.shape)
    scores_nofault = np.empty(np.array(predictions).shape)
    scores_fault   = np.empty(np.array(fault_predictions).shape)

    for label in NUM_CLASSES:
        sim_score_nofault = np.array(sim_scores_cos(train_features[str(label)], no_fault_features[np.where(np.array(gnd_label) == int(label))[0]])[0], dtype=float)
        sim_score_fault = np.array(sim_scores_cos(train_features[str(label)], fault_features[np.where(np.array(gnd_label) == int(label))[0]])[0], dtype=float)

        # scores_nofault[str(label)] = sim_score_nofault
        # scores_fault[str(label)]   = sim_score_fault
        scores_nofault[np.where(np.array(gnd_label) == int(label))[0]] = sim_score_nofault
        scores_fault[np.where(np.array(gnd_label) == int(label))[0]] = sim_score_fault
    scores_nofault = np.tile(scores_nofault, int(scores_fault.shape[0]/scores_nofault.shape[0]))
    return scores_nofault, scores_fault
