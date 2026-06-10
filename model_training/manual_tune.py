# bow

# # Part 1: Environment Setup

import os
os.getcwd()

# import general_module which locate at my parent directory's child
import sys
from pathlib import Path
import copy

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from general_module.evaluation import *
from general_module.training import *
from model_training.deployment import (
    CustomNetwork,
    load_vectorizer,
    save_bundle,
    verify_vectorizer,
)
from model_training.loss import *

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle


torch.manual_seed(42)

from transformers import logging
logging.set_verbosity_error()


# # Part 2: Load dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, dimension, feature):
        self.dataframe = dataframe
        self.dimension = dimension
        self.feature = feature

    def __getitem__(self, index):
        if self.feature == "bow":
            fea = torch.tensor(self.dataframe['text'].values[index])
        elif self.feature == "psycho":
            fea = torch.tensor(self.dataframe['psychofeature'].values[index])
        elif self.feature == "bow+psycho":
            #merge two features
            fea = torch.tensor(np.concatenate((self.dataframe['text'].values[0], self.dataframe['psychofeature'].values[0])))

        fea = fea.type(torch.FloatTensor)

        label = torch.tensor(float(str(self.dataframe[self.dimension].values[index])))
        label = label.type(torch.FloatTensor)

        return fea, label

    def __len__(self):
        return len(self.dataframe)


def train(loss_function, epochs, trainloader, testloader):
    # get the input_size from trainloader
    in_size= trainloader.dataset[0][0].shape[0]

    network = CustomNetwork(input_size=in_size)
    network = best_device(network)

    model = CustomModel(network)
        
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    best_state = copy.deepcopy(network.state_dict())

    for e in range(epochs):
        # a dictionary that store the training loss, test loss, train_size, test_size, TP, FP, TN, FN
        running_info = {'train_loss':0, 'test_loss':0, 'train_size':0, 'test_size':0, 'TP':0, 'FP':0, 'TN':0, 'FN':0}

        # set to training mode
        network.train(True)

        # per epoch training activity
        for inputs, labels in trainloader:

            # clear all the gradient to 0
            optimizer.zero_grad()

            inputs,labels = best_device(inputs, labels)

            # forward propagation
            outs = network(inputs)
            outs = outs.view(-1)
            
            # compute loss
            loss = loss_function.forward(inputs=outs, targets=labels)
            
            # backpropagation
            loss.backward()

            # update w
            optimizer.step()

            # update running_info
            running_info['train_loss'] += loss.item()*labels.size(0)
            running_info['train_size'] += labels.size(0)



        # Turn off training mode for reporting test loss
        network.train(False)

        # per epoch test activity
        for inputs, labels in testloader:

 
            inputs,labels = best_device(inputs, labels)

            # forward propagation
            outs = network(inputs)
            outs = outs.view(-1)

            # update running_info
            running_info['test_loss'] += loss.item()*labels.size(0)
            running_info['test_size'] += labels.size(0)

            preds = (outs > 0.5).type(torch.FloatTensor)
            running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN'] = e_confusion_matrix(preds,labels)

        
        train_loss = running_info['train_loss']/running_info['train_size']
        test_loss = running_info['test_loss']/running_info['test_size']

        confusion_matrix = running_info['TP'],running_info['FP'],running_info['TN'],running_info['FN']
        regular_accuracy,balanced_accuracy = e_accuracy(confusion_matrix)
        

        print(f'[Epoch {e + 1:2d}/{epochs:d}]: train_loss = {train_loss:.4f}, test_loss = {test_loss:.4f}, RA = {regular_accuracy:.4f}, BA: {balanced_accuracy:.4f}, CM:{confusion_matrix}')

        if regular_accuracy > model.ra:
            model.update(
                network,
                epochs=e + 1,
                ba=balanced_accuracy,
                ra=regular_accuracy,
            )
            best_state = copy.deepcopy(network.state_dict())

    network.load_state_dict(best_state)
    model.network = network
    
    return model



# # Execution

def run(config):
    dataset_directory = REPOSITORY_ROOT / "dataset/merged"
    trainset_dataframe = extract(
        dataset_directory / f"{config['dataset']}_train.pickle"
    )
    testset_dataframe = extract(
        dataset_directory / f"{config['dataset']}_test.pickle"
    )
    
    # a dictionary of model on different personality dimension, "O", "C", "E", "A", "N"
    if config["dataset"] =="essays":
        models = {"O":None, "C":None, "E":None, "A":None, "N":None}
    elif config["dataset"] =="mbti":
        models = {"O":None, "C":None, "E":None, "A":None}
    else:
        raise Exception("dataset name not found")
    

    if config["feature"] != "bow+psycho" and config["feature"] != "bow" and config["feature"] != "psycho":
        raise ValueError("feature must be one of bow, psycho, bow+psycho")
    


    # train the model on different personality dimension using for loop on the key of the dictionary
    for dimension in models.keys():

        if config["loss"] == "bce":
            loss_function = BCE()

        # weighted
        elif config["loss"] == "wbces":
            loss_function = WBCEs(trainset_dataframe[dimension])
        elif config["loss"] == "wbceb":
            loss_function = WBCEb()
        elif config["loss"] == "bbces":
            loss_function = BBCEs(trainset_dataframe[dimension])
        elif config["loss"] == "bbceb":
            loss_function = BBCEb()
        
        # focal loss
        elif config["loss"] == "cfbce":
            loss_function = CFBCE()
        elif config["loss"] == "wfbces":
            loss_function = WFBCEb()
        else:
            raise Exception("loss function not found")

        print("P_"+dimension)

        custom_trainset = CustomDataset(dataframe=trainset_dataframe,dimension=dimension, feature=config["feature"])
        custom_textset = CustomDataset(dataframe=testset_dataframe,dimension=dimension, feature=config["feature"])

        batch_size = 256
        trainloader = DataLoader(custom_trainset, batch_size=batch_size, shuffle=False)
        testloader = DataLoader(custom_textset,batch_size=batch_size, shuffle=False)

        model = train(loss_function = loss_function, epochs= config["epochs"], trainloader=trainloader,testloader = testloader)
        
        models[dimension] = model

    if config["feature"] != "bow":
        raise ValueError(
            "Deployment currently supports feature='bow' because raw-text "
            "psycholinguistic feature extraction is not included in this repo."
        )

    print("Rebuilding and verifying the TF-IDF vectorizer...")
    vectorizer = load_vectorizer(config["dataset"])
    verify_vectorizer(vectorizer, trainset_dataframe)
    artifact_path = config.get(
        "artifact_path",
        REPOSITORY_ROOT
        / "artifacts"
        / f"{config['dataset']}_{config['feature']}_{config['loss']}.pt",
    )
    saved_path = save_bundle(artifact_path, models, vectorizer, config)
    print(f"Saved deployment bundle to {saved_path}")
    return models

config ={
    "dataset":"essays", #mbti, essays
    "feature":"bow", #bow, psycho, bow+psycho
    "loss":"cfbce", #bce, wbces, wbceb, bbces, bbceb, cfbce, wfbceb
    "epochs":200, #any
    "artifact_path": REPOSITORY_ROOT / "artifacts/essays_bow_cfbce.pt",
}

if __name__ == "__main__":
    print_results(run(config=config))

