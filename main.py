import csv
import time

import numpy as np
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import data_fetching
import plot_funs as plfs
import vars
import processing_funs as pf
import euclidian_distance as ed
import torch
import perceptron
import random
import scikit_perceptron as sci_perc
import perc_sk_ex as prcEx
import convolutional_neural_network as CNN
import data_fetching as dataf
import perceptron_MLP_only2class  as prc2
import preprocess_data as preproc

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

#C:\\Users\\Tangsten\\PycharmProjects\\readFromCSV\\OldSamples"
#C:\\Users\\IonescCristi\\PycharmProjects\\Diploma\\OldSamples"

path = ""
#path = "C:\\Users\\IonescCristi\\PycharmProjects\\Diploma\\OldSamples"
path = "C:\\Users\\Tangsten\\PycharmProjects\\readFromCSV\\450Samples_2"

def setup():
    # sort all csv names in appropriate lists, with values attached!
    # Samples strucutre: tupl = (name, indx, vals, xlabel, ylabel)
    for name in os.listdir(path):
        if ".csv" in name:
            if path != "":
                vals, indx, xlabel, ylabel = pf.process_csv((path+"\\"+name))
            else:
                vals, indx, xlabel, ylabel = pf.process_csv(name)
            tupl = (name, indx, vals, xlabel, ylabel)
            if vars.noPrsNick in name:
                vars.noPrsSamples.append(tupl)
            if vars.lowPrsNick in name:
                vars.lowPrsSamples.append(tupl)
            if vars.hiPrsNick in name:
                vars.hiPrsSamples.append(tupl)
            vars.Samples.append(tupl)
    preproc.obtainFullProcessed()
    preproc.obtainFullProcessedNorm()
    preproc.obtainFullProcessedNormCombo()
    #setup color vector for potential plotting scenarios
    clrs = list(colors.CSS4_COLORS.keys())
    for colour in clrs:
        if colour not in vars.veryBadColors:
            vars.colorVect.append(colour)

def euclidianDemo():
        # plfs.plotGenericPartial(plotableSamples = (vars.noPrsMean, vars.lowPrsMean, vars.hiPrsMean), colors = ('blue', 'yellow', 'orange'))
        # test = pf.euclidian_dist(vars.Samples[30][2], vars.noPrsMean[1])

        ed.showAllEucls()  # does euclidian distance between WoIs and plots the results for all refrence types
        # ed.showAllEucls()
        # vars.colorTest()
        # plfs.simplePlot(vars.Samples[0][2])

def demoPreprocess():
    data, labels = dataf.getTrainDataPreprocNormCombo()
    plfs.listListPlot3SplitMulticolor(data)

def computeTimePreprocs():
    time_start = time.time()
    for i in range(0, 10):
        data, labels = dataf.getTrainData()
        data, labels = dataf.getTestData()
    time_end = time.time()
    time_no_preproc = time_end - time_start

    time_start = time.time()
    for i in range(0, 10):
        preproc.obtainFullProcessed()
    time_end = time.time()
    time_ea = (time_end - time_start)/450

    time_start = time.time()
    for i in range(0, 10):
        preproc.obtainFullProcessedNorm()
    time_end = time.time()
    time_norm = (time_end - time_start)/450

    time_start = time.time()
    for i in range(0, 10):
        preproc.obtainFullProcessedNormCombo()
    time_end = time.time()
    time_combo = (time_end - time_start)/450
    print("Total times:\nNormal: {}[s]\nEA: {}[s]\nNORM: {}[s]\nCombo: {}[s]".format(time_no_preproc, time_ea, time_norm, time_combo))
    print(
        "Average times:\nNormal: {}[s]\nEA: {}[s]\nNORM: {}[s]\nCombo: {}[s]".format(time_no_preproc/10, time_ea/10, time_norm/10,
                                                                                   time_combo/10))


setup()
#pf.calcMeans()
#euclidianDemo()

perceptron.runPerceptron(testOnMnist = False, preproc2 = True)
#perceptron.runMLP(neuronsHL=100, testOnMnist=False, preproc2=True, nrLayers=1)
#CNN.runCNN(testOnMnist=False, preproc2=True)
#computeTimePreprocs()

#demoPreprocess()
#plfs.plotGenericFull(plotableSamples = (vars.noPrsSamplesProcessed, vars.lowPrsSamplesProcessed, vars.hiPrsSamplesProcessed))
#plfs.plotKnownPreprocessed()
#plfs.plotKnownPreprocessedNorm()
#prc2.runPerceptron(preproc=True, loPEn=False)
#prc2.runMLP(preproc=True)

#plfs.plotGetdataTest()
#data, labels = dataf.getTrainDataPreprocNorm()
#plfs.listListPlot3SplitMulticolor(data)

#perceptron.runPerceptronMNIST()

#perceptron.runMLPOnMNIST(neuronsHL=1000)
#plfs.pltTest()



#sci_perc.testScikitPercN(iterations=1)
#prc_at2.whatsTheProblemDoc()
#prcEx.iulianPercExample()

#data, labels = dataf.getTrainData(loPEn=False, var=0)
#print("shapes: ", np.shape(data)," ", np.shape(labels))
#print("Labels: ", labels)
#perceptron.verifyData()







