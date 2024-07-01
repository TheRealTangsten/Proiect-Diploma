import csv
import time

import numpy as np
import os
import matplotlib
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
import preprocess_metrics as pre_metrics


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

def euclidianDemo(preproc_option =  0):
    # pf.calcMeans()
    if preproc_option == 0:
        pf.calcMeans()
        ed.showAllEuclsPreproc(0, 2048, 0)
    if preproc_option == 1: # EA
        ed.showAllEuclsPreproc(0, 2048, 1)
    if preproc_option == 2: #NORM
        ed.showAllEuclsPreproc(0, 2048, 2)
    if preproc_option == 3: #EA+NORM
        ed.showAllEuclsPreproc(0, 2048, 3)

def demoPreprocess():
    matplotlib.rcParams.update({'font.size': 20})
    data, labels = dataf.getTrainData() #dataf.getTrainDataPreprocNormCombo()
    #data, labels = dataf.getTrainDataPreprocNorm()
    #data, labels = dataf.getTrainDataPreproc()
    plfs.listListPlot3SplitMulticolor(data)
    matplotlib.rcParams.update({'font.size': 12})

def computeTimePreprocs():
    doNothing = 0
    samplesPerClass = vars.nrSamplesPerClass
    time_start = time.time()
    for i in range(0, 100):
        vars.noPrsSamplesProcessed = [list(t) for t in vars.noPrsSamples]
        for j in range(0, samplesPerClass):
            doNothing += 1
    time_end = time.time()
    time_no_preproc = 3*(time_end - time_start)

    time_proc = 0
    time_form = 0

    time_proc_total = 0
    time_form_total = 0
    for i in range(0, 100):
        time_proc, time_form = pre_metrics.obtainFullProcessed()
        time_proc_total += time_proc
        time_form_total += time_form
    time_ea = time_proc_total/450
    time_form_ea = time_form_total/450

    time_proc_total = 0
    time_form_total = 0
    for i in range(0, 100):
        time_proc, time_form = pre_metrics.obtainFullProcessedNorm()
        time_proc_total += time_proc
        time_form_total += time_form

    time_norm = time_proc_total/450
    time_form_norm = time_form_total / 450

    time_proc_total = 0
    time_form_total = 0
    for i in range(0, 100):
        time_proc, time_form = pre_metrics.obtainFullProcessedNormCombo()
        time_proc_total += time_proc
        time_form_total += time_form

    time_combo = time_proc_total/450
    time_form_combo = time_form_total / 450
    print("Total times:\nNormal: {}[s]\nEA: {}[s]\nNORM: {}[s]\nCombo: {}[s]\n\n".format(time_no_preproc, time_ea, time_norm, time_combo))
    print(
        "Average times:\nNormal: {}[s]\nEA: {}[s] - Formalism: {}[s]\nNORM: {}[s] - Formalism: {}[s]\nCombo: {}[s] - Formalism: {}[s]".format(time_no_preproc/100, time_ea/100, time_form_ea/100, time_norm/100, time_form_norm/100,
                                                                                   time_combo/100, time_form_combo/100))


setup()
#computeTimePreprocs()
demoPreprocess()

#pf.calcMeans()
#euclidianDemo(1) # any - raw, 1 - EA, 2 - NORM, 3 - EA+NORM

#perceptron.runPerceptron(testOnMnist = True, preproc2 = False)
#perceptron.runMLP(neuronsHL=100, testOnMnist=False, preproc2=True, nrLayers=1)
#CNN.runCNN(testOnMnist=False, preproc2=True)




#demoPreprocess()
#plfs.plotGenericFull(plotableSamples = (vars.noPrsSamplesProcessedNormCombo, vars.lowPrsSamplesProcessedNormCombo, vars.hiPrsSamplesProcessedNormCombo))

#prc2.runPerceptron(preproc=True, loPEn=False)
#prc2.runMLP(preproc=True)

#plfs.plotGetdataTest()
#data, labels = dataf.getTrainDataPreprocNorm()
#plfs.listListPlot3SplitMulticolor(data)

#perceptron.runPerceptronMNIST()

#perceptron.runMLPOnMNIST(neuronsHL=1000)
#plfs.pltTest()


#ed.showAllEuclsPreproc(0, 2048, 1)
#sci_perc.testScikitPercN(iterations=1)
#prc_at2.whatsTheProblemDoc()
#prcEx.iulianPercExample()

#data, labels = dataf.getTrainData(loPEn=False, var=0)
#print("shapes: ", np.shape(data)," ", np.shape(labels))
#print("Labels: ", labels)
#perceptron.verifyData()







