import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import vars
import numpy as np
import processing_funs as pf
import plot_funs as plfs
from sklearn.metrics import confusion_matrix
import perceptron as perc

ratioTrain = 0.8
ratioTest = 0.2

def strToLabel(name, var=1, dict_class ={vars.noPrsNick:0, vars.lowPrsNick:1, vars.hiPrsNick:2}, nr_classes = 3):
    #var = 0
    keys = list(dict_class.keys())
    ret_vect = [0]*nr_classes
    if vars.noPrsNick in name:
        if var == 1:
            return dict_class[keys[0]]
        else:
            ret_vect[dict_class[keys[0]]] = 1
            return ret_vect
    if vars.lowPrsNick in name:
        if var == 1:
            return dict_class[keys[1]]
        else:
            ret_vect[dict_class[keys[1]]] = 1
            return ret_vect
    if vars.hiPrsNick in name:
        if var == 1:
            return dict_class[keys[2]]
        else:
            ret_vect[dict_class[keys[2]]] = 1
            return ret_vect

def getTrainData(wois = 0, woie = 2048, var = 1, noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1

    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_train_total = int( nr_samples_total * ratioTrain)
    nr_samples_train_perClass = int(vars.nrSamplesPerClass * ratioTrain)

    all_train_samples = np.zeros([nr_samples_train_total, woie-wois])

    #all_train_labels = np.zeros(nr_samples_train_total)

    all_train_labels_list = []
    offset = 0
    #print("All samples shape: ",np.shape(all_train_samples), "\nnoPrsSamples shape: ", len(vars.noPrsSamples))

    if noPEn:
        #no Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.noPrsSamples[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.noPrsSamples[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if loPEn:
        #low Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.lowPrsSamples[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.lowPrsSamples[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if hiPEn:
        #high Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.hiPrsSamples[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.hiPrsSamples[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    all_train_labels_list = np.array(all_train_labels_list)

    all_train_samples = all_train_samples.astype(np.float32)
    #all_train_labels = all_train_labels.astype(np.int64)
    all_train_labels_list = all_train_labels_list.astype(np.int64)
   # print("Label array of lists shape: {0}, Array itself: {1}".format(all_train_labels_list.shape, all_train_labels_list))
    return all_train_samples, all_train_labels_list #all_train_labels

def getTestData(wois = 0, woie = 2048, var = 1,noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1
    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_test_perClass = int(vars.nrSamplesPerClass * ratioTest)
    nr_samples_test_total = nr_classes * nr_samples_test_perClass

    nr_samples_start = vars.nrSamplesPerClass - nr_samples_test_perClass
    nr_samples_end = vars.nrSamplesPerClass

    all_test_samples = np.zeros([nr_samples_test_total, woie-wois])
    all_test_labels = np.zeros(nr_samples_test_total)

    all_test_labels_list = []
    offset = 0

    #print(vars.nrSamplesPerClass * ratioTest)
    #print(nr_samples_test_perClass)
    #print(str(nr_samples_start) +"  "+ str(nr_samples_end))

    if noPEn:
        #no Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.noPrsSamples[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.noPrsSamples[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if loPEn:
        #low Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.lowPrsSamples[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.lowPrsSamples[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if hiPEn:
        #high Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.hiPrsSamples[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.hiPrsSamples[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    all_test_labels_list = np.array(all_test_labels_list)

    all_test_samples = all_test_samples.astype(np.float32)
    #all_test_labels = all_test_labels.astype(np.int64)
    all_test_labels_list = all_test_labels_list.astype(np.int64)

    return all_test_samples, all_test_labels_list #all_test_labels


def getTrainDataPreproc(wois = 0, woie = 2048, var = 1, noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1

    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_train_total = int( nr_samples_total * ratioTrain)
    nr_samples_train_perClass = int(vars.nrSamplesPerClass * ratioTrain)

    all_train_samples = np.zeros([nr_samples_train_total, woie-wois])

    #all_train_labels = np.zeros(nr_samples_train_total)

    all_train_labels_list = []
    offset = 0
    #print("All samples shape: ",np.shape(all_train_samples), "\nnoPrsSamples shape: ", len(vars.noPrsSamples))

    if noPEn:
        #no Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.noPrsSamplesProcessed[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.noPrsSamplesProcessed[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if loPEn:
        #low Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.lowPrsSamplesProcessed[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.lowPrsSamplesProcessed[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if hiPEn:
        #high Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.hiPrsSamplesProcessed[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.hiPrsSamplesProcessed[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    all_train_labels_list = np.array(all_train_labels_list)

    all_train_samples = all_train_samples.astype(np.float32)
    #all_train_labels = all_train_labels.astype(np.int64)
    all_train_labels_list = all_train_labels_list.astype(np.int64)
   # print("Label array of lists shape: {0}, Array itself: {1}".format(all_train_labels_list.shape, all_train_labels_list))
    return all_train_samples, all_train_labels_list #all_train_labels


def getTestDataPreproc(wois = 0, woie = 2048, var = 1,noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1
    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_test_perClass = int(vars.nrSamplesPerClass * ratioTest)
    nr_samples_test_total = nr_classes * nr_samples_test_perClass

    nr_samples_start = vars.nrSamplesPerClass - nr_samples_test_perClass
    nr_samples_end = vars.nrSamplesPerClass

    all_test_samples = np.zeros([nr_samples_test_total, woie-wois])
    all_test_labels = np.zeros(nr_samples_test_total)

    all_test_labels_list = []
    offset = 0

    #print(vars.nrSamplesPerClass * ratioTest)
    #print(nr_samples_test_perClass)
    #print(str(nr_samples_start) +"  "+ str(nr_samples_end))

    if noPEn:
        #no Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.noPrsSamplesProcessed[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.noPrsSamplesProcessed[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if loPEn:
        #low Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.lowPrsSamplesProcessed[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.lowPrsSamplesProcessed[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if hiPEn:
        #high Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.hiPrsSamplesProcessed[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.hiPrsSamplesProcessed[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    all_test_labels_list = np.array(all_test_labels_list)

    all_test_samples = all_test_samples.astype(np.float32)
    #all_test_labels = all_test_labels.astype(np.int64)
    all_test_labels_list = all_test_labels_list.astype(np.int64)

    return all_test_samples, all_test_labels_list #all_test_labels


def getTrainDataPreprocNorm(wois = 0, woie = 2048, var = 1, noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1

    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_train_total = int( nr_samples_total * ratioTrain)
    nr_samples_train_perClass = int(vars.nrSamplesPerClass * ratioTrain)

    all_train_samples = np.zeros([nr_samples_train_total, woie-wois])

    #all_train_labels = np.zeros(nr_samples_train_total)

    all_train_labels_list = []
    offset = 0
    #print("All samples shape: ",np.shape(all_train_samples), "\nnoPrsSamples shape: ", len(vars.noPrsSamples))

    if noPEn:
        #no Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.noPrsSamplesProcessedNorm[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.noPrsSamplesProcessedNorm[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if loPEn:
        #low Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.lowPrsSamplesProcessedNorm[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.lowPrsSamplesProcessedNorm[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if hiPEn:
        #high Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.hiPrsSamplesProcessedNorm[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.hiPrsSamplesProcessedNorm[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    all_train_labels_list = np.array(all_train_labels_list)

    all_train_samples = all_train_samples.astype(np.float32)
    #all_train_labels = all_train_labels.astype(np.int64)
    all_train_labels_list = all_train_labels_list.astype(np.int64)
   # print("Label array of lists shape: {0}, Array itself: {1}".format(all_train_labels_list.shape, all_train_labels_list))
    return all_train_samples, all_train_labels_list #all_train_labels

def getTestDataPreprocNorm(wois = 0, woie = 2048, var = 1,noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1
    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_test_perClass = int(vars.nrSamplesPerClass * ratioTest)
    nr_samples_test_total = nr_classes * nr_samples_test_perClass

    nr_samples_start = vars.nrSamplesPerClass - nr_samples_test_perClass
    nr_samples_end = vars.nrSamplesPerClass

    all_test_samples = np.zeros([nr_samples_test_total, woie-wois])
    all_test_labels = np.zeros(nr_samples_test_total)

    all_test_labels_list = []
    offset = 0

    #print(vars.nrSamplesPerClass * ratioTest)
    #print(nr_samples_test_perClass)
    #print(str(nr_samples_start) +"  "+ str(nr_samples_end))

    if noPEn:
        #no Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.noPrsSamplesProcessedNorm[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.noPrsSamplesProcessedNorm[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if loPEn:
        #low Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.lowPrsSamplesProcessedNorm[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.lowPrsSamplesProcessedNorm[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if hiPEn:
        #high Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.hiPrsSamplesProcessedNorm[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.hiPrsSamplesProcessedNorm[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    all_test_labels_list = np.array(all_test_labels_list)

    all_test_samples = all_test_samples.astype(np.float32)
    #all_test_labels = all_test_labels.astype(np.int64)
    all_test_labels_list = all_test_labels_list.astype(np.int64)

    return all_test_samples, all_test_labels_list #all_test_labels

def getTrainDataPreprocNormCombo(wois = 0, woie = 2048, var = 1, noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1

    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_train_total = int( nr_samples_total * ratioTrain)
    nr_samples_train_perClass = int(vars.nrSamplesPerClass * ratioTrain)

    all_train_samples = np.zeros([nr_samples_train_total, woie-wois])

    #all_train_labels = np.zeros(nr_samples_train_total)

    all_train_labels_list = []
    offset = 0
    #print("All samples shape: ",np.shape(all_train_samples), "\nnoPrsSamples shape: ", len(vars.noPrsSamples))

    if noPEn:
        #no Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.noPrsSamplesProcessedNormCombo[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.noPrsSamplesProcessedNormCombo[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if loPEn:
        #low Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.lowPrsSamplesProcessedNormCombo[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.lowPrsSamplesProcessedNormCombo[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    if hiPEn:
        #high Pressure
        for i in range(0, nr_samples_train_perClass):
            all_train_samples[i + offset] = vars.hiPrsSamplesProcessedNormCombo[i][2][wois:woie]
            #all_train_labels[i + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_train_labels_list.append(strToLabel(vars.hiPrsSamplesProcessedNormCombo[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_train_perClass

    all_train_labels_list = np.array(all_train_labels_list)

    all_train_samples = all_train_samples.astype(np.float32)
    #all_train_labels = all_train_labels.astype(np.int64)
    all_train_labels_list = all_train_labels_list.astype(np.int64)
   # print("Label array of lists shape: {0}, Array itself: {1}".format(all_train_labels_list.shape, all_train_labels_list))
    return all_train_samples, all_train_labels_list #all_train_labels

def getTestDataPreprocNormCombo(wois = 0, woie = 2048, var = 1,noPEn = True, loPEn = True, hiPEn = True):
    nr_classes = 0
    dict_classes = {"noP":0, "loP":1, "hiP":2}
    dict_actives = {"noP":1, "loP":1, "hiP":1}
    if noPEn:
        nr_classes+=1
    else:
        dict_actives["noP"] = 0
    if loPEn:
        nr_classes+=1
    else:
        dict_actives["loP"] = 0
    if hiPEn:
        nr_classes+=1
    else:
        dict_actives["hiP"] = 0
    keys_actives = dict_actives.keys()
    cls_index = 0
    for key in keys_actives:
        if dict_actives[key] == 1:
            dict_classes[key] = cls_index
            cls_index += 1
    nr_samples_total  = (vars.nrSamplesTotal // 3) * nr_classes

    nr_samples_test_perClass = int(vars.nrSamplesPerClass * ratioTest)
    nr_samples_test_total = nr_classes * nr_samples_test_perClass

    nr_samples_start = vars.nrSamplesPerClass - nr_samples_test_perClass
    nr_samples_end = vars.nrSamplesPerClass

    all_test_samples = np.zeros([nr_samples_test_total, woie-wois])
    all_test_labels = np.zeros(nr_samples_test_total)

    all_test_labels_list = []
    offset = 0

    #print(vars.nrSamplesPerClass * ratioTest)
    #print(nr_samples_test_perClass)
    #print(str(nr_samples_start) +"  "+ str(nr_samples_end))

    if noPEn:
        #no Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.noPrsSamplesProcessedNormCombo[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.noPrsSamplesProcessedNormCombo[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if loPEn:
        #low Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.lowPrsSamplesProcessedNormCombo[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.lowPrsSamplesProcessedNormCombo[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    if hiPEn:
        #high Pressure
        for i in range(nr_samples_start, nr_samples_end):
            all_test_samples[i-nr_samples_start + offset] = vars.hiPrsSamplesProcessedNormCombo[i][2][wois:woie]
            #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
            all_test_labels_list.append(strToLabel(vars.hiPrsSamplesProcessedNormCombo[i][0], var = var, dict_class=dict_classes, nr_classes = nr_classes))
        offset += nr_samples_test_perClass

    all_test_labels_list = np.array(all_test_labels_list)

    all_test_samples = all_test_samples.astype(np.float32)
    #all_test_labels = all_test_labels.astype(np.int64)
    all_test_labels_list = all_test_labels_list.astype(np.int64)

    return all_test_samples, all_test_labels_list #all_test_labels