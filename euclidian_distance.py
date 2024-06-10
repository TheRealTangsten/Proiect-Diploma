import copy
import csv
import vars
import numpy as np
import matplotlib.pyplot as plt

listOfPlots = []
def euclidian_dist(signal, refrence): #returns euclidian distance between signal and ref signal
    eulcDist = np.sqrt(np.sum(((np.array(signal) - np.array(refrence))**2)))
    return eulcDist

def calc_eucl_dists(signals, refrence, woi_start=0, woi_end = 2048):
    vars.euclNoPrs.clear()
    vars.euclLowPrs.clear()
    vars.euclHiPrs.clear()
    i = 0
    for sampl in signals:
        nameSignal = signals[i][0]
        valsSample = signals[i][2]
        vaslRef = refrence[1]
        if vars.noPrsNick in nameSignal:
            vars.euclNoPrs.append(euclidian_dist(valsSample[woi_start:woi_end],vaslRef[woi_start:woi_end]))
        if vars.lowPrsNick in nameSignal:
            vars.euclLowPrs.append(euclidian_dist(valsSample[woi_start:woi_end],vaslRef[woi_start:woi_end]))
        if vars.hiPrsNick in nameSignal:
            vars.euclHiPrs.append(euclidian_dist(valsSample[woi_start:woi_end],vaslRef[woi_start:woi_end]))
        i+=1

def calculate_ths(scenario):
    # 1 - no pressure ref, 2 - low P ref, 3 - High P ref, 4 - Zero ref
    if scenario == 1:
        vars.th01 = (np.sum(vars.euclNoPrs) + np.sum(vars.euclLowPrs)) / (vars.nrSamplesPerClass*2)
        vars.th21 = (np.sum(vars.euclLowPrs) + np.sum(vars.euclHiPrs)) / (vars.nrSamplesPerClass*2)
        vars.classList = ["No_Pressure", "Low_Pressure", "High_Pressure"]
    elif scenario == 2:
        vars.th01 = (np.sum(vars.euclHiPrs) + np.sum(vars.euclLowPrs)) / (vars.nrSamplesPerClass*2)
        vars.th21 = (np.sum(vars.euclNoPrs) + np.sum(vars.euclHiPrs)) / (vars.nrSamplesPerClass*2)
        vars.classList = ["Low_Pressure", "High_Pressure", "No_Pressure"]
    elif scenario == 3:
        vars.th01 = (np.sum(vars.euclHiPrs) + np.sum(vars.euclLowPrs)) / (vars.nrSamplesPerClass*2)
        vars.th21 = (np.sum(vars.euclLowPrs) + np.sum(vars.euclNoPrs)) / (vars.nrSamplesPerClass*2)
        vars.classList = ["High_Pressure", "Low_Pressure", "No_Pressure"]
    elif scenario == 4:
        vars.th01 = (np.sum(vars.euclHiPrs) + np.sum(vars.euclLowPrs)) / (vars.nrSamplesPerClass*2)
        vars.th21 = (np.sum(vars.euclLowPrs) + np.sum(vars.euclHiPrs)) / (vars.nrSamplesPerClass*2)
        vars.classList = ["High_Pressure", "Low_Pressure", "No_Pressure"]
    vars.euclDictNameToNumber[vars.classList[0]] = 0
    vars.euclDictNameToNumber[vars.classList[1]] = 1
    vars.euclDictNameToNumber[vars.classList[2]] = 2

def calculateError():
    data = [(vars.euclNoPrs, "No_Pressure"), (vars.euclLowPrs, "Low_Pressure"), (vars.euclHiPrs, "High_Pressure")]
    class0Count = 0
    class1Count = 0
    class2Count = 0
    error = 0
    for set in data:
        name = set[1]
        for sampl in set[0]:
            if sampl > vars.th21: # assign class
                samplClass = 2
            elif sampl > vars.th01:
                samplClass = 1
            else:
                samplClass = 0
            if samplClass == 2: # see class count
                class2Count += 1
            elif samplClass == 1:
                class1Count += 1
            else:
                class0Count += 1
            if vars.euclDictNameToNumber[name] != samplClass: # check proper labling
                error +=1
    error = error / vars.nrSamplesTotal
    print(vars.classList[0] + " : " + str(class0Count) + "  Cls 0")
    print(vars.classList[1] + " : " + str(class1Count) + "  Cls 1")
    print(vars.classList[2] + " : " + str(class2Count) + "  Cls 2")
    print("Error: " + str(error*100) + "%")
    print(" ")

def eucl_plotNormal(plotableSamples, colors = ('green', 'red', 'black'), titl = "Default Title" ): #plotableSamples needs more than 1 list
    names = ['noP', 'lowP', 'hiP']
    plt.figure()
    j = 0
    th01AsVector = np.zeros(vars.nrSamplesPerClass)
    th01AsVector += vars.th01
    th21AsVector = np.zeros(vars.nrSamplesPerClass)
    th21AsVector += vars.th21
    print(titl + " vec len: " + str(len(plotableSamples)) + "THs: "+ str(vars.th01) + " " + str(vars.th21))
    for i in range(0, len(plotableSamples)):
        plt.plot(plotableSamples[i], color=colors[i%len(colors)], label=names[i])

    j+=1
    plt.plot(th01AsVector, color = vars.colorVect[j%len(vars.colorVect)], label = "Threshold 01")
    j+=1
    plt.plot(th21AsVector, color = vars.colorVect[j % len(vars.colorVect)], label = "Threshold 21")
    plt.xlabel("Trial number"), plt.ylabel("Eucl_Dist"),plt.legend(loc="upper left"),plt.title(label=titl)



def showAllEucls(ws=0, we=2048):
    calc_eucl_dists(vars.Samples, vars.noPrsMean, woi_start=ws, woi_end=we)
    calculate_ths(1)
    eucl_plotNormal((vars.euclNoPrs, vars.euclLowPrs, vars.euclHiPrs), titl = "No Pressure Ref")
    calculateError()

    calc_eucl_dists(vars.Samples, vars.lowPrsMean, woi_start=ws, woi_end=we)
    calculate_ths(2)
    eucl_plotNormal((vars.euclNoPrs, vars.euclLowPrs, vars.euclHiPrs), titl = "Low Pressure Ref")
    calculateError()

    calc_eucl_dists(vars.Samples, vars.hiPrsMean, woi_start=ws, woi_end=we)
    calculate_ths(3)
    eucl_plotNormal((vars.euclNoPrs, vars.euclLowPrs, vars.euclHiPrs), titl = "High Pressure Ref")
    calculateError()

    calc_eucl_dists(vars.Samples, vars.zeroRef, woi_start=ws, woi_end=we)
    calculate_ths(4)
    eucl_plotNormal((vars.euclNoPrs, vars.euclLowPrs, vars.euclHiPrs), titl = "Zero Ref")
    calculateError()
    plt.show()