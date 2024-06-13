import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import vars
import data_fetching as dataf

def simplePlot(signal): # plot a singular list
    plt.figure(vars.figureIndex)
    vars.figureIndex+=1
    plt.plot(signal, color='purple', label="Signal")
    plt.xlabel("Sample Index"), plt.ylabel("Sample Amplitude"),plt.legend(loc="upper left"),plt.show()

def listListPlot(listOfSignals): # plot lists from list
    colors = vars.colorVect
    i = 0
    plt.figure(vars.figureIndex)
    vars.figureIndex+=1
    for signal in listOfSignals:
        plt.plot(signal, color=colors[i%len(colors)], label="Signal" +str(i))
        i += 1
    plt.xlabel("Sample Index"), plt.ylabel("Sample Amplitude"), plt.legend(loc="upper left"), plt.show()

def listListPlotMonocolor(listOfSignals, color = "green"): # plot lists from list
    colors = vars.colorVect
    i = 0
    plt.figure(vars.figureIndex)
    vars.figureIndex+=1
    for signal in listOfSignals:
        plt.plot(signal, color=color, label="Signal" +str(i))
        i += 1
    plt.xlabel("Sample Index"), plt.ylabel("Sample Amplitude"), plt.legend(loc="upper left"), plt.show()

def listListPlot3SplitMulticolor(listOfSignals, colors = ('black', 'green', 'red')): # plot lists from list
    length = len(listOfSignals) // 3
    #colors = vars.colorVect
    i = 0
    plt.figure(vars.figureIndex, figsize=(10, 8))
    vars.figureIndex+=1
    for signal in listOfSignals[0:length]:
        if i == 0:
            i+=1
            plt.plot(signal, color=colors[0], label = "Lipsa Presiune")
        else:
            plt.plot(signal, color=colors[0])
    i=0
    for signal in listOfSignals[length:length*2]:
        if i == 0:
            i += 1
            plt.plot(signal, color=colors[1], label="Mica Presiune")
        else:
            plt.plot(signal, color=colors[1])
    i = 0
    for signal in listOfSignals[length*2:length*3]:
        if i == 0:
            i += 1
            plt.plot(signal, color=colors[2], label="Mare Presiune")
        else:
            plt.plot(signal, color=colors[2])
    i = 0
    plt.xlabel("Sample Index"), plt.ylabel("Sample Amplitude"), plt.legend(loc="upper left"), plt.show()

#this only works on tuples of type ("file_name, list_indx, list_vals, xlbl_str, ylbl_str")
def plotGenericFull(plotableSamples = (vars.lowPrsSamples, vars.hiPrsSamples, vars.noPrsSamples), colors = ('green', 'red', 'black') ):
    plt.figure(vars.figureIndex)
    vars.figureIndex+=1
    for i in range(0, len(plotableSamples)):
        for sample in plotableSamples[i]:
            plt.plot(sample[1], sample[2], color=colors[i])
    plt.xlabel(vars.Samples[0][3]), plt.ylabel(vars.Samples[0][4]), plt.show()

#this only works for lists of format [0:"list_indx, 1:"list_vals"]
def plotGenericPartial(plotableSamples = (vars.lowPrsMean, vars.hiPrsMean, vars.noPrsMean), colors = ('green', 'red', 'black') ):
    plt.figure(vars.figureIndex)
    vars.figureIndex+=1
    for i in range(0, len(plotableSamples)):
    #    print(len(plotableSamples[i][0]))
       plt.plot(plotableSamples[i][0], plotableSamples[i][1], color=colors[i])
    plt.xlabel(vars.Samples[0][3]), plt.ylabel(vars.Samples[0][4]), plt.show()


def plotInOrder():
    plt.figure(vars.figureIndex, figsize=(20, 8))
    vars.figureIndex+=1
    index_label = 0
    for sample in vars.noPrsSamples:
        if index_label == 0:
            index_label+=1
            plt.plot(sample[1], sample[2], color='black', label = "noPressure")
        else:
            plt.plot(sample[1], sample[2], color='black')

    index_label = 0
    for sample in vars.lowPrsSamples:
        if index_label == 0:
            index_label +=1
            plt.plot(sample[1], sample[2], color='green', label = "lowPressure")
        else:
            plt.plot(sample[1], sample[2], color='green')
    index_label = 0
    for sample in vars.hiPrsSamples:
        if index_label == 0:
            index_label+=1
            plt.plot(sample[1], sample[2], color='red', label = "highPressure")
        else:
            plt.plot(sample[1], sample[2], color='red')
    plt.xlabel(vars.Samples[0][3]), plt.ylabel(vars.Samples[0][4]),plt.legend(loc="upper left"), plt.show()


def plotKnown():
    plt.figure(vars.figureIndex, figsize=(20, 8))
    vars.figureIndex+=1
    color = 'white'
    for sample in vars.Samples:
        if vars.noPrsNick in sample[0]:
            color = 'black'
        if vars.hiPrsNick in sample[0]:
            color = 'red'
        if vars.lowPrsNick in sample[0]:
            color = 'green'
        plt.plot(sample[1], sample[2], color)
    plt.xlabel(vars.Samples[0][3]), plt.ylabel(vars.Samples[0][4]), plt.show()

def colorTest():
    dummySample = vars.Samples[0][2]
    testSamples = []
    for i in range(0, 100, 5):
        dummySample = np.array(dummySample) + i
        testSamples.append(dummySample)
    listListPlot(testSamples)

def pltTest():
    valx1 = [100, 200, 300]
    valy1 = [1, 2, 3]
    valx2 = [400, 450, 550]
    valy2 = [3, 5, 6]
    plt.plot(valy1, valx1, color = "blue")
    plt.plot(valy2, valx2, color = "green"), plt.show()

def plotKnownPreprocessed():
    samplesPerClass = vars.nrSamplesPerClass
    colors = ('green', 'red', 'black')
    # noPressure
    for i in range(0, samplesPerClass):
        plt.plot(vars.noPrsSamplesProcessed[i][1], vars.noPrsSamplesProcessed[i][2], color=colors[0])

    # lowPressure
    for i in range(0, samplesPerClass):
        plt.plot(vars.lowPrsSamplesProcessed[i][1], vars.lowPrsSamplesProcessed[i][2], color=colors[1])

    # highPressure
    for i in range(0, samplesPerClass):
        plt.plot(vars.hiPrsSamplesProcessed[i][1], vars.hiPrsSamplesProcessed[i][2], color=colors[2])

    plt.xlabel(vars.Samples[0][3]), plt.ylabel(vars.Samples[0][4]), plt.show()

def plotKnownPreprocessedNorm():
    samplesPerClass = vars.nrSamplesPerClass
    colors = ('green', 'red', 'black')
    # noPressure
    for i in range(0, samplesPerClass):
        plt.plot(vars.noPrsSamplesProcessedNorm[i][1], vars.noPrsSamplesProcessedNorm[i][2], color=colors[0])

    # lowPressure
    for i in range(0, samplesPerClass):
        plt.plot(vars.lowPrsSamplesProcessedNorm[i][1], vars.lowPrsSamplesProcessedNorm[i][2], color=colors[1])

    # highPressure
    for i in range(0, samplesPerClass):
        plt.plot(vars.hiPrsSamplesProcessedNorm[i][1], vars.hiPrsSamplesProcessedNorm[i][2], color=colors[2])

    plt.xlabel(vars.Samples[0][3]), plt.ylabel(vars.Samples[0][4]), plt.show()

def plotGetdataTest():
    data, labels = dataf.getTestData()
    length = len(data)
    print(length)

