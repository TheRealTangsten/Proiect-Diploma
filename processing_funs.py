import csv
import vars
import numpy as np


def strToLabel(name, var=1):
    #var = 0
    if vars.noPrsNick in name:
        if var == 1:
            return 0
        else:
            return [1, 0, 0]
    if vars.lowPrsNick in name:
        if var == 1:
            return 1
        else:
            return [0, 1, 0]
    if vars.hiPrsNick in name:
        if var == 1:
            return 2
        else:
            return [0, 0, 1]

def removeBrackets(string):
    string.replace('[', '').replace(']', '')
    return str(string)

def calcMeans():
    indexes_meaning = []
    mean = np.zeros(2048)
    #global vars.vars.noPrsMean
    #global vars.vars.lowPrsMean
    #global vars.vars.hiPrsMean
    for element in vars.noPrsSamples:
        indexes_meaning = element[1].copy()
        ind = 0
        for sampl in element[2]:
            mean[ind] += sampl
            ind += 1
    mean = mean / len(vars.noPrsSamples)
    vars.noPrsMean.append(indexes_meaning)
    vars.noPrsMean.append(mean.copy())
    vars.noPrsMean.append("No_Pressure")
    #print(mean)
    #plt.figure('Mean Values', figsize=(20, 8)), plt.plot(indexes_meaing, mean, color='green'), plt.show()
    indexes_meaning = []
    mean = np.zeros(2048)
    for element in vars.lowPrsSamples:
        indexes_meaning = element[1].copy()
        ind = 0
        for sampl in element[2]:
            mean[ind] += sampl
            ind += 1
    mean = mean / len(vars.noPrsSamples)
    vars.lowPrsMean.append(indexes_meaning)
    vars.lowPrsMean.append(mean.copy())
    vars.lowPrsMean.append("Low_Pressure")
    #print(mean)
    #plt.figure('Mean Values', figsize=(20, 8)), plt.plot(indexes_meaing, mean, color='red'), plt.show()
    indexes_meaning = []
    mean = np.zeros(2048)
    for element in vars.hiPrsSamples:
        indexes_meaning = element[1].copy()
        ind = 0
        for sampl in element[2]:
            mean[ind] += sampl
            ind += 1
    mean = mean / len(vars.hiPrsSamples)
    vars.hiPrsMean.append(indexes_meaning)
    vars.hiPrsMean.append(mean.copy())
    vars.hiPrsMean.append("High_Pressure")
    #print(mean)
    #plt.figure('Mean Values', figsize=(20, 8)), plt.plot(indexes_meaing, mean, color='black'), plt.show()
    #plt.figure('Mean Values', figsize=(20, 8)), plt.plot(vars.noPrsMean[0], vars.noPrsMean[1], color='green'), plt.plot(vars.lowPrsMean[0], vars.lowPrsMean[1], color='red'), plt.plot(vars.hiPrsMean[0], vars.hiPrsMean[1], color='black'), plt.show()

def process_csv(csv_name):
    indexes = []
    values = []
    iteration = 0
    xlabel = ""
    ylabel = ""
    with open(csv_name, newline='') as csvfile:
        readList = csv.reader(csvfile, delimiter=' ')
        iteration = 0
        for row in readList:  # row is a list obj containing 1 string obj: row[0]
            row_split = removeBrackets(row[0]).split(',')
            if iteration == 0:
                xlabel = row_split[1]
                ylabel = row_split[0]
            else:
                indexes.append(row_split[0])
                values.append(row_split[1])
            iteration += 1
    values2 = [eval(i) for i in values]
    indexes2 = [eval(i) for i in indexes]
    return values2, indexes2, xlabel, ylabel



