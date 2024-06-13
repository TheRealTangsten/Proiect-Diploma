import csv
import numpy as np
import scipy.signal
import vars as vars
import sklearn

def preprocessVect(in_signal): # EA
    removed_offset = in_signal + np.negative(np.average(in_signal))
    abs_sig = abs(removed_offset)
    lpf_cutoff = 499999.0
    sampling_freq = 24000000.0
    order = 1
    b, a = scipy.signal.butter(N=order, Wn=lpf_cutoff, fs=sampling_freq, btype = "low", analog = False)
    y = scipy.signal.lfilter(b, a, abs_sig)
    return y

def preprocessVectNorm(in_signal, offset = 0): # NORM
    y= in_signal
    y = sklearn.preprocessing.normalize([y])
    z = y[0] + offset
    return z

def preprocessVectNormCombo(in_signal, offset= 0): # EA+NORM
    res = preprocessVect(in_signal)
    res2 = preprocessVectNorm(res)
    return res2

def obtainFullProcessed():
    vars.noPrsSamplesProcessed = vars.noPrsSamples
    vars.noPrsSamplesProcessed = [list(t) for t in vars.noPrsSamples]
    vars.lowPrsSamplesProcessed = vars.lowPrsSamples
    vars.lowPrsSamplesProcessed = [list(t) for t in vars.lowPrsSamples]
    vars.hiPrsSamplesProcessed = vars.hiPrsSamples
    vars.hiPrsSamplesProcessed = [list(t) for t in vars.hiPrsSamples]
    samplesPerClass = vars.nrSamplesPerClass

    #noPressure
    for i in range(0, samplesPerClass):
        vars.noPrsSamplesProcessed[i][2] = preprocessVect(vars.noPrsSamples[i][2])

    #lowPressure
    for i in range(0, samplesPerClass):
        vars.lowPrsSamplesProcessed[i][2] = preprocessVect(vars.lowPrsSamples[i][2])

    #highPressure
    for i in range(0, samplesPerClass):
        vars.hiPrsSamplesProcessed[i][2] = preprocessVect(vars.hiPrsSamples[i][2])

def obtainFullProcessedNorm():
    vars.noPrsSamplesProcessedNorm = vars.noPrsSamples
    vars.noPrsSamplesProcessedNorm = [list(t) for t in vars.noPrsSamples]
    vars.lowPrsSamplesProcessedNorm = vars.lowPrsSamples
    vars.lowPrsSamplesProcessedNorm = [list(t) for t in vars.lowPrsSamples]
    vars.hiPrsSamplesProcessedNorm = vars.hiPrsSamples
    vars.hiPrsSamplesProcessedNorm = [list(t) for t in vars.hiPrsSamples]
    samplesPerClass = vars.nrSamplesPerClass

    #noPressure
    for i in range(0, samplesPerClass):
        vars.noPrsSamplesProcessedNorm[i][2] = preprocessVectNorm(vars.noPrsSamples[i][2])

    #lowPressure
    for i in range(0, samplesPerClass):
        vars.lowPrsSamplesProcessedNorm[i][2] = preprocessVectNorm(vars.lowPrsSamples[i][2], offset=0)

    #highPressure
    for i in range(0, samplesPerClass):
        vars.hiPrsSamplesProcessedNorm[i][2] = preprocessVectNorm(vars.hiPrsSamples[i][2], offset=0)

def obtainFullProcessedNormCombo():
    vars.noPrsSamplesProcessedNormCombo = vars.noPrsSamples
    vars.noPrsSamplesProcessedNormCombo = [list(t) for t in vars.noPrsSamples]
    vars.lowPrsSamplesProcessedNormCombo = vars.lowPrsSamples
    vars.lowPrsSamplesProcessedNormCombo = [list(t) for t in vars.lowPrsSamples]
    vars.hiPrsSamplesProcessedNormCombo = vars.hiPrsSamples
    vars.hiPrsSamplesProcessedNormCombo = [list(t) for t in vars.hiPrsSamples]
    samplesPerClass = vars.nrSamplesPerClass

    #noPressure
    for i in range(0, samplesPerClass):
        vars.noPrsSamplesProcessedNormCombo[i][2] = preprocessVectNormCombo(vars.noPrsSamples[i][2])

    #lowPressure
    for i in range(0, samplesPerClass):
        vars.lowPrsSamplesProcessedNormCombo[i][2] = preprocessVectNormCombo(vars.lowPrsSamples[i][2], offset=0)

    #highPressure
    for i in range(0, samplesPerClass):
        vars.hiPrsSamplesProcessedNormCombo[i][2] = preprocessVectNormCombo(vars.hiPrsSamples[i][2], offset=0)

