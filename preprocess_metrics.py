import csv
import numpy as np
import scipy.signal
import vars as vars
import sklearn
import time

def preprocessVect(in_signal): # EA
    time_s = time.time()
    removed_offset = in_signal + np.negative(np.average(in_signal))
    abs_sig = abs(removed_offset)
    lpf_cutoff = 499999.0
    sampling_freq = 24000000.0
    order = 1
    b, a = scipy.signal.butter(N=order, Wn=lpf_cutoff, fs=sampling_freq, btype = "low", analog = False)
    y = scipy.signal.lfilter(b, a, abs_sig)
    time_e = time.time()
    total_time = time_e-time_s
    return y, total_time

def preprocessVectNorm(in_signal, offset = 0): # NORM
    time_s = time.time()
    y= in_signal
    y = sklearn.preprocessing.normalize([y])
    z = y[0] + offset
    time_e = time.time()
    total_time = time_e - time_s
    return z, total_time

def preprocessVectNormCombo(in_signal, offset= 0): # EA+NORM
    time_s = time.time()
    res = preprocessVect(in_signal)[0]
    res2 = preprocessVectNorm(res)[0]
    time_e = time.time()
    total_time = time_e - time_s
    return res2, total_time

def obtainFullProcessed():

    time_proc = 0
    time_formalism_s = time.time()
    vars.noPrsSamplesProcessed = vars.noPrsSamples
    vars.noPrsSamplesProcessed = [list(t) for t in vars.noPrsSamples]
    vars.lowPrsSamplesProcessed = vars.lowPrsSamples
    vars.lowPrsSamplesProcessed = [list(t) for t in vars.lowPrsSamples]
    vars.hiPrsSamplesProcessed = vars.hiPrsSamples
    vars.hiPrsSamplesProcessed = [list(t) for t in vars.hiPrsSamples]
    samplesPerClass = vars.nrSamplesPerClass

    #noPressure
    for i in range(0, samplesPerClass):
        vars.noPrsSamplesProcessed[i][2], t_proc = preprocessVect(vars.noPrsSamples[i][2])
        time_proc += t_proc

    #lowPressure
    for i in range(0, samplesPerClass):
        vars.lowPrsSamplesProcessed[i][2], t_proc = preprocessVect(vars.lowPrsSamples[i][2])
        time_proc += t_proc

    #highPressure
    for i in range(0, samplesPerClass):
        vars.hiPrsSamplesProcessed[i][2], t_proc = preprocessVect(vars.hiPrsSamples[i][2])
        time_proc += t_proc

    vars.samplesPreprocessed = vars.noPrsSamplesProcessed + vars.lowPrsSamplesProcessed + vars.hiPrsSamplesProcessed
    time_formalism_e = time.time()
    total_time_formalism = time_formalism_e - time_formalism_s

    return time_proc, total_time_formalism

def obtainFullProcessedNorm():

    time_proc = 0
    time_formalism_s = time.time()
    vars.noPrsSamplesProcessedNorm = vars.noPrsSamples
    vars.noPrsSamplesProcessedNorm = [list(t) for t in vars.noPrsSamples]
    vars.lowPrsSamplesProcessedNorm = vars.lowPrsSamples
    vars.lowPrsSamplesProcessedNorm = [list(t) for t in vars.lowPrsSamples]
    vars.hiPrsSamplesProcessedNorm = vars.hiPrsSamples
    vars.hiPrsSamplesProcessedNorm = [list(t) for t in vars.hiPrsSamples]
    samplesPerClass = vars.nrSamplesPerClass

    #noPressure
    for i in range(0, samplesPerClass):
        vars.noPrsSamplesProcessedNorm[i][2], t_proc = preprocessVectNorm(vars.noPrsSamples[i][2])
        time_proc += t_proc

    #lowPressure
    for i in range(0, samplesPerClass):
        vars.lowPrsSamplesProcessedNorm[i][2], t_proc = preprocessVectNorm(vars.lowPrsSamples[i][2], offset=0)
        time_proc += t_proc

    #highPressure
    for i in range(0, samplesPerClass):
        vars.hiPrsSamplesProcessedNorm[i][2], t_proc = preprocessVectNorm(vars.hiPrsSamples[i][2], offset=0)
        time_proc += t_proc

    vars.samplesPreprocessedNorm = vars.noPrsSamplesProcessedNorm + vars.lowPrsSamplesProcessedNorm + vars.hiPrsSamplesProcessedNorm
    time_formalism_e = time.time()
    total_time_formalism = time_formalism_e - time_formalism_s

    return time_proc, total_time_formalism


def obtainFullProcessedNormCombo():

    time_proc = 0
    time_formalism_s = time.time()
    vars.noPrsSamplesProcessedNormCombo = vars.noPrsSamples
    vars.noPrsSamplesProcessedNormCombo = [list(t) for t in vars.noPrsSamples]
    vars.lowPrsSamplesProcessedNormCombo = vars.lowPrsSamples
    vars.lowPrsSamplesProcessedNormCombo = [list(t) for t in vars.lowPrsSamples]
    vars.hiPrsSamplesProcessedNormCombo = vars.hiPrsSamples
    vars.hiPrsSamplesProcessedNormCombo = [list(t) for t in vars.hiPrsSamples]
    samplesPerClass = vars.nrSamplesPerClass

    #noPressure
    for i in range(0, samplesPerClass):
        vars.noPrsSamplesProcessedNormCombo[i][2], t_proc = preprocessVectNormCombo(vars.noPrsSamples[i][2])
        time_proc += t_proc

    #lowPressure
    for i in range(0, samplesPerClass):
        vars.lowPrsSamplesProcessedNormCombo[i][2], t_proc = preprocessVectNormCombo(vars.lowPrsSamples[i][2], offset=0)
        time_proc += t_proc

    #highPressure
    for i in range(0, samplesPerClass):
        vars.hiPrsSamplesProcessedNormCombo[i][2], t_proc = preprocessVectNormCombo(vars.hiPrsSamples[i][2], offset=0)
        time_proc += t_proc

    vars.samplesPreprocessedNormCombo = vars.noPrsSamplesProcessedNormCombo + vars.lowPrsSamplesProcessedNormCombo + vars.hiPrsSamplesProcessedNormCombo
    time_formalism_e = time.time()
    total_time_formalism = time_formalism_e - time_formalism_s
    return time_proc, total_time_formalism