import numpy as np

test_csv = "20231213_162234_square_gel_plexi_light_touch.csv"
test_csvs = ["20231213_162103_square_gel_plexi_notouch.csv", "20231213_162234_square_gel_plexi_light_touch.csv",
             "20231213_162358_square_gel_plexi_force_touch.csv"]

#path_models = "C:\\Users\\IonescCristi\\PycharmProjects\\Diploma\\Saved_Models"
path_models = "C:\\Users\\Tangsten\\PycharmProjects\\readFromCSV\\Saved_Models"

nrSamplesTotal = 450
nrSamplesPerClass = 150
nrClasses = 3

noPrsNick = "notouch"
lowPrsNick = "light_touch"
hiPrsNick = "force_touch"
noPrsSamples = [] #contains all "nrSamplesPerClass" no pressure samples
noPrsMean = [] # contains mean signal over all no pressure samples
lowPrsSamples = []#contains all "nrSamplesPerClass" low P samples
lowPrsMean = []# contains mean signal over all low P samples
hiPrsSamples = []# contains all "nrSamplesPerClass" high P samples
hiPrsMean = []# contains mean signal over all high P samples
Samples = []# contains ALL pressure samples
#^ tupl = (name, indx, vals, xlabel, ylabel)
figureIndex = 1

############ Euclidian Stuff ####################
euclNoPrs = [] # Euclidian Distance Sum for each no P signal
euclLowPrs = []# Euclidian Distance Sum for each low P signal
euclHiPrs = []# Euclidian Distance Sum for each high P signal

th01 = 0 #threshold between first and second signal
th21 = 0#threshold between second and third signal
classList = ["class 0", "class 1", "class 2"]

euclDictNameToNumber = {"No_Pressure":0, "Low_Pressure":1, "High_Pressure":2}


zeroRef = ('zero-ref', np.zeros(2048))
################################################

noPrsSamplesProcessed = []
lowPrsSamplesProcessed = []
hiPrsSamplesProcessed = []

noPrsSamplesProcessedNorm = []
lowPrsSamplesProcessedNorm = []
hiPrsSamplesProcessedNorm = []

noPrsSamplesProcessedNormCombo = []
lowPrsSamplesProcessedNormCombo = []
hiPrsSamplesProcessedNormCombo = []


#colorVect = ['black','gray','silver','whitesmoke','rosybrown','firebrick','red','darksalmon','sandybrown','moccasin','gold','olivedrab','chartreuse','darkgreen','mediumspringgreen','lightseagreen','darkcyan','deepskyblue','royalblue','mediumpurple',]
colorVect = []
veryBadColors = ["dimgray","dimgrey","gray","grey","silver","lightgray","whitesmoke","white","snow","mistyrose","rosybrown","salmon","tomato","lightsalmon"
                   ,"sienna","seashell","saddlebrown","linen","antiquewhite"
                   ,"blanchedalmond","papayawhip","wheat","oldlace","floralwhite"
                   ,"darkgoldenrod","cornsilk","ivory","beige","lightyellow","chartreuse"
                   ,"palegreen","mediumspringgreen","mediumturquoise","darkslategray","darkslategrey"
                   ,"teal","cyan","powderblue","lightslategray","lightslategrey","slategray"
                   ,"darkblue"]
badColorKeywords = ["light", "gray", "white", "dark"]

