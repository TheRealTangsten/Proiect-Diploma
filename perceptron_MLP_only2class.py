import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import vars
import numpy as np
import processing_funs as pf
import perceptron as perc
import plot_funs as plfs
from sklearn.metrics import confusion_matrix
import data_fetching as dataf

torch.set_grad_enabled(True)
ratioTrain = 0.8
ratioTest = 0.2

# Samples strucutre: tupl = (name, indx, vals, xlabel, ylabel)
def get_MNIST_train():
    mnist_train_data = np.zeros([60000, 784])
    mnist_train_labels = np.zeros(60000)

    f = open('train-images.idx3-ubyte', 'r', encoding='latin-1')
    g = open('train-labels.idx1-ubyte', 'r', encoding='latin-1')

    byte = f.read(16)  # 4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8)  # 4 bytes magic number, 4 bytes nr labels

    mnist_train_data = np.fromfile(f, dtype=np.uint8).reshape(60000, 784)
    mnist_train_labels = np.fromfile(g, dtype=np.uint8)

    # Conversii pentru a se potrivi cu procesul de antrenare
    mnist_train_data = mnist_train_data.astype(np.float32)
    mnist_train_labels = mnist_train_labels.astype(np.int64)

    return mnist_train_data, mnist_train_labels

def get_MNIST_test():
    mnist_test_data = np.zeros([10000, 784])
    mnist_test_labels = np.zeros(10000)

    f = open('t10k-images.idx3-ubyte', 'r', encoding='latin-1')
    g = open('t10k-labels.idx1-ubyte', 'r', encoding='latin-1')

    byte = f.read(16)  # 4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8)  # 4 bytes magic number, 4 bytes nr labels

    mnist_test_data = np.fromfile(f, dtype=np.uint8).reshape(10000, 784)
    mnist_test_labels = np.fromfile(g, dtype=np.uint8)

    # Conversii pentru a se potrivi cu procesul de testare
    mnist_test_data = mnist_test_data.astype(np.float32)
    mnist_test_labels = mnist_test_labels.astype(np.int64)

    return mnist_test_data, mnist_test_labels

def getTrainData(wois = 0, woie = 2048, var = 1):
    nr_samples_train_total = int(vars.nrSamplesTotal * ratioTrain)
    nr_samples_train_perClass = int(vars.nrSamplesPerClass * ratioTrain)
    all_train_samples = np.zeros([nr_samples_train_total, woie-wois])

    #all_train_labels = np.zeros(nr_samples_train_total)

    all_train_labels_list = []
    offset = 0
    #print("All samples shape: ",np.shape(all_train_samples), "\nnoPrsSamples shape: ", len(vars.noPrsSamples))

    #no Pressure
    for i in range(0, nr_samples_train_perClass):
        all_train_samples[i + offset] = vars.noPrsSamples[i][2][wois:woie]
        #all_train_labels[i + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
        all_train_labels_list.append(pf.strToLabel(vars.noPrsSamples[i][0], var = var))
    offset += nr_samples_train_perClass

    #low Pressure
    for i in range(0, nr_samples_train_perClass):
        all_train_samples[i + offset] = vars.lowPrsSamples[i][2][wois:woie]
        #all_train_labels[i + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
        all_train_labels_list.append(pf.strToLabel(vars.lowPrsSamples[i][0], var = var))
    offset += nr_samples_train_perClass

    #high Pressure
    for i in range(0, nr_samples_train_perClass):
        all_train_samples[i + offset] = vars.hiPrsSamples[i][2][wois:woie]
        #all_train_labels[i + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
        all_train_labels_list.append(pf.strToLabel(vars.hiPrsSamples[i][0], var = var))
    offset += nr_samples_train_perClass

    all_train_labels_list = np.array(all_train_labels_list)

    all_train_samples = all_train_samples.astype(np.float32)
    #all_train_labels = all_train_labels.astype(np.int64)
    all_train_labels_list = all_train_labels_list.astype(np.int64)
    print("Label array of lists shape: {0}, Array itself: {1}".format(all_train_labels_list.shape, all_train_labels_list))
    return all_train_samples, all_train_labels_list #all_train_labels

def getTestData(wois = 0, woie = 2048, var = 1):
    nr_samples_test_perClass = int(vars.nrSamplesPerClass * ratioTest)
    nr_samples_test_total = vars.nrClasses*nr_samples_test_perClass

    nr_samples_start = vars.nrSamplesPerClass - nr_samples_test_perClass
    nr_samples_end = vars.nrSamplesPerClass

    all_test_samples = np.zeros([nr_samples_test_total, woie-wois])
    all_test_labels = np.zeros(nr_samples_test_total)

    all_test_labels_list = []
    offset = 0

    #print(vars.nrSamplesPerClass * ratioTest)
    #print(nr_samples_test_perClass)
    #print(str(nr_samples_start) +"  "+ str(nr_samples_end))
    #no Pressure
    for i in range(nr_samples_start, nr_samples_end):
        all_test_samples[i-nr_samples_start + offset] = vars.noPrsSamples[i][2][wois:woie]
        #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.noPrsSamples[i][0])
        all_test_labels_list.append(pf.strToLabel(vars.noPrsSamples[i][0], var = var))
    offset += nr_samples_test_perClass

    #low Pressure
    for i in range(nr_samples_start, nr_samples_end):
        all_test_samples[i-nr_samples_start + offset] = vars.lowPrsSamples[i][2][wois:woie]
        #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.lowPrsSamples[i][0])
        all_test_labels_list.append(pf.strToLabel(vars.lowPrsSamples[i][0], var = var))
    offset += nr_samples_test_perClass

    #high Pressure
    for i in range(nr_samples_start, nr_samples_end):
        all_test_samples[i-nr_samples_start + offset] = vars.hiPrsSamples[i][2][wois:woie]
        #all_test_labels[i-nr_samples_start + offset] = pf.strToLabel(vars.hiPrsSamples[i][0])
        all_test_labels_list.append(pf.strToLabel(vars.hiPrsSamples[i][0], var = var))
    offset += nr_samples_test_perClass

    all_test_labels_list = np.array(all_test_labels_list)

    all_test_samples = all_test_samples.astype(np.float32)
    #all_test_labels = all_test_labels.astype(np.int64)
    all_test_labels_list = all_test_labels_list.astype(np.int64)

    return all_test_samples, all_test_labels_list #all_test_labels

def verifyData(wois = 0, woie = 2048):
    printLabels = []
    checkpoint  = 0
    train_data, train_labels = getTrainData(wois, woie)
    test_data, test_labels = getTestData(wois, woie)
    print("Train Data Shape: " + str(train_data.shape) + "  Train Label Shape: " + str(train_labels.shape))
    print("First train data: " + str(train_data[0]))
    print("First train label: " + str(train_labels[0]))
    for i in range(0, train_labels.shape[0]):
        printLabels.append(train_labels[i])
        if i % 20 == 0:
            print("Lables {0}:{1} : {2}".format(checkpoint,i,printLabels[checkpoint:i]))
            checkpoint = i
    print()
    print("Test Data Shape: " + str(test_data.shape) + "  Test Label Shape: " + str(test_labels.shape))
    print("First test data: " + str(test_data[0]))
    print("First test label: " + str(test_labels[0]))

    data, labels = getTrainData()
    plfs.listListPlot3SplitMulticolor(data)
    plfs.plotInOrder()

class oneLayerMLP(nn.Module):

    def __init__(self, in_size, nrLabels = 1):
        # Pentru a putea folosi mai departe reteaua, este recomandata mostenirea
        # clasei de baza nn.Module
        super(oneLayerMLP, self).__init__()

        #self.hidden_layer0 = nn.Linear(in_size, nr_neurons_layer0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax()

        self.in_out = nn.Linear(in_size, nrLabels)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_batch):
        input_batch = torch.from_numpy(input_batch)
        output = self.soft(self.in_out(input_batch))
        #output = self.sigmoid(self.in_out(input_batch))
        #output = self.relu(self.in_out(input_batch))

        return output

class bigMLP(nn.Module):

    def __init__(self, in_size, neurons_per_layer, nr_layers, nrLabels):
        # Pentru a putea folosi mai departe reteaua, este recomandata mostenirea
        # clasei de baza nn.Module
        super(bigMLP, self).__init__()

        self.layers_lin = nn.ModuleList()
        self.layers_nonLin = nn.ModuleList()

        self.hidden_layer0 = nn.Linear(in_size, neurons_per_layer)
        self.hidden_layer0_relu = nn.ReLU()

        if nr_layers > 1:
            for i in range(0, nr_layers-1):
                self.layers_lin.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                self.layers_nonLin.append(nn.ReLU()) #add RELU here too


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.additional_layers = nr_layers-1

        self.relu = nn.ReLU()
        self.out_nonLin = nn.Softmax()
        self.out_layer = nn.Linear(neurons_per_layer, nrLabels)

    def forward(self, input_batch):
        input_batch = torch.from_numpy(input_batch)

        hidden0_out = self.hidden_layer0_relu(self.hidden_layer0(input_batch))

        if self.additional_layers > 0:
            intermediate_layers = [hidden0_out]
            for i in range(0, self.additional_layers):
                #intermediate_layers.append(self.hidden_layers[i](intermediate_layers[i]))
                intermediate_layers.append(self.layers_nonLin[i](self.layers_lin[i](intermediate_layers[i])))
            last_layer = intermediate_layers[self.additional_layers]
        else:
            last_layer = hidden0_out
        #output = self.out_nonLin(self.out_layer(last_layer))
        output = self.out_layer(last_layer)

        return output

def runPerceptron(wois = 0, woie = 2048, testOnMnist = False, debug = False, nrLbls = 3, var = 0, noPEn = True, loPEn = True, hiPEn = True, preproc = False):
    input_size = woie - wois

    nr_classes = 0
    if noPEn:
        nr_classes+=1
    if loPEn:
        nr_classes+=1
    if hiPEn:
        nr_classes+=1


    mlp = oneLayerMLP(in_size=input_size, nrLabels = nr_classes)
    #mlp = bigMLP(in_size = input_size, neurons_per_layer = neuronsHL, nr_layers = 5, nrLabels=3)
    train_data, train_labels = dataf.getTrainData(wois, woie, var = var, noPEn = noPEn, loPEn=loPEn, hiPEn=hiPEn)
    test_data, test_labels = dataf.getTestData(wois, woie, var = var, noPEn = noPEn, loPEn=loPEn, hiPEn=hiPEn)
    if preproc:
        train_data, train_labels = dataf.getTrainDataPreproc(wois, woie, var=var, noPEn=noPEn, loPEn=loPEn, hiPEn=hiPEn)
        test_data, test_labels = dataf.getTestDataPreproc(wois, woie, var=var, noPEn=noPEn, loPEn=loPEn, hiPEn=hiPEn)
    batch_size = nr_classes * 120

    if testOnMnist:
        mlp = oneLayerMLP(in_size = 28 * 28, nrLabels = 10)
        #mlp = percRef.Retea_MLP(28 * 28, 1000, 10)
        #mlp = bigMLP(in_size = 28 * 28, neurons_per_layer = 1000, nrLabels = 10, nr_layers = 1)
        train_data, train_labels = get_MNIST_train()
        test_data, test_labels = get_MNIST_test()
        batch_size = 128  # Se poate si mai mult in cazul curent, dar este o valoare frecventa

    #optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    optim = torch.optim.SGD(mlp.parameters(), lr=1e-3)
    #loss_function = nn.CrossEntropyLoss(reduction='sum')
    #loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.L1Loss()
    if testOnMnist or var == 1:
        loss_function = nn.CrossEntropyLoss(reduction='sum')
    loss_function.requires_grad = True

    print(str(input_size) +"\n\n")

    nr_epoci = 15
    nr_iteratii = train_data.shape[0] // batch_size  # Din simplitate, vom ignora ultimul batch, daca este incomplet

    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm, :]
    train_labels = train_labels[perm]

    losses_epochs = []
    losses_vector = np.zeros(nr_iteratii)
    accuracy_epochs = []
    print("Train Labels Shape: ", np.shape(train_labels))
    for ep in range(nr_epoci):
        predictii = []
        etichete = []

        for it in range(nr_iteratii):
            batch_data = train_data[it * batch_size: it * batch_size + batch_size, :]
            batch_labels = train_labels[it * batch_size: it * batch_size + batch_size]
            #print(batch_data.shape)
            #print(batch_labels.shape)

            current_predict = mlp.forward(batch_data)

            #print(str(current_predict.shape) + " " + str(torch.from_numpy(batch_labels).shape),current_predict)
            #current_predict = current_predict.astype(np.float32) # if you remove this, it will give you error for loss function: it only takes float or complex. :(
            #print("Iteration: {0}.0\nCurr predict: {1}".format(it, current_predict))
            loss = loss_function(current_predict, torch.from_numpy(batch_labels))
            losses_vector[it] = loss
            #print("LOSS: ", loss)

            current_predict = np.argmax(current_predict.clone().detach().numpy(), axis=1)
            predictii = np.concatenate((predictii, current_predict))
            if testOnMnist:
                etichete = np.concatenate((etichete, batch_labels))
            else:
                etichete = np.concatenate((etichete, np.argmax(batch_labels, axis = 1)))
            #print("Iteration: {0}\nCurr predict: {1}\nActual labels: {2}".format(it, current_predict, etichete))
            #print("Iteration: {0}.1\nCurr predict: {1}".format(it, current_predict))

            optim.zero_grad()
            loss.backward() # <--- gives error here, doesn't like data format maybe ?
            optim.step()
        losses_epochs.append(np.average(losses_vector))
        acc = np.sum(predictii == etichete) / len(predictii)
        #accuracy_epochs.append(acc*100)
        print('Acuratetea la epoca {} este {}%'.format(ep + 1, acc * 100))
        #conf_mat = confusion_matrix(y_pred=predictii, y_true=etichete, labels=[0,1,2])
        conf_mat = confusion_matrix(y_pred=predictii, y_true=etichete, labels=list(set(etichete)))
        print("Confusion matrix: \n{0}".format(conf_mat))
        with torch.no_grad():
            curr_pred = mlp.forward(test_data)
            test_pred = np.argmax(curr_pred.detach().numpy(), axis=1)
            #print("Shape of curr pred:",np.shape(curr_pred))
            #print("Shape of curr test_pred:",np.shape(test_pred))
            #print("Shape of curr test_labels:",np.shape(test_labels))
            if testOnMnist == False:
                tst_lbls = np.argmax(test_labels, axis=1)
            test_acc = np.sum(test_pred == tst_lbls) / len(test_pred)
            print('\n-------\nAcuratetea la test este {}%\n-------\n'.format(test_acc * 100))
            accuracy_epochs.append(test_acc * 100)

        perm = np.random.permutation(train_data.shape[0])
        train_data = train_data[perm, :]
        train_labels = train_labels[perm]
    plt.figure(),plt.plot(losses_epochs),plt.xlabel("Epoci"), plt.ylabel("Valoare pierderi"), plt.title(" Pierderi la antrenare per epoca ")
    plt.figure(),plt.plot(accuracy_epochs), plt.xlabel("Epoci"), plt.ylabel("Acuratete [%]"), plt.title("Progresia acuratetii la antrenare ")
    if debug:
        print("Losses: ",losses_epochs,"\nAccuracies: ",accuracy_epochs)


    plt.show()

def runMLP(wois = 0, woie = 2048, neuronsHL = 100, nrLayers = 1, testOnMnist = False, debug = False, nrLbls = 3, var = 0, noPEn = True, loPEn = True, hiPEn = True, preproc = False):
    input_size = woie - wois
    nr_classes = 0
    if noPEn:
        nr_classes+=1
    if loPEn:
        nr_classes+=1
    if hiPEn:
        nr_classes+=1

    #mlp = oneLayerMLP(in_size=input_size, nrLabels = nr_classes)
    mlp = bigMLP(in_size = input_size, neurons_per_layer = neuronsHL, nr_layers = nrLayers, nrLabels=nr_classes )
    train_data, train_labels = dataf.getTrainData(wois, woie, var = var, noPEn = noPEn, loPEn=loPEn, hiPEn=hiPEn)
    test_data, test_labels = dataf.getTestData(wois, woie, var = var, noPEn = noPEn, loPEn=loPEn, hiPEn=hiPEn)
    if preproc:
        print("YAAAS\n")
        train_data, train_labels = dataf.getTrainDataPreproc(wois, woie, var=var, noPEn=noPEn, loPEn=loPEn, hiPEn=hiPEn)
        test_data, test_labels = dataf.getTestDataPreproc(wois, woie, var=var, noPEn=noPEn, loPEn=loPEn, hiPEn=hiPEn)
    batch_size = nr_classes * 120

    if testOnMnist:
        #mlp = oneLayerMLP(28 * 28, 1000, 10)
        #mlp = percRef.Retea_MLP(28 * 28, 1000, 10)
        mlp = perc.bigMLP(in_size = 28 * 28, neurons_per_layer = 1000, nrLabels = 10, nr_layers = 1)
        train_data, train_labels = get_MNIST_train()
        test_data, test_labels = get_MNIST_test()
        batch_size = 128  # Se poate si mai mult in cazul curent, dar este o valoare frecventa

    optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    #optim = torch.optim.SGD(mlp.parameters(), lr=1e-3)
    #loss_function = nn.CrossEntropyLoss(reduction='sum')
    #loss_function = nn.BCEWithLogitsLoss()
    loss_function = nn.L1Loss()
    if testOnMnist or var==1:
        loss_function = nn.CrossEntropyLoss(reduction='sum')

    loss_function.requires_grad = True

    print(str(input_size) +" "+ str(neuronsHL)+"\n\n")

    nr_epoci = 15
    nr_iteratii = train_data.shape[0] // batch_size

    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm, :]
    train_labels = train_labels[perm]

    losses_epochs = []
    losses_vector = np.zeros(nr_iteratii)
    accuracy_epochs = []

    for ep in range(nr_epoci):
        predictii = []
        etichete = []

        for it in range(nr_iteratii):
            batch_data = train_data[it * batch_size: it * batch_size + batch_size, :]
            batch_labels = train_labels[it * batch_size: it * batch_size + batch_size]
            # print(batch_data.shape)
            # print(batch_labels.shape)

            current_predict = mlp.forward(batch_data)
            print("Shape of current_predict: ",np.shape(current_predict))
            print("Shape of batch_labels: ", np.shape(batch_labels))
            loss = loss_function(current_predict, torch.from_numpy(batch_labels))
            losses_vector[it] = loss
            # print("LOSS: ", loss)

            current_predict = np.argmax(current_predict.clone().detach().numpy(), axis=1)
            predictii = np.concatenate((predictii, current_predict))
            if testOnMnist:
                etichete = np.concatenate((etichete, batch_labels))
            else:
                etichete = np.concatenate((etichete, np.argmax(batch_labels, axis=1)))

            optim.zero_grad()
            loss.backward()
            optim.step()
        losses_epochs.append(np.average(losses_vector))
        acc = np.sum(predictii == etichete) / len(predictii)
        #accuracy_epochs.append(acc * 100)
        print('Acuratetea la epoca {} este {}%'.format(ep + 1, acc * 100))
        # conf_mat = confusion_matrix(y_pred=predictii, y_true=etichete, labels=[0,1,2])
        conf_mat = confusion_matrix(y_pred=predictii, y_true=etichete, labels=list(set(etichete)))
        print("Confusion matrix: \n{0}".format(conf_mat))
        with torch.no_grad():
            curr_pred = mlp.forward(test_data)
            test_pred = np.argmax(curr_pred.detach().numpy(), axis=1)
            if testOnMnist == False and var == 0:
                tst_lbls = np.argmax(test_labels, axis=1)
            test_acc = np.sum(test_pred == tst_lbls) / len(test_pred)
            accuracy_epochs.append(test_acc*100)

        perm = np.random.permutation(train_data.shape[0])
        train_data = train_data[perm, :]
        train_labels = train_labels[perm]
    plt.figure(),plt.plot(losses_epochs),plt.xlabel("Epoci"), plt.ylabel("Valoare pierderi"), plt.title(" Pierderi la antrenare per epoca ")
    plt.figure(),plt.plot(accuracy_epochs), plt.xlabel("Epoci"), plt.ylabel("Acuratete [%]"), plt.title("Progresia acuratetii la antrenare ")
    if debug:
        print("Losses: ",losses_epochs,"\nAccuracies: ",accuracy_epochs)
    plt.show()

