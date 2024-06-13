import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import vars
import numpy as np
import processing_funs as pf
import plot_funs as plfs
from sklearn.metrics import confusion_matrix
import perceptron as perc
import data_fetching as dataf
import sklearn
import copy
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_MNIST_train():
    mnist_train_data = np.zeros([60000, 784])
    mnist_train_labels = np.zeros(60000)

    f = open('train-images.idx3-ubyte', 'r', encoding='latin-1')
    g = open('train-labels.idx1-ubyte', 'r', encoding='latin-1')

    byte = f.read(16)  # 4 bytes magic number, 4 bytes nr imag, 4 bytes nr linii, 4 bytes nr coloane
    byte_label = g.read(8)  # 4bytes magic number, 4 bytes nr labels

    mnist_train_data = np.fromfile(f, dtype=np.uint8).reshape(60000, 784)
    mnist_train_data = mnist_train_data.reshape(60000, 1, 28, 28)
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
    byte_label = g.read(8)  # 4bytes magic number, 4 bytes nr labels

    mnist_test_data = np.fromfile(f, dtype=np.uint8).reshape(10000, 784)
    mnist_test_data = mnist_test_data.reshape(10000, 1, 28, 28)
    mnist_test_labels = np.fromfile(g, dtype=np.uint8)

    # Conversii pentru a se potrivi cu procesul de antrenare
    mnist_test_data = mnist_test_data.astype(np.float32)
    mnist_test_labels = mnist_test_labels.astype(np.int64)

    return mnist_test_data, mnist_test_labels

def debug_computeFeatureVectorLen(in_size,stride,kernel_size, padding):
    return (in_size + 2*padding -  kernel_size // 2) + 1


def computeFinalConvSize(init_size, stride, kern, pad, iters):
    init_term = init_size // (stride ** iters)

    pf_term_1 = np.arange(1, iters + 1) * (2 * pad - kern)
    #print(pf_term_1)
    pf_term_2 = 1 / (stride ** np.arange(1, iters + 1))
    #print(pf_term_2)
    pf_term_final = np.sum(pf_term_1 * pf_term_2)

    one_term = np.sum(1 / (stride ** np.arange(0, iters)))

    return np.ceil(init_term + pf_term_final + one_term)


def computeFinalConvSizeWithPol(init_size, stride, kern, pad, iters, pol_stride):
    iters = iters * 2  # account for polling op
    init_term = init_size * (stride ** (-iters // 2)) * (pol_stride ** (-iters // 2))

    # pf_term_stride =  1/(stride**np.ceil(np.arange(1,iters+1)/2))
    pf_term_stride = 1 / (stride ** (np.arange(1, iters + 1) // 2))
    # pf_term_pol_stride = 1/(pol_stride**(np.arange(1,iters+1)//2))
    pf_term_pol_stride = 1 / (pol_stride ** np.ceil(np.arange(1, iters + 1) / 2))
    pf_term_final = (2 * pad - kern) * np.sum(pf_term_pol_stride * pf_term_stride)

    one_term_stride = 1 / (stride ** (np.arange(0, iters) // 2))
    #print(one_term_stride)
    one_term_pol_stride = 1 / (pol_stride ** np.ceil(np.arange(0, iters) / 2))
    #print(one_term_pol_stride)
    one_term_final = np.sum(one_term_stride * one_term_pol_stride)
    #print("Init term: {0}\nPf term: {1}\nOne term: {2}\n".format(init_term,pf_term_final,one_term_final))

    return np.floor(init_term + pf_term_final + one_term_final)

def computeFinalConvSizeWithPol2d(init_size, stride, kern, pad, iters, pol_stride):
    iters = iters * 2  # account for polling op
    init_term = init_size * (stride ** (-iters // 2)) * (pol_stride ** (-iters // 2))
    pf_term_final = 0
    one_term_final = 0
    #print("Init term: {0}\nPf term: {1}\nOne term: {2}\n".format(init_term,pf_term_final,one_term_final))

    return init_term**2
class CNN_1D(nn.Module): # 2 straturi convolutionale, 2 straturi de max pooling , 1 strat intermediar de 6 neuroni, 1 strat de 3 final
    def __init__(self, in_size = 2048, out_channels = (256, 64), nrLabels = 3, kernel_size = (4,3), stride = 4, padding = 0, dilation = 1, lin_out1 = 32, stride_poll = 1):
        super(CNN_1D, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax()

        #convolutional layers don't care for size of vector, they care for number of vectors (in_channels)
        #out_channels is number of features, or number of vectors. These vectors are different between themselves by varying the weights of the sliding window of size(kernel_size). Each vector
        #is the resulting vector size of convoluting the whole window with the desired step over the input vectors

        self.conv_l1 = nn.Conv1d(in_channels=1, out_channels = out_channels[0], kernel_size = kernel_size[0], stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.LeakyReLU()#nn.ReLU()
        self.maxpol_1 = nn.MaxPool1d(kernel_size = kernel_size[0], stride=stride_poll, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.conv_l2 = nn.Conv1d(in_channels= out_channels[0], out_channels = out_channels[1], kernel_size = kernel_size[0], stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.LeakyReLU()#nn.ReLU()
        self.maxpol_2 = nn.MaxPool1d(kernel_size = kernel_size[0], stride=stride_poll, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lin_in_size = int(computeFinalConvSizeWithPol(init_size = in_size, stride = stride, kern = kernel_size[0], pad = padding, iters = 2, pol_stride=stride_poll) * out_channels[1])
        #print("lin in size: ",self.lin_in_size,np.shape(self.lin_in_size),"\nlin_out1: ",lin_out1,np.shape(lin_out1))
        self.lin1  = nn.Linear(in_features = self.lin_in_size, out_features= lin_out1)
        self.relu3 = nn.LeakyReLU()#nn.ReLU()
        self.out = nn.Linear(in_features = lin_out1, out_features= nrLabels)

        #conv_x = nn.Conv2d(in_channels=nr_canale_input, out_channels=nr_canale_output,kernel_size=[linii_filtru, coloane_filtru], stride=[pas_orizontal, pas_vertical],padding=[bordare_linii, bordare_coloane])
        #pool_x =  nn.MaxPool2d(kernel_size = [linii_vecinatate, coloane_vecinatate], stride = [pas_orizontal, pas_vertical])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_batch):
        input_batch = torch.from_numpy(input_batch)
        #print(np.shape(input_batch))
        res_conv1 = self.conv_l1(input_batch) # first layer
        #print("Shape of res_conv1:",np.shape(res_conv1))
        res_relu1 = self.relu1(res_conv1)
        #print("Shape of res_relu1:",np.shape(res_relu1))
        res_mxpol1 = self.maxpol_1(res_relu1)
        #print("Shape of res_mxpol1:",np.shape(res_mxpol1))

        res_conv2 = self.conv_l2(res_mxpol1) # second layer
        #print("Shape of res_conv2:", np.shape(res_conv2))
        res_relu2 = self.relu2(res_conv2)
        #print("Shape of res_relu2:", np.shape(res_relu2))
        res_mxpol2 = self.maxpol_2(res_relu2)

        #print("Shape of res_mxpol2: ",np.shape(res_mxpol2))

        flat = torch.flatten(res_mxpol2, 1, 2)
        #print("Shape of flat: ",np.shape(flat))
        #print("End forward function \n\n")
        res_lin = self.lin1(flat)
        res_relu3 = self.relu3(res_lin)
        output = self.soft(self.out(res_relu3))
        # flat = torch.flatten(rezultat_strat_anterior, 1,3) # Se aplatizeaza dimensiunile 1-3 (adica se obtine ceva de dimensiunea canale x linii x coloane
        return output

class CNN_2D(nn.Module): # 2 straturi convolutionale, 2 straturi de max pooling , 1 strat intermediar de 6 neuroni, 1 strat de 3 final
    def __init__(self, in_size = 28*28, out_channels = (256, 64), nrLabels = 3, kernel_size = (4,4), stride = 4, padding = 0, dilation = 1, lin_out1 = 32, stride_poll = 1):
        super(CNN_2D, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax()
        self.poll_kern = (int(kernel_size[0]/2), int(kernel_size[0]/2))
        #convolutional layers don't care for size of vector, they care for number of vectors (in_channels)
        #out_channels is number of features, or number of vectors. These vectors are different between themselves by varying the weights of the sliding window of size(kernel_size). Each vector
        #is the resulting vector size of convoluting the whole window with the desired step over the input vectors

        self.conv_l1 = nn.Conv2d(in_channels=1, out_channels = out_channels[0], kernel_size = kernel_size, stride=(stride,stride), padding=padding)
        self.relu1 = nn.ReLU()#nn.LeakyReLU()
        self.maxpol_1 = nn.MaxPool2d(kernel_size = self.poll_kern, stride=(stride_poll,stride_poll), padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.conv_l2 = nn.Conv2d(in_channels= out_channels[0], out_channels = out_channels[1], kernel_size = kernel_size, stride=(stride,stride), padding=padding)
        self.relu2 = nn.ReLU()#nn.ReLU()
        self.maxpol_2 = nn.MaxPool2d(kernel_size = self.poll_kern, stride=(stride_poll,stride_poll), padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.lin_in_size = int(computeFinalConvSizeWithPol2d(init_size = 28, stride = stride, kern = kernel_size[0], pad = padding, iters = 2, pol_stride=stride_poll) * out_channels[1])

        self.lin1  = nn.Linear(in_features = self.lin_in_size, out_features= lin_out1)
        self.relu3 = nn.ReLU()#nn.ReLU()
        self.out = nn.Linear(in_features = lin_out1, out_features= nrLabels)

        #conv_x = nn.Conv2d(in_channels=nr_canale_input, out_channels=nr_canale_output,kernel_size=[linii_filtru, coloane_filtru], stride=[pas_orizontal, pas_vertical],padding=[bordare_linii, bordare_coloane])
        #pool_x =  nn.MaxPool2d(kernel_size = [linii_vecinatate, coloane_vecinatate], stride = [pas_orizontal, pas_vertical])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(('\nCUDA\n' if torch.cuda.is_available() else '\nCPU\n'))
        self.to(self.device)

    def forward(self, input_batch):
        input_batch = torch.from_numpy(input_batch)
        #print("shape of input: ",np.shape(input_batch))

        res_conv1 = self.conv_l1(input_batch) # first layer
        #print("shape of res_conv1: ", np.shape(res_conv1))
        res_relu1 = self.relu1(res_conv1)
        #print("shape of res_relu1: ", np.shape(res_relu1))
        res_mxpol1 = self.maxpol_1(res_relu1)
        #print("shape of res_mxpol1: ", np.shape(res_mxpol1))

        res_conv2 = self.conv_l2(res_mxpol1) # second layer
        #print("shape of res_conv2: ", np.shape(res_conv2))
        res_relu2 = self.relu2(res_conv2)
        #print("shape of res_relu2: ", np.shape(res_relu2))
        res_mxpol2 = self.maxpol_2(res_relu2)
        #print("shape of res_mxpol2: ", np.shape(res_mxpol2))

        flat = torch.flatten(res_mxpol2, 1, 3)
        #print("Shape of flattening: ",np.shape(flat))
        #print("value of dense layer size: ",self.lin_in_size)

        res_lin = self.lin1(flat)
        res_relu3 = self.relu3(res_lin)
        output = self.relu(self.out(res_relu3))
        # flat = torch.flatten(rezultat_strat_anterior, 1,3) # Se aplatizeaza dimensiunile 1-3 (adica se obtine ceva de dimensiunea canale x linii x coloane
        return output


def runCNN(wois = 0, woie = 2048, testOnMnist = False, debug = False, preproc = False, preproc1 = False, preproc2 = False):
    input_size = woie - wois
    var = 1

    cnn = CNN_1D(in_size = input_size, out_channels = (256, 64), nrLabels = 3, kernel_size = (4,3), stride = 4, padding = 0, dilation = 1, lin_out1 = 32)
    #mlp = bigMLP(in_size = input_size, neurons_per_layer = neuronsHL, nr_layers = 5, nrLabels=3)
    train_data, train_labels = perc.getTrainData(wois, woie, var)
    test_data, test_labels = perc.getTestData(wois, woie, var)
    if preproc:
        train_data, train_labels = dataf.getTrainDataPreproc(wois, woie, var=var)
        test_data, test_labels = dataf.getTestDataPreproc(wois, woie, var=var)
    if preproc1:
        train_data, train_labels = dataf.getTrainDataPreprocNorm(wois, woie, var=var)
        test_data, test_labels = dataf.getTestDataPreprocNorm(wois, woie, var=var)
    if preproc2:
        train_data, train_labels = dataf.getTrainDataPreprocNormCombo(wois, woie, var=var)
        test_data, test_labels = dataf.getTestDataPreprocNormCombo(wois, woie, var=var)

    batch_size = 30

    if testOnMnist:
        cnn = CNN_2D(in_size = 28*28, out_channels = (256, 64), nrLabels = 10, kernel_size = (2,2), stride = 2, padding = 0, dilation = 1, lin_out1 = 32)
        train_data, train_labels = get_MNIST_train()
        test_data, test_labels = get_MNIST_test()
        batch_size = 128

    optim = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    #optim = torch.optim.SGD(cnn.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss(reduction='sum')

    print(str(input_size) +"\n\n")

    nr_epoci = 60
    nr_iteratii = train_data.shape[0] // batch_size

    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm, :]
    train_labels = train_labels[perm]

    losses_epochs = []
    losses_epochs_test = []
    losses_vector = np.zeros(nr_iteratii)
    accuracy_epochs = []
    accuracy_epochs_test = []

    min_not_changed_counter = 0
    min_losses = 9999999
    ep = 0
    test_acc = 0
    test_scaling_en = True
    test_scaling_counter_wrong = 0
    test_scaling_counter_correct = 0
    keep_test_scaling_en = False
    print("Train Labels Shape: ", np.shape(train_labels))

    max_performance = 0
    top_model = cnn
    top_model_state_dict = copy.deepcopy(cnn.state_dict())
    time_start = time.time()  # --------------------TIMER--------------------

    count = count_parameters(cnn)
    print("\n---NUMBER OF PARAMETERS OF CNN: {} ---\n".format(count))


    tm_inf_s = 0
    tm_inf_e = 0
    tm_infs = []
    tm_inf_avg = 0
    #for ep in range(nr_epoci):
    while min_not_changed_counter < 20 and test_acc < 0.9999 and min_losses > 0.01:
        predictii = []
        etichete = []

        for it in range(nr_iteratii):
            batch_data = train_data[it * batch_size: it * batch_size + batch_size, :]
            batch_labels = train_labels[it * batch_size: it * batch_size + batch_size]
            if testOnMnist == False:
                batch_data = np.expand_dims(batch_data, axis = 1)
            #print("Batch data shape: ",batch_data.shape)
            #print("Batch labels shape: ",batch_labels.shape)

            current_predict = cnn.forward(batch_data)

            #print(str(current_predict.shape) + " " + str(torch.from_numpy(batch_labels).shape),current_predict)
            #current_predict = current_predict.astype(np.float32) # if you remove this, it will give you error for loss function: it only takes float or complex. :(
            #print("Iteration: {0}.0\nCurr predict: {1}".format(it, current_predict))
            loss = loss_function(current_predict, torch.from_numpy(batch_labels))
            losses_vector[it] = loss
            #print("LOSS: ", loss)

            current_predict = np.argmax(current_predict.clone().detach().numpy(), axis=1)
            predictii = np.concatenate((predictii, current_predict))
            etichete = np.concatenate((etichete, batch_labels))

            #print("Iteration: {0}\nCurr predict: {1}\nActual labels: {2}".format(it, current_predict, etichete))
            #print("Iteration: {0}.1\nCurr predict: {1}".format(it, current_predict))

            optim.zero_grad()
            loss.backward() # <--- gives error here, doesn't like data format maybe ?
            optim.step()
        epoch_loss = np.average(losses_vector)
        losses_epochs.append(epoch_loss)
        acc = np.sum(predictii == etichete) / len(predictii)
        accuracy_epochs.append(acc*100)
        print('Acuratetea la epoca {} este {}%'.format(ep + 1, acc * 100))
        print('Loss-ul la epoca {} este {}'.format(ep + 1, epoch_loss))
        print('Min Loss la epoca {} este {}'.format(ep + 1, min_losses))
        #conf_mat = confusion_matrix(y_pred=predictii, y_true=etichete, labels=[0,1,2])
        conf_mat = confusion_matrix(y_pred=predictii, y_true=etichete, labels=list(set(etichete)))
        print("Confusion matrix: \n{0}".format(conf_mat))
        with torch.no_grad():
            tst_data = test_data
            if testOnMnist == False:
                tst_data = np.expand_dims(tst_data, axis=1)
            tm_inf_s = time.time()
            curr_pred = cnn.forward(tst_data)
            tm_inf_e = time.time()
            print("---INFERRENCE TIME: {}s".format((tm_inf_e - tm_inf_s)))
            tm_infs.append((tm_inf_e - tm_inf_s))
            test_pred = np.argmax(curr_pred.detach().numpy(), axis=1)
            # print("Shape of curr pred:",np.shape(curr_pred))
            # print("Shape of curr test_pred:",np.shape(test_pred))
            # print("Shape of curr test_labels:",np.shape(test_labels))
            tst_lbls = test_labels
            test_acc = np.sum(test_pred == tst_lbls) / len(test_pred)
            loss_test_epoch = loss_function(curr_pred, torch.from_numpy(test_labels))
            factor_scale_loss_test = loss_test_epoch // epoch_loss
            #if factor_scale_loss_test > 3:
            #    loss_test_epoch = loss_test_epoch // factor_scale_loss_test
            if factor_scale_loss_test > 3 and (test_scaling_en == True or keep_test_scaling_en == True):
                loss_test_epoch = loss_test_epoch // (factor_scale_loss_test - 1)
                test_scaling_counter_correct += 1
            else:
                test_scaling_counter_wrong += 1
            if test_scaling_counter_wrong > 4:
                test_scaling_en = False
            if test_scaling_counter_correct >= 10:
                keep_test_scaling_en = True
            print('\n-------\nAcuratetea la test este {}%\n-------\n'.format(test_acc * 100))
            accuracy_epochs_test.append(test_acc * 100)
            losses_epochs_test.append(loss_test_epoch)
        ep += 1

        if test_acc*100 > max_performance:
            #torch.save(mlp.state_dict(), vars.path_models)
            top_model_state_dict = copy.deepcopy(cnn.state_dict())
            max_performance = test_acc*100
        if (epoch_loss < min_losses):
            min_losses = epoch_loss
            min_not_changed_counter = 0
        else:
            min_not_changed_counter += 1

        perm = np.random.permutation(train_data.shape[0])
        train_data = train_data[perm, :]
        train_labels = train_labels[perm]

    time_end = time.time()  # --------------------TIMER--------------------
    plt.figure(),plt.plot(losses_epochs, color="blue", label="Antrenare"),plt.xlabel("Epoci"), plt.ylabel("Valoare pierderi"), plt.title("Pierderi per epoca")
    plt.plot(losses_epochs_test, color="gold", label="Test"), plt.legend(loc="upper left")
    plt.figure(),plt.plot(accuracy_epochs, color="blue", label="Antrenare"), plt.xlabel("Epoci"), plt.ylabel("Acuratete [%]"), plt.title("Progresia acuratetii per epoca")
    plt.plot(accuracy_epochs_test, color="gold", label="Test"), plt.legend(loc="upper left")

    tm_inf_avg = np.average(np.array(tm_infs))
    #test_data = np.expand_dims(test_data, axis=1)
    with torch.no_grad():
        tst_data = test_data
        if testOnMnist == False:
            tst_data = np.expand_dims(tst_data, axis=1)
        top_model.load_state_dict(top_model_state_dict)
        curr_pred = top_model.forward(tst_data)
        #curr_pred = cnn.forward(tst_data)
        test_pred = np.argmax(curr_pred.detach().numpy(), axis=1)
        # print("Shape of curr pred:",np.shape(curr_pred))
        # print("Shape of curr test_pred:",np.shape(test_pred))
        # print("Shape of curr test_labels:",np.shape(test_labels))
        tst_lbls = test_labels
        test_acc = np.sum(test_pred == tst_lbls) / len(test_pred)
        loss_test_epoch = loss_function(curr_pred, torch.from_numpy(test_labels))
        factor_scale_loss_test = loss_test_epoch // epoch_loss
        if factor_scale_loss_test > 3 and (test_scaling_en == True or keep_test_scaling_en == True):
            loss_test_epoch = loss_test_epoch // (factor_scale_loss_test - 1)
            test_scaling_counter_correct += 1
        else:
            test_scaling_counter_wrong += 1
        if test_scaling_counter_wrong > 4:
            test_scaling_en = False
        if test_scaling_counter_correct >= 10:
            keep_test_scaling_en = True
        print('\n-------\nAcuratetea la test este {}%, acurateta maxima recorded: {}%\n-------\n'.format(test_acc * 100, max_performance))
        print('\n-------\nLoss la test este {}%\n-------\n'.format(loss_test_epoch))
        print('\n-------\nTimp de antrenare este {}%\n-------\n'.format(time_end-time_start))
        print('\n-------\nTimp de inferenta este {}%\n-------\n'.format(tm_inf_avg))
        predicts_final = np.argmax(curr_pred.clone().detach().numpy(), axis=1)
        conf_mat_final = confusion_matrix(y_pred=predicts_final, y_true=tst_lbls, labels=list(set(tst_lbls)))
        conf_mat_final_graphic = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat_final)
        conf_mat_final_graphic.plot()
    plt.show()

