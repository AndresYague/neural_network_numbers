import numpy as np
import matplotlib.pyplot as plt
import struct, os, sys
from nn_lib import *

def readMNIST(fil, typ):
    '''Read the MNIST data, depending on the type'''

    size_big = 4
    with open(fil, "rb") as fread:
        # First read the number of items. Remember big endian!
        lineBytes = fread.read(size_big*2)
        magic_num, num_items = struct.unpack(">" + 2 * "i", lineBytes)

        # Read
        if typ == "label":
            lineBytes = fread.read(num_items)
            items = struct.unpack(">" + num_items * "B", lineBytes)
        elif typ == "image":
            # Get the sizes
            lineBytes = fread.read(size_big*2)
            siz1, siz2 = struct.unpack(">" + 2 * "i", lineBytes)

            # Now the items
            items = []
            for ii in range(num_items):
                lineBytes = fread.read(siz1 * siz2)
                img = struct.unpack(">" + siz1 * siz2 * "B", lineBytes)

                # Add it
                items.append(img)

    if typ == "label":
        return items
    elif typ == "image":
        return items, siz1, siz2

def show_number(label, image, siz1, siz2):
    '''Print the label and show the image'''

    print("Label is {}".format(label))

    # Arrange the image
    xx = np.zeros((siz1, siz2))
    for ii in range(siz1):
        init = ii * siz2
        fin = init + siz2
        xx[ii] = image[init:fin]

    plt.imshow(xx)
    plt.show()

def get_fileName(nn, hidden, outpt, cost):
    '''Create a consistent filename for the network'''

    fileName = "saved_nn_{}_[".format(nn)
    for hid in hidden:
        fileName += "{}_".format(hid)
    fileName = fileName[:-1] + "]_{}".format(outpt)
    fileName += "_cost_{:.3f}.npy".format(cost)

    return fileName

def main():
    '''Create and train neural network'''

    # Read the test data
    print("Loading test data...")
    file_lab = "ML_numbers/t10k-labels-idx1-ubyte"
    file_img = "ML_numbers/t10k-images-idx3-ubyte"
    test_lab = readMNIST(file_lab, typ = "label")
    test_img, siz1, siz2 = readMNIST(file_img, typ = "image")
    print("Loaded")

    # Create the network
    nn = siz1 * siz2
    hidden = [30]
    outpt = 10
    numbers_nn = NetworkObject(inpt = nn, hidden = hidden, outpt = outpt,
                                lbda = 1e-3)

    # Get file
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    else:
        fileName = None

    if fileName is not None:

        # If exists, recover thetas
        print("Loading network found in {}".format(fileName))

        thetas = []
        nLayers = 2 + len(hidden)
        with open(fileName, "rb") as fread:
            for ii in range(nLayers - 1):
                thetas.append(np.load(fread))

        numbers_nn.set_thetas(thetas)

    else:
        # Otherwise, train network

        # Read the data
        print("Loading training data...")
        file_lab = "ML_numbers/train-labels-idx1-ubyte"
        file_img = "ML_numbers/train-images-idx3-ubyte"
        train_lab = readMNIST(file_lab, typ = "label")
        train_img, siz1, siz2 = readMNIST(file_img, typ = "image")
        print("Loaded")

        cost = numbers_nn.train(train_img, train_lab, batch_siz = 10,
                             alpha = 1e-3, verbose = True, tol = 1e-5,
                             low_cost = 0.2)

        # Let user know that the network is trained
        print("Trained, last cost is {:.2f}".format(cost))

        fileName = get_fileName(nn, hidden, outpt, cost)

        # Trained, save thetas
        thetas = numbers_nn.get_thetas()
        with open(fileName, "wb") as fwrite:
            for theta in thetas:
                np.save(fwrite, theta)

    # Check network in training set:
    #print("Checking accuracy with training set")
    #correct_cases = 0
    #for img, lab in zip(train_img, train_lab):
        #lab_net = numbers_nn.propagate_indx(img)
        #if lab_net == lab:
            #correct_cases += 1

    #acc = correct_cases/len(train_lab) * 100
    #print("Accuracy = {:.2f}%".format(acc))

    # Now check accuracy with test set:

    # Check network in test set:
    print("Checking accuracy with test set")
    correct_cases = 0
    for img, lab in zip(test_img, test_lab):
        lab_net = numbers_nn.propagate_indx(img)
        if lab_net == lab:
            correct_cases += 1

    acc = correct_cases/len(test_lab) * 100
    print("Accuracy = {:.2f}%".format(acc))

    print("Showing random examples")
    for ii in range(10):
        jj = np.random.choice(range(len(test_lab)))
        idxMax = numbers_nn.propagate_indx(test_img[jj])

        # Print network output and plot number
        print("Network thinks this is a {}".format(idxMax))
        show_number(test_lab[jj], test_img[jj], siz1, siz2)

if __name__ == "__main__":
    main()
