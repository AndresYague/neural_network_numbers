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

def show_number(image, siz1, siz2):
    '''Show the image'''

    # Arrange the image
    xx = np.zeros((siz1, siz2))
    for ii in range(siz1):
        init = ii * siz2
        fin = init + siz2
        xx[ii] = image[init:fin]

    plt.imshow(xx)
    plt.show()

def main():
    '''Create and train neural network'''

    # Read the data
    print("Loading training data...")
    file_lab = "ML_numbers/train-labels-idx1-ubyte"
    file_img = "ML_numbers/train-images-idx3-ubyte"
    train_lab = readMNIST(file_lab, typ = "label")
    train_img, siz1, siz2 = readMNIST(file_img, typ = "image")
    print("Loaded")

    # Create the network
    nn = siz1 * siz2
    hidden = [100, 100]
    outpt = 10
    numbers_nn = NetworkObject(inpt = nn, hidden = hidden, outpt = outpt,
                                lbda = 1e-4)

    cost = numbers_nn.train(train_img, train_lab, batch_siz = 10,
                         alpha = 1e-2, verbose = True, tol = 1e-8,
                         low_cost = 1.00, cycle_cost = 2000)
    cost = numbers_nn.train(train_img, train_lab, batch_siz = 10,
                         alpha = 1e-3, verbose = True, tol = 1e-8,
                         low_cost = 0.05, cycle_cost = 2000)

    # Let user know that the network is trained
    print("Trained, last cost is {:.2f}".format(cost))

    # Trained, save thetas
    numbers_nn.save_network(cost)

    # Check network in training set:
    print("Checking accuracy with training set")
    correct_cases = 0
    labels, confidence = numbers_nn.propagate_indx_conf(train_img)
    for ii in range(len(train_lab)):
        if train_lab[ii] == labels[ii]:
            correct_cases += 1

    acc = correct_cases/len(train_lab) * 100
    print("Accuracy = {:.2f}%".format(acc))

if __name__ == "__main__":
    main()
