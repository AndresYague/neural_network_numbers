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

    # Get filename
    if len(sys.argv) > 1:
        fileName = sys.argv[1]
    else:
        print("Use: python3 {} <network_params>".format(sys.argv[0]))
        return 1

    # Read the test data
    print("Loading test data...")
    file_lab = "ML_numbers/t10k-labels-idx1-ubyte"
    file_img = "ML_numbers/t10k-images-idx3-ubyte"
    test_lab = readMNIST(file_lab, typ = "label")
    test_img, siz1, siz2 = readMNIST(file_img, typ = "image")
    print("Loaded")

    # Load the network
    numbers_nn = NetworkObject(fileName = fileName)

    # Check network in test set:
    print("Checking accuracy with test set")
    correct_cases = 0; conf_incorrect = 0
    conf_cases = 0
    conf_level = 0.7
    labels, confidence = numbers_nn.propagate_indx_conf(test_img)
    for ii in range(len(test_lab)):
        lab_net = labels[ii]
        conf = confidence[ii]
        lab = test_lab[ii]

        # Check correct rate
        if lab_net == lab:
            correct_cases += 1
        elif conf > conf_level:
            conf_incorrect += 1

        # Check confidence
        if conf > conf_level:
            conf_cases += 1

    tot_cases = len(test_lab)
    acc = correct_cases/tot_cases * 100
    inc_acc = conf_incorrect/tot_cases * 100
    high_conf = conf_cases/tot_cases * 100

    print("Overall correct rate = {:.2f}%".format(acc))
    print("Proportion of highly confident cases = {:.2f}%".format(high_conf))
    print("Confidently incorrect cases = {:.2f}%".format(inc_acc))

    print("Showing random examples")
    for ii in range(10):
        jj = np.random.choice(range(len(test_lab)))
        idxMax, conf = numbers_nn.propagate_indx_conf(test_img[jj])

        # Print network output and plot number
        print()
        print(f"Network thinks this is a {idxMax[0]}", end = "")
        print(" with a confidence of {:.2f}%".format(conf[0] * 100))
        print(f"The label is {test_lab[jj]}")
        show_number(test_img[jj], siz1, siz2)

if __name__ == "__main__":
    main()
