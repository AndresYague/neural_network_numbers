import numpy as np

class NetworkObject(object):
    '''Class to create and manipulate neural networks
    the input is:

    -inpt: number of input neurons
    -hidden: list of length equal to hidden layers and each element
             is the number of neurons in each layer
    -output: number of output neurons
    -lbda: lambda parameter for regularization
    -fileName: if given, it will load appropriate network from file
    '''

    def __init__(self, inpt = 1, hidden = [], outpt = 1, lbda = 0.1,
                 fileName = None):

        # Load from filename if present
        self.fileName = fileName
        if fileName is not None:
            inpt, hidden, outpt = self.__parse_fileName(fileName)

        # Define sizes of layers
        self.inpt_num = inpt
        self.hidden_num = hidden
        self.outpt_num = outpt
        self.lbda = lbda

        # Now define theta and big deltas arrays
        self.theta_arrs = []
        self.big_deltas = []
        hid_len = len(self.hidden_num)

        # First theta
        num1 = self.inpt_num + 1
        if hid_len > 0:
            num2 = self.hidden_num[0]
            hid_len -= 1
        else:
            num2 = self.outpt_num

        sizRand = np.sqrt(6)/np.sqrt(num1 + num2)
        self.theta_arrs.append((np.random.random((num1, num2)) - 0.5) * sizRand)
        self.big_deltas.append(np.zeros((num1, num2)))

        # Subsequent
        ii = 1
        while hid_len > 0:
            num1 = num2 + 1
            num2 = self.hidden_num[ii]
            hid_len -= 1

            sizRand = np.sqrt(6)/np.sqrt(num1 + num2)
            self.theta_arrs.append((np.random.random((num1, num2)) - 0.5) * sizRand)
            self.big_deltas.append(np.zeros((num1, num2)))

        # Last one
        num1 = num2 + 1
        num2 = self.outpt_num

        sizRand = np.sqrt(6)/np.sqrt(num1 + num2)
        self.theta_arrs.append((np.random.random((num1, num2)) - 0.5) * sizRand)
        self.big_deltas.append(np.zeros((num1, num2)))

        if self.fileName is not None:
            self.__load_network()

    def __sigmoid(self, val):
        '''Calculate sigmoid function of val'''

        val = np.minimum(val, 100)
        val = np.maximum(val, -100)

        return 1/(1 + np.exp(-val))

    def __sigmoid_derv(self, val):
        '''Calculate sigmoid function derivative of val'''

        return self.__sigmoid(val) * (1 - self.__sigmoid(val))

    def get_cost(self, inpts, label_indices):
        '''Calculate cost function'''

        lab_arr = None
        cost = 0
        # Add the cost of every example
        for inpt, label in zip(inpts, label_indices):
            output = self.propagate(inpt)
            if lab_arr is None:
                lab_arr = np.zeros(np.shape(output))
            else:
                lab_arr *= 0

            # This is the y-array
            lab_arr[0][label] = 1

            # Adding the cost
            cost -= sum(sum(lab_arr * np.log(output) +
                            (1 - lab_arr) * np.log(1 - output)))

        # Regularize
        for theta in self.theta_arrs:
            cost += sum(sum(theta[1:]**2)) * self.lbda * 0.5

        cost /= len(inpts)
        return cost

    def calc_gradient(self, inpts, indices, batch_siz):
        '''Get gradients'''

        mm = batch_siz

        for ii in range(len(self.big_deltas)):
            self.big_deltas[ii] *= 0

        for inpt, label_index in zip(inpts, indices):
            # Propagate forwards
            activations, activations_no_bias = self.propagate(inpt, grad = True)

            # Now backwards
            jj = -1
            delt_fin = activations_no_bias[jj]
            delt_fin[0][label_index] -= 1

            # Save the deltas
            deltas = [delt_fin]

            ii = len(self.theta_arrs)
            while ii > 0:
                jj -= 1; ii -= 1

                # First add to big_deltas
                self.big_deltas[ii] += np.matmul(activations[jj].T, deltas[-1])

                # Now calculate new delta and add
                new_delt = np.matmul(deltas[-1], self.theta_arrs[ii][1:].T)
                new_delt *= activations_no_bias[jj] * (1 - activations_no_bias[jj])
                deltas.append(new_delt)

        # Add regularization and divide
        for ii in range(len(self.big_deltas)):
            self.big_deltas[ii][1:] += self.lbda * self.theta_arrs[ii][1:]
            self.big_deltas[ii] /= mm

    def train(self, train_inpts, label_indices, batch_siz = 10, alpha = 0.1,
              verbose = False, tol = 1e-4, low_cost = 0.3):
        '''Train network with this example'''

        ii = 0
        prevCost = None
        while True:
            # Define start and end indices
            init = ii * batch_siz
            end = min(init + batch_siz, len(train_inpts))

            if ii % 10 == 0:
                cost = self.get_cost(train_inpts[init:end],
                                     label_indices[init:end])

                if verbose:
                    print("The cost is {:.4f}".format(cost))

                if prevCost is not None:
                    diff = abs(prevCost - cost)/cost
                    if diff < tol:
                        return cost

                prevCost = cost
                if cost < low_cost:
                    return cost

            self.calc_gradient(train_inpts[init:end], label_indices[init:end],
                               batch_siz = batch_siz)

            # Gradient descent
            for jj in range(len(self.big_deltas)):
                self.theta_arrs[jj] -= self.big_deltas[jj] * alpha

            ii += 1
            if end == len(train_inpts):
                ii = 0
                if verbose:
                    print("\n--> Starting again\n")

    def propagate_indx_conf(self, inpt_given):
        '''Give back the index after propagation'''

        output = self.propagate(inpt_given = inpt_given)
        idxMax = np.argmax(output[0])

        # Calculate the confidence
        sum_outputs = sum(output[0])
        conf = output[0][idxMax]/sum_outputs

        return idxMax, conf

    def propagate(self, inpt_given, grad = False):
        '''Propagate network with given input'''

        # To add the one
        one = np.ones((1, 1))

        # Reshape so all are proper vectors
        no_bias = np.reshape(inpt_given, (1, len(inpt_given)))
        curr_layer = np.append(one, no_bias, axis = 1)

        # If grad, add this one
        if grad:
            activations_no_bias = [no_bias]
            activations = [curr_layer]

        # Now propagate and add to activations
        for theta in self.theta_arrs:
            next_layer = self.__sigmoid(np.matmul(curr_layer, theta))
            curr_layer = np.append(one, next_layer, axis = 1)

            if grad:
                activations_no_bias.append(next_layer)
                activations.append(curr_layer)

        if grad:
            return activations, activations_no_bias
        else:
            return next_layer

    def get_thetas(self):
        '''Return the thetas'''

        return self.theta_arrs

    def set_thetas(self, thetas):
        '''Set the thetas to the given value'''

        self.theta_arrs = thetas

    def get_gradient(self):
        '''Return the gradient'''

        return self.big_deltas

    def __get_fileName(self, cost):
        '''Create a consistent filename for the network'''

        self.fileName = "saved_nn_{}_".format(self.inpt_num)
        for hid in self.hidden_num:
            self.fileName += "{}_".format(hid)
        self.fileName = self.fileName[:-1] + "_{}".format(self.outpt_num)
        self.fileName += "_cost_{:.3f}.npy".format(cost)

    def save_network(self, cost):
        '''Save this network in file'''

        if self.fileName is None:
            self.__get_fileName(cost)

        # Trained, save thetas
        thetas = self.get_thetas()
        with open(self.fileName, "wb") as fwrite:
            for theta in thetas:
                np.save(fwrite, theta)

    def __load_network(self):
        '''Load network from file'''

        thetas = []
        nLayers = 2 + len(self.hidden_num)
        with open(self.fileName, "rb") as fread:
            for ii in range(nLayers - 1):
                thetas.append(np.load(fread))

        self.set_thetas(thetas)

    def __parse_fileName(self, fileName):
        '''Parse fileName to get parameters'''

        break_fileName = fileName.split("_")

        # Input and output
        inpt = int(break_fileName[2])
        outpt = int(break_fileName[-3])

        # Search for hidden layers
        hidden = list(map(lambda x: int(x), break_fileName[3:-3]))

        return inpt, hidden, outpt
