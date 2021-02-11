This is a simple neural network library. An example of use is:

To create a neural network:

new_network = NetworkObject(inpt = nn1, hidden = [nn2, nn3, ...], outpt = nn4,
                            lbda = lambda)

Where nn1 is the size of the input layer, nn4 is the size of the output layer
and nn2, nn3, ... are the sizes of the hidden layers. hidden also accepts an
empty list for no hidden layers.

One can save a network at any moment by using the method save_network(), which
accepts cost as an optional argument in case that it is relevant information.
If not given, cost will be set to zero.

To load a network:

load_network = NetworkObject(fileName = file)

where file is the network parameters saved by save_network().

To mini-batch train the network one can use the method train, which accepts
the batch size, the learning rate, a verbose flag, and the treshold for the
cost to stop training.
