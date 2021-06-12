import argparse
import logging
import numpy as np
from scipy.io import loadmat

logging.basicConfig(level = logging.INFO)

def load_network_weights(weight_path):

	# Loads the pre-trained weight files stored in .mat format.
	#
	# Params:-
	# weight_path = Absolute path to the weight file.
	#
	# Returns:-
	# weights = Unloaded weights into a list from a .mat file.
	# net_dims = Dimension of the unpacked network.

    dat = loadmat(weight_path)
    weights = dat['weights']
    weights = weights[0, :].tolist()
    net_dims = [weights[0].shape[1]]

    for i in range(len(weights)):
        net_dims.append(weights[i].shape[0])

    return weights, net_dims

def split_weights_by_layer(weights, net_dims):

	# Splits the network weights layer by layer and returns the weight matrix for single layer.
	#
	# Params:-
	# weights = Unloaded weights stored in a list.
	# net_dims = Dimension of the unpacked network.
	#
	# Returns:-
	# split_w = The splitted weight matrix( in vector norm) for a single layer.
	# net_dims = Dimension of the layer( 1 in this case, kept for flexibility of size).

    num_weights = len(weights)
    logging.info('Size of weight network is {}'.format(num_weights))

    split_w = []
    split_net_dims = []

    counter = 1
    for i in range(0, num_weights, 1):

        if i > num_weights:
            i = num_weights

        split_w.append(weights[i : i+1])
        split_net_dims.append(net_dims[i : i+1])
        counter = counter + 1

    return split_w, split_net_dims

def calculate_lipschitz_bound(split_w, split_net_dims):

	# Calculates the spectral norm of each weight matrix per layer and returns the product.
	#
	# Params:-
	# split_w = Weights for the single incoming layer.
	# split_net_dims = Size of the split weight layer. Here it is not needed since it's always 1.
	#
	# Returns:-
	# Lip_prod = The product of the spectral norms of weight matrices for each layer.

	length = len(split_w)
	Lip_prod = 1 

	for i in range(length):
		spec_norm = np.linalg.norm(np.squeeze(split_w[i], axis=(0,)), 2)
		logging.info('Lipschitz bound for layer {} is {:.3f}'.format(i + 1, spec_norm))
		Lip_prod = Lip_prod * spec_norm

	return Lip_prod

def main():

	# Parses user arguments and displays the value of the Lipschitz constant.
	#
	# Params: None.
	#
	# Returns: Trivial Lipschitz upper bound for a fc network. 

	parser = argparse.ArgumentParser()
	parser.add_argument('--weightpath', type = str, help = 'Path to weights of trained NN.')
	args = parser.parse_args()

	loaded_weights, network_dim = load_network_weights(args.weightpath)
	single_weights, single_dims = split_weights_by_layer(loaded_weights, network_dim)

	L2 = calculate_lipschitz_bound(single_weights, single_dims)

	print('The trivial upper Lipschitz bound for the network is {:.3f}'.format(L2))


if __name__ == '__main__':
	main()



	
