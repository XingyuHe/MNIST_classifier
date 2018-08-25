import numpy as np
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import h5py
import time



def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def load_dog_data():
    import glob

    f = h5py.File('mnist_datasets.hdf5', 'r')
    train_set_x = f['train_dataset_x'][:]
    train_set_y = f['train_dataset_y'][:]
    # test_set_x = f['test_dataset_x'][:]
    # test_set_y = f['test_dataset_y'][:]

    classes = []
    for filename in glob.glob('/Users/xhe/Desktop/MNIST_classifier/trainingSet/*'):
        classes.append(filename[53:len(filename)])

    return train_set_x.T, train_set_y.T, classes

def save_mnist_data():

    import glob
    import pickle
    import scipy
    from scipy import ndimage

    classes = []

    set_choices = ['train', 'test', 'skip']
    set_weight = [0.5, 0.1, 0.4]
    num_px = 28

    # create dataset for training and testing
    dataset_file = h5py.File('mnist_datasets.hdf5', 'a')
    train_dataset_x = dataset_file.create_dataset('train_dataset_x', (1, num_px * num_px * 1) , chunks = True, maxshape = (None, None))
    test_dataset_y = dataset_file.create_dataset('test_dataset_y', (1, 10), chunks = True, maxshape = (None, None))
    test_dataset_x = dataset_file.create_dataset('test_dataset_x', (1, num_px * num_px * 1) , chunks = True, maxshape = (None, None))
    train_dataset_y = dataset_file.create_dataset('train_dataset_y', (1, 10), chunks = True, maxshape = (None, None))
    
    j = 0

    for filename in glob.glob('/Users/xhe/Desktop/MNIST_classifier/trainingSet/*'):
        number_name = filename[48:len(filename)]
        print(filename)

        train_set_x = np.empty([1, num_px * num_px * 1])
        train_set_y = np.empty([1, 1])
        test_set_x = np.empty([1, num_px * num_px* 1])
        test_set_y = np.empty([1, 1])



        classes.append(number_name)

        i = 0


        for image_name in glob.glob('%s/*'%filename):
            print(dataset_file['train_dataset_x'].shape, dataset_file['train_dataset_y'].shape, dataset_file['test_dataset_x'].shape, dataset_file['test_dataset_y'].shape)
            print(number_name + " : ", i)
            image_orig = np.array(ndimage.imread(image_name, flatten=False))
            # print(image_orig)
            # print(image_orig.shape)
            my_image = image_orig.reshape((num_px*num_px*1,1)).T
            my_image = my_image/255.

            append_pos = -1

            if j == 0 and i == 0:
                append_pos = 0

            train_set_x = my_image
            train_set_y = np.zeros([1, 10])
            train_set_y[:,j] = 1

            dataset_file['train_dataset_x'].resize(dataset_file['train_dataset_x'].shape[0] - append_pos, axis = 0)
            dataset_file['train_dataset_x'][append_pos:] = train_set_x
            dataset_file['train_dataset_y'].resize(dataset_file['train_dataset_y'].shape[0] - append_pos, axis = 0)
            dataset_file['train_dataset_y'][append_pos:] = train_set_y
                       
            i += 1

        j += 1



def load_data():

    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters     


def initialize_parameters_deep(layer_dims):

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):

    print("W shape: ", W.shape)
    print("A shape: ", A.shape)
    print("b shape: ", b.shape)
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        time_1 = time.time()
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        time_onelayer = time.time() - time_1
        print("time for layer %i" %l,time_onelayer)
        with open("time_stats", 'a+') as f:
            f.write("time for layer %i: %f \n" %(l,time_onelayer))
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    print(AL.shape)
    assert(AL.shape == (10,X.shape[1]))
            
    return AL, caches

def compute_cost(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    time_1 = time.time()
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = cost[0]
    time_cost = time.time() - time_1
    print("time to compute cost: %f" %time_cost)
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    # assert(cost.shape == ())
    with open("time_stats", 'a+') as f:
        f.write("time to compute cost: %f \n" %time_cost)
    
    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        time_1 = time.time()
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        time_onelayer = time.time() - time_1
        print("time for layer %i backward: %f" %(l, time_onelayer))
        with open("time_stats", 'a+') as f:
            f.write("time for layer %i backward: %f \n" %(l, time_onelayer))

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    print("number of layers: %i"%n)
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    print("probas shape: ", probas.shape)
    print("y shape: ", y.shape)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))

    correct_predictions = np.all(probas == y, axis=0)
    print("correct_predictions: ",np.sum(correct_predictions))
    print(probas.T[1])
    print(y.T[1])

    y = y.T
    probas = probas.T
    for i in range(probas.shape[0]):
        for j in range(probas.shape[1]):
            if probas[i][j] == y[i][j]:
                correct_predictions += 1
    # print("Accuracy: %i out of %i" %(correct_predictions, m))
        
    return p, probas

def print_mislabeled_images(classes, X, y, p):

    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
