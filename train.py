import numpy as np
import h5py
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from main import *
import pickle

time_1 = time.time()
train_x_orig, train_y, classes = load_dog_data()
time_2 = time.time()
time_load_data = time_2 - time_1
print(time_load_data)
with open("time_stats", "w") as f:
    f.write("time to load data: %f\n" %time_load_data)

# index = 10
# plt.imshow(train_x_orig[index])
# print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

train_x = train_x_orig
# test_x = test_x_orig

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[0]
# m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 1)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# train_x = train_x_flatten/255.
# test_x = test_x_flatten/255.


print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))

n_x = train_x.shape[0]     
n_h = 7
n_y = len(classes)
layers_dims = (n_x, n_h, 1)



def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    np.random.seed(1)
    grads = {}
    costs = []                              
    m = X.shape[1]                           
    (n_x, n_h, n_y) = layers_dims
           
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
        
    for i in range(0, num_iterations):
               
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
                                
        cost = compute_cost(A2, Y)
                
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
                            
        parameters = update_parameters(parameters, grads, learning_rate)
                
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
               
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost) 
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

layers_dims = [n_x,300,n_y ]


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):

    costs = []                         
    
    parameters = initialize_parameters_deep(layers_dims)
    
    
    for i in range(0, num_iterations):
        time_1 = time.time()
        

        AL, caches = L_model_forward(X, parameters)
        # print(AL)
        
        cost = compute_cost(AL, Y)        
        
        grads = L_model_backward(AL, Y, caches)
        # print(grads['dW1'])
        
        parameters = update_parameters(parameters, grads, learning_rate)


        with open("dog_model", 'w') as dog_model:
            pickle.dump(parameters, dog_model)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        
        print("cost for iteration %i" %i,cost)
        time_iter = time.time() - time_1
        print("time for iteration %i: %i \n" %(i, time_iter))
        with open("time_stats", "a+") as f:
            f.write("time for iteration %i: %i \n" %(i, time_iter))



    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters =  L_layer_model(train_x, train_y, layers_dims, num_iterations = 3000)


