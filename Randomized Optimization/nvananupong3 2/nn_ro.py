import numpy as np
import mlrose as mlr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

#takes csv and makes it into numpy array
csv = np.genfromtxt ('newAdult.csv', delimiter=",", dtype=int)
instances = csv[1:,0:-1]
labels = csv[1:,-1]



#makes training and test data
X_train, X_test, y_train, y_test = train_test_split(instances, labels, \
                                                    test_size = 0.3)

#test = X_train[:len(X_train)//10, :]
#print(test)

splitlen = len(X_train) // 10
x_splits = [X_train[:splitlen, :], X_train[:2*splitlen, :], X_train[:3*splitlen, :], \
            X_train[:4*splitlen, :], X_train[:5*splitlen, :], X_train[:6*splitlen, :], \
            X_train[:7*splitlen, :], X_train[:8*splitlen, :], X_train[:9*splitlen, :], X_train] 

y_splits = [y_train[:splitlen], y_train[:2*splitlen], y_train[:3*splitlen],\
            y_train[:4*splitlen], y_train[:5*splitlen], y_train[:6*splitlen], \
            y_train[:7*splitlen], y_train[:8*splitlen], y_train[:9*splitlen], y_train] 


# One hot encode target values
#one_hot = OneHotEncoder(categories='auto')

#y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
#y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()



#RHC to get weights

#out = open("adult_results2.txt", "w")

with open('results/rhc_rate.txt', "w") as out:
    out.writelines("rate, train_error, test_error\n")
    print(("Random Hill Climb Changing Rates: (rate, train error, test error)\n"))
    try:
        for rate in range(50,1001,50):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'random_hill_climb', max_iters = 500, \
                                             bias = True, is_classifier = True, learning_rate = rate / 10000, \
                                             early_stopping = False, max_attempts = 100)
            nn_model1.fit(X_train, y_train)
            
            # Predict labels for train set and assess accuracy
            y_train_pred = nn_model1.predict(X_train)
            #y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
            #print(rate, " Training accuracy: ", y_train_accuracy)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            #y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            #print(rate, " Test accuracy: ", y_test_accuracy)
            out.writelines(str(rate) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Random Hill Climb Changing Rates")

with open('results/sa_rate.txt', 'w') as out:
    out.writelines("rate, train_error, test_error\n")
    print("Simulated Annealing Changing Rates: (rate, train error, test error)\n")
    try:
        for rate in range(50,1001,50):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'simulated_annealing', max_iters = 500, \
                                             bias = True, is_classifier = True, learning_rate = rate / 10000, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(rate) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Simulated Annealing Changing Rates")

with open('results/ga_rate.txt', 'w') as out:
    out.writelines("rate, train_error, test_error\n")
    print("Genetic Alg Changing Rates: (rate, train error, test error)\n")
    try:
        for rate in range(50,1001,50):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'genetic_alg', max_iters = 50, \
                                             bias = True, is_classifier = True, learning_rate = rate / 10000, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(rate) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n") 
    except:
        print("Error in Genetic Alg Changing Rates")

with open('results/rhc_iterations2.txt', 'w') as out:
    out.writelines("iterations, train error, test error\n")
    print("Random Hill Climb Changing Iterations: (iterations, train error, test error)\n")
    try:
        for iters in range(50,1001, 50):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'random_hill_climb', max_iters = iters, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(iters) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Random Hill Climb Changing Iterations")

with open('results/sa_iterations2.txt', 'w') as out:  
    out.writelines("iterations, train error, test error\n")
    print("Simulated Annealing Changing Iterations: (iterations, train error, test error)\n")
    try:
        for iters in range(50,1001, 50):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'simulated_annealing', max_iters = iters, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(iters) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Simulated Annealing Changing Iterations")
        
with open('results/ga_iterations2.txt', 'w') as out:  
    try:
        out.writelines("iterations, train error, test error\n")
        print("Genetic Alg Changing Iterations: (iterations, train error, test error)\n")
        for iters in range(5,101, 5):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'genetic_alg', max_iters = iters, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            #Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(iters) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Genetic Alg Changing Iterations")
        

with open('results/ga_popsize.txt', 'w') as out:  
    try:
        out.writelines("iterations, train error, test error\n")
        print("Genetic Alg Changing Iterations: (iterations, train error, test error)\n")
        for pop in [5,10,20,30,40,50,75,100,150,200,250,300,400,500]:
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'genetic_alg', max_iters = 100, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = True, clip_max = 5, pop_size=pop, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(iters) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Genetic Alg Changing Population Size")
        
with open('results/ga_mutprob.txt', 'w') as out:  
    try:
        out.writelines("iterations, train error, test error\n")
        print("Genetic Alg Changing Iterations: (iterations, train error, test error)\n")
        for prob in [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,0.5]:
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'genetic_alg', max_iters = 100, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = True, clip_max = 5, pop_size=100, mutation_prob=prob, max_attempts = 100)
        
            nn_model1.fit(X_train, y_train)
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(iters) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Genetic Alg Changing Mutation Chance")
        
print("Testing size of training test")
with open('results/rhc_setsize.txt', 'w') as out:  
    try:
        out.writelines("setsize, train error, test error\n")
        print("Random Hill Climb Changing Size: (set size, train error, test error)\n")
        for size in range(len(x_splits)):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'random_hill_climb', max_iters = 500, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(x_splits[size], y_splits[size])
            y_train_pred = nn_model1.predict(x_splits[size])
            y_train_accuracy = accuracy_score(y_splits[size], y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(size) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Random Hill Climb Changing Size")
        
with open('results/sa_setsize.txt', 'w') as out:  
    try:
        out.writelines("setsize, train error, test error\n")
        print("Simmulated Annealing Changing Size: (set size, train error, test error)\n")
        for size in range(len(x_splits)):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'simulated_annealing', max_iters = 500, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(x_splits[size], y_splits[size])
            y_train_pred = nn_model1.predict(x_splits[size])
            y_train_accuracy = accuracy_score(y_splits[size], y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(size) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Simmulated Annealing Changing Size")
        
with open('results/ga_setsize.txt', 'w') as out:  
    try:
        out.writelines("setsize, train error, test error\n")
        print("Random Hill Climb Changing Size: (set size, train error, test error)\n")
        for size in range(len(x_splits)):
            nn_model1 = mlr.NeuralNetwork(hidden_nodes = [7,7,7], activation = 'relu', \
                                             algorithm = 'genetic_alg', max_iters = 50, \
                                             bias = True, is_classifier = True, learning_rate = 0.001, \
                                             early_stopping = False, clip_max = 5, max_attempts = 100)
        
            nn_model1.fit(x_splits[size], y_splits[size])
            y_train_pred = nn_model1.predict(x_splits[size])
            y_train_accuracy = accuracy_score(y_splits[size], y_train_pred)
        
            
            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)
            out.writelines(str(size) + ", " + str(y_train_accuracy) + ", " + str(y_test_accuracy) + "\n")
    except:
        print("Error in Genetic Alg Changing Size")


print("Finished")