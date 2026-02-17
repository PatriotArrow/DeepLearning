import numpy as np
import pandas as pd

#this is the same code as in the logisticRegressor.ipynb, but as a .py file so that its easier to reuse in task 2
class LogisticRegressor:
    def __init__(self, alpha=0.01, max_iters=100000, threshold=1e-6, N=1000):
        self.alpha = alpha
        self.max_iters = max_iters
        self.threshold = threshold
        self.N = N
        self.trained = False #to be used in predict

    def sigmoid(self, t):
        return 1/(1 + np.exp(-t))
    
    def loss_function(self, y, y_cap):
        epsilon = np.finfo(float).eps
        y_cap = np.clip(y_cap, epsilon, 1 - epsilon)
        return -(y * np.log(y_cap) + (1 - y) * np.log(1 - y_cap))
    
    def fit(self, X, Y):

        #X = X.iloc[:, :-1].values
        #Y = Y.iloc[:, -1].values.reshape(-1, 1)
        m = X.shape[0]
        n = X.shape[1]

        w = np.array(X.shape[1] * [.5]).reshape(-1, 1)
        b = 0.5

        stopping = False; J_running = 0; J_prev = 0; iteration = 0; acc = 0
        

        while not stopping:
            i = np.random.randint(0,m)
            x = X[i].reshape(-1,1)
            y = Y[i,0]

            #forward propagation
            z = (w.T @ x + b).item()
            y_hat = self.sigmoid(z)
            #calculating J_current from y, y_hat
            J_current = float(self.loss_function(y, y_hat))
            #gradient descent 
            dz = (y_hat - y)

            #looping over j elements of w to calculate partial derivative of the loss with respect to each wt
            delta_w = np.zeros_like(w)
            for j in range(n):
                delta_w[j,0] = x[j,0] * dz

            #calculating delta_b
            delta_b = dz
            #looping again over j elemets of w: w_j -= alpha * partial derivative
            for j in range(n) :
                w[j,0] = w[j, 0] - self.alpha* delta_w[j, 0]

            b -= self.alpha*delta_b
            #we need to check the stopping criteria 
            iteration +=1
            J_running += J_current

            if iteration > self.max_iters:
                #failing to converge
                stopping = True 

            if (iteration % self.N ) == 0:
                #comparing J_running with J_running_prev
                #if its less than threshold we will put stopping = true 
                if abs(J_running - J_prev) < self.threshold :
                    stopping = True 

                J_prev = J_running
                J_running = 0
                #predictions and training accuracy
                Y_prob = self.sigmoid(X @ w + b)
                Y_pred = (Y_prob >= 0.5).astype(int)
                acc = np.mean(Y_pred == Y)
                print("Training accuracy:", float(acc))
        #set weights here
        self.w = w
        self.b = b
        self.accuracy = acc
        self.trained = True

    def predict(self,x):
        if (self.trained == False):
            raise ValueError("Not trained uyet!")
        
        y_prob = self.sigmoid(x @ self.w + self.b)
        Y_pred = (y_prob >= 0.5).astype(int)
        return Y_pred
    
