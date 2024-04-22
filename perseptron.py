import joblib
import os
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



class Perseptron:
    def __init__(self,eta:float=None,epochs:int=None):
        self.weights = np.random.randn(3)*1e-4
        training = (eta is not None) and (epochs is not None)
        if training:
            print("initial weights", self.weights)
        self.eta = eta
        self.epochs = epochs

    def _z_outcome(self,inputs,weights):
        return np.dot(inputs,weights)

    def activation_function(self,z):
        return np.where(z > 0,1,0)
    
    def fit(self,X,Y):
        self.X = X
        self.Y= Y
        X_with_bias= np.c_[self.X,-np.ones((len(self.X),1))]
        print("X_with_bias",X_with_bias) 
        for epoch in range(self.epochs):
            print("for epoch",epoch)
            z = self._z_outcome(X_with_bias,self.weights)
            Y_hat = self.activation_function(z)
            print("predicted value after forward pass",Y_hat)
            self.error= self.Y - Y_hat
            self.weights = self.weights+self.eta*np.dot(X_with_bias.T,self.error)
            print(f"updated weights after epoch {epoch+1}/{self.epochs}:\n {self.weights}")
            print("##"*10)

    def predict(self,X):
        X_with_bias= np.c_[self.X,-np.ones((len(self.X),1))]
        z = self._z_outcome(X_with_bias,self.weights)
        return self.activation_function(z)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print("total loss", total_loss)
        return(total_loss)
    
    def _create_dir_return_path(self,model_dir,filename):
        os.makedirs(model_dir,exist_ok=True)
        return os.path.join(model_dir,filename)
    
    def save(self,filename,model_dir = None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir,filename)
            joblib.dump(self,model_file_path)
        else:
            model_file_path = self._create_dir_return_path("model",filename)
            joblib.dump(self,model_file_path)
    
    def load(self,filepath):
        return joblib.load(filepath)
    
#implementation of Andgate
AND = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,0,0,1]
}

df_and = pd.DataFrame(AND)
print(df_and)
def prepare_data(df,target_col = "y"):
    X = df.drop(target_col,axis = 1)
    Y = df[target_col]
    return X,Y

X,Y = prepare_data(df_and)
ETA = 0.1
EPOCHS = 10
model_and = Perseptron(eta = ETA, epochs=EPOCHS)
model_and.fit(X,Y)
_ = model_and.total_loss()

model_and.save(filename="and.model")


reload_and = Perseptron().load(filepath="model/and.model")


print("reloaded", reload_and.predict(X=[[1,0]]))
# OR gate
OR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,1]
}

df_or = pd.DataFrame(OR)
X,Y = prepare_data(df_or)
ETA=0.1
EPOCHS = 10
model_or = Perseptron(eta = ETA, epochs=EPOCHS)
model_or.fit(X,Y)
_ = model_or.total_loss()
print(_)

#XOR gate
XOR = {
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y":[0,1,1,0]
}

df_xor = pd.DataFrame(XOR)
X,Y = prepare_data(df_xor)
ETA=0.1
EPOCHS = 10
model_xor = Perseptron(eta = ETA, epochs=EPOCHS)
model_xor.fit(X,Y)
_ = model_xor.total_loss()
print(_)

def save_plot(df,model,filename = "plot.png", plot_dir = "plots"):
    def _create_base_plot(df):
        df.plot(kind = "scatter",x = "x1",y = "x2",c = "y",s = 100,cmap = "coolwarm")
        plt.axhline(y =0,color="black",linestyle="--",linewidth=1)
        plt.axvline(x=0,color="black",linestyle="--",linewidth=1)
        figure=plt.gcf() 
        figure.set_size_inches(10,8)

    def _plot_decision_region(X,Y,classifier,resolution=0.02):
        colors=("cyan","lightgreen")
        cmap=ListedColormap(colors)
        X=X.values
        x1=X[:,0]
        x2=X[:,1]
        x1_min,x1_max=x1.min()-1,x1.max()+1
        x2_min,x2_max=x2.min()-1,x2.max()+1
        # xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
        xx1,xx2 = np.meshgrid(
            np.arange(x1_min,x1_max, resolution),
            np.arange(x2_min,x2_max,resolution)
        )
        print(xx1, "this is xx1")
        y_hat=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        xx1 = xx1.ravel().T
        print("this is y hat, ", y_hat)
        y_hat=y_hat.reshape(xx1.shape)
        plt.contourf(xx1,xx2,y_hat,alpha=0.3,cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())
        plt.plot()


    X,Y = prepare_data(df)
    _create_base_plot(df)
    _plot_decision_region(X,Y,model)
    os.makedirs(plot_dir,exist_ok=True)
    path_plot = os.path.join(plot_dir,filename)
    plt.savefig(path_plot)
save_plot(df_or, model_or,filename="or.png")


