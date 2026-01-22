import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import descriptive as de
import parse_data as pa
import classical as cl


def Multiple_regression_analysis(df):
    df_clean = df[['Happiness','GDP','Social Support','Health','Freedom','Generosity']].dropna()
    y_clean = df_clean['Happiness']
    x_clean = df_clean[['GDP','Social Support','Health','Freedom','Generosity']]
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x_clean)
    x_train, x_test, y_train, y_test = train_test_split(
    x_std, y_clean, test_size=0.2, random_state=42
)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print("R2:", r2)
    coef = pd.Series(model.coef_, index=x_clean.columns)
    print(coef)
    
    coef.sort_values().plot(kind='barh', color='skyblue')
    plt.xlabel('Coefficients of the multiple regression analysis')
    plt.show()


def machine_learning_analysis(df):

    df_clean = df[['Happiness','GDP','Social Support','Health','Freedom','Generosity']].dropna()
    GDP_group = (df_clean['GDP'] >= df_clean['GDP'].median()).map(
    {True: 0 , False: 1}
)
    x = df_clean[['Happiness','Health']].values

    x_train, x_test, labels_train, labels_test = train_test_split(x, GDP_group, test_size=0.33, random_state=1)
    
    def plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            label_name = 'High GDP' if label == 0 else 'Low GDP'
            xx   = x[labels==label]
            ax.scatter( xx[:,0], xx[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=label_name )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Health')
        ax.set_ylabel('Happiness')
        ax.legend()

    # create and train a classifier:
    mlp    = MLPClassifier(solver='lbfgs', alpha=2, hidden_layer_sizes=(50, 20), max_iter=500, random_state=0)
    mlp.fit(x_train, labels_train)
    
    
    # calculate the CRs for the training and test sets":
    labels_pred_train = mlp.predict(x_train)
    labels_pred_test  = mlp.predict(x_test)
    cr_train          = accuracy_score(labels_train, labels_pred_train)
    cr_test           = accuracy_score(labels_test, labels_pred_test)
    print( f'Classification rate (training) of the model using health = {cr_train}' )
    print( f'Classification rate (test) of the model using health   = {cr_test}' )
    
    
    # plot the decision surface:

    fig, axes = plt.subplots(1, 2, figsize=(16,7))

    plt.sca(axes[0])
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test[:,0], x_test[:,1], 'ko', label='Test set')
    plt.legend()




    s = df_clean[['Happiness','Social Support']].values
    s_train, s_test, s_labels_train, s_labels_test = train_test_split(s, GDP_group, test_size=0.33, random_state=50)
    
    def s_plot_decision_surface(classifier, x, labels, ax=None, colors=None, n=50, alpha=0.3, marker_size=200, marker_alpha=0.9):
        stratify  = GDP_group
        nlabels   = np.unique( labels ).size
        colors    = plt.cm.viridis( np.linspace(0,1,nlabels) )  if (colors is None) else colors
        ax        = plt.gca() if (ax is None) else ax
        xmin,xmax = x.min(axis=0), x.max(axis=0)
        Xp,Yp     = np.meshgrid( np.linspace(xmin[0],xmax[0],n) , np.linspace(xmin[1],xmax[1],n) )
        xp        = np.vstack( [Xp.flatten(), Yp.flatten()] ).T
        labelsp   = classifier.predict(xp)
        Labelsp   = np.reshape(labelsp, Xp.shape)
        cmap      = ListedColormap(colors)
        for i,label in enumerate( np.unique(labels) ):
            label_name =  'High GDP' if label == 0 else 'Low GDP'
            xx   = x[labels==label]
            ax.scatter( xx[:,0], xx[:,1], color=colors[i], s=marker_size, alpha=marker_alpha, label=label_name )
        plt.pcolormesh(Xp, Yp, Labelsp, cmap=cmap, alpha=alpha)
        ax.set_xlabel('Social Support')
        ax.set_ylabel('Happiness')
        ax.legend()
        
    # create and train a classifier:
    s_mlp    = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(50, 20), max_iter=500, random_state=0)
    s_mlp.fit(s_train, s_labels_train)
        
        
    # calculate the CRs for the training and test sets":
    s_labels_pred_train = s_mlp.predict(s_train)
    s_labels_pred_test  = s_mlp.predict(s_test)
    s_cr_train          = accuracy_score(s_labels_train, s_labels_pred_train)
    s_cr_test           = accuracy_score(s_labels_test, s_labels_pred_test)
    print( f'Classification rate (training) of the model using social support = {s_cr_train}' )
    print( f'Classification rate (test) of the model using social support    = {s_cr_test}' )
    
    # plot the decision surface:
    plt.sca(axes[1])
    s_plot_decision_surface(s_mlp, s_train, s_labels_train, colors=['b','r'])
    plt.plot(s_test[:,0], s_test[:,1], 'ko', label='Test set')
    plt.legend()
    plt.show()
    
    
    
    
