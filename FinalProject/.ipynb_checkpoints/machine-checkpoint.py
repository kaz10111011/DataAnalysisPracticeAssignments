import numpy as np
import pandas as pd
from IPython.display import display, Markdown
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

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )



def machine_learning_analysis(df):

    df_clean = df[['Happiness','GDP','Social Support','Health','Freedom','Generosity']].dropna()
    GDP_group = (df_clean['GDP'] >= df_clean['GDP'].median()).map(
    {True: 0 , False: 1}
)
    x = df_clean[['Happiness','Health']].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

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
    train_scores = []
    test_scores = []
    for i in range(50):
        x_train, x_test, labels_train, labels_test = train_test_split(x, GDP_group, test_size=0.33)
        mlp    = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(50, 20), max_iter=2000 )
        mlp.fit(x_train, labels_train)
        # calculate the CRs for the training and test sets":
        labels_pred_train = mlp.predict(x_train)
        labels_pred_test  = mlp.predict(x_test)
        cr_train          = accuracy_score(labels_train, labels_pred_train)
        cr_test           = accuracy_score(labels_test, labels_pred_test)
        train_scores.append(cr_train)
        test_scores.append(cr_test)
    print( 'Classification rate (training) of the model using health =', np.mean(train_scores) )
    print( 'Classification rate (test) of the model using health   =', np.mean(test_scores ) )
    
    
    # plot the decision surface:

    fig, axes = plt.subplots(1, 2, figsize=(16,7))

    plt.sca(axes[0])
    plot_decision_surface(mlp, x_train, labels_train, colors=['b','r'])
    plt.plot(x_test[:,0], x_test[:,1], 'ko', label='Test set')
    plt.legend()




    s = df_clean[['Happiness','Social Support']].values
    s = scaler.fit_transform(s)
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
    s_train_scores = []
    s_test_scores = []
    for i in range(50):
        s_train, s_test, s_labels_train, s_labels_test = train_test_split(s, GDP_group, test_size=0.33)
        s_mlp    = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(50, 20), max_iter=2000 )
        s_mlp.fit(s_train, s_labels_train)
        
        
        
        # calculate the CRs for the training and test sets":
        s_labels_pred_train = s_mlp.predict(s_train)
        s_labels_pred_test  = s_mlp.predict(s_test)
        s_cr_train          = accuracy_score(s_labels_train, s_labels_pred_train)
        s_cr_test           = accuracy_score(s_labels_test, s_labels_pred_test)
        train_scores.append(s_cr_train)
        test_scores.append(s_cr_test)
    print( 'Classification rate (training) of the model using social support =',np.mean(s_cr_train) )
    print( 'Classification rate (test) of the model using social support    =',np.mean(s_cr_test) )
    
    # plot the decision surface:
    plt.sca(axes[1])
    s_plot_decision_surface(s_mlp, s_train, s_labels_train, colors=['b','r'])
    plt.plot(s_test[:,0], s_test[:,1], 'ko', label='Test set')
    plt.legend()
    display_title('Decision surface for classifying GDP groups using social support and happiness', pref='Fig', num=4, center = True )
    plt.show()


def evaluate_alpha(df):
    
    df_clean = df[['Happiness','GDP','Social Support','Health','Freedom','Generosity']].dropna()
    GDP_group = (df_clean['GDP'] >= df_clean['GDP'].median()).map(
    {True: 0 , False: 1}
)
    x = df_clean[['Happiness','Health']].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)


    import warnings
    warnings.filterwarnings('ignore')   # this will suppress warning from sklearn
    
    ALPHA   = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15 ]
    niter   = 20  # number of iterations for each ALPHA value
    
    np.random.seed(0)
    
    CR      = []
    for alpha in ALPHA:
        cr  = []
        for i in range(niter):
            x_train, x_test, labels_train, labels_test = train_test_split(x, GDP_group, test_size=0.33)
            mlp    = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(50, 20), max_iter=1000)
            mlp.fit(x_train, labels_train)
            labels_pred_test  = mlp.predict(x_test)
            cr_test           = accuracy_score(labels_test, labels_pred_test)
            cr.append( cr_test )
        CR.append( cr )
        
    CR      = np.array(CR)
    
    
    plt.figure()
    ax = plt.axes()
    h0 = ax.plot(ALPHA, CR, 'k.', ms=3)[0]
    h1 = ax.plot(ALPHA, CR.mean(axis=1), 'k-', lw=3, label='Average CR')[0]
    ax.legend([h0,h1], ['Single classifier', 'Average performance'])
    ax.set_xlabel('The coplexity of the model', size=16)
    ax.set_ylabel('Correct rate', size=16)
    display_title('The relation between the correct rate and the complexity of the model', pref='Fig', num=5, center = True )
    plt.show()
    
