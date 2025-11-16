from IPython.display import display,Markdown 
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd


def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )


import parse_data as pa



def central(x, print_output=True):
    x      = x[~np.isnan(x)]
    x0     = np.mean( x )
    x1     = np.median( x )
    x2     = stats.mode( x, keepdims=True ).mode[0]
    return x0, x1, x2


def dispersion(x, print_output=True):
    x  = x[~np.isnan(x)]
    y0 = np.std( x ) 
    y1 = np.min( x ) 
    y2 = np.max( x ) 
    y3 = y2 - y1      
    y4 = np.percentile( x, 25 )
    y5 = np.percentile( x, 75 )
    y6 = y5 - y4 
    return y0,y1,y2,y3,y4,y5,y6


def display_central_tendency_table(num=1):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    pa.df_central = pa.df.apply(lambda x: central(x), axis=0)
    round_dict = {'Happiness': 3, 'GDP': 3, 'Social Support': 3, 'Health': 3, 'Freedom':3}
    pa.df_central = pa.df_central.round( round_dict )
    row_labels = 'mean', 'median', 'mode'
    pa.df_central.index = row_labels
    display(pa.df_central)

def display_dispersion_table(num=1):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    round_dict            = {'Happiness': 3, 'GDP': 3, 'Social Support': 3, 'Health': 3, 'Freedom':3}
    pa.df_dispersion         = pa.df.apply(lambda x: dispersion(x), axis=0).round( round_dict )
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    pa.df_dispersion.index   = row_labels_dispersion
    display(pa.df_dispersion)



y    = pa.df['Happiness']
GDP  = pa.df['GDP']
SS   = pa.df['Social Support']
HE   = pa.df['Health']
FR = pa.df['Freedom']
GE = pa.df['Generosity']


SS1 = np.around(2*SS, 1)




def corrcoeff(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    r = np.corrcoef(x, y)[0,1]
    return r

def plot_regression_line(ax, x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    a,b   = np.polyfit(x, y, deg=1)
    x0,x1 = min(x), max(x)
    y0,y1 = a*x0 + b, a*x1 + b
    ax.plot([x0,x1], [y0,y1], **kwargs)




def plot_descriptive():
    
    fig,axs = plt.subplots( 2, 3, figsize=(8,6), tight_layout=True )
    ivs     = [GDP, SS, HE, FR, GE]
    colors  = 'b', 'r', 'g','y' ,'orange'
    for ax,x,c in zip(axs.ravel(), ivs, colors):
        ax.scatter( x, y, alpha=0.5, color=c , s = 10)
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        r   = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    xlabels = 'GDP', 'Social Support', 'Health','Freedom' ,'Generosity'
    [ax.set_xlabel(s) for ax,s in zip(axs.ravel(),xlabels)]
    axs[0,1].set_xticks([0, 1, 2])
    [ax.set_ylabel('Happiness') for ax in axs[:,0]]
    [ax.set_yticklabels([])  for ax in axs[:,1]]
    [ax.set_yticklabels([])  for ax in axs[:,2]]

    ax       = axs[1,2]
    i_low    = y <= 5
    i_high   = y > 5
    fcolors  = 'm', 'c'
    labels   = 'Low-happiness', 'High-happiness'
    q_groups = [[3,4,5], [6,7,8]]
    ylocs    = 0.3, 0.7
    for i,c,s,qs,yloc in zip([i_low, i_high], fcolors, labels, q_groups, ylocs):
        ax.scatter( SS[i], y[i], alpha=0.5, color=c, facecolor=c, label=s )
        plot_regression_line(ax, SS[i], y[i], color=c, ls='-', lw=2)
        [ax.plot(SS[i].mean(), q, 'o', color=c, mfc='w', ms=10)  for q in qs]
        r   = corrcoeff(SS[i], y[i])
        ax.text(0.7, yloc, f'r = {r:.3f}', color=c, transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

    ax.legend()
    ax.set_xlabel('Social Support')

    panel_labels = 'a', 'b', 'c', 'd','e','f'
    [ax.text(0.02, 0.92, f'({s})', size=12, transform=ax.transAxes)  for ax,s in zip(axs.ravel(), panel_labels)]
    plt.show()
    
    display_title('Correlations amongst main variables.', pref='Figure', num=1)

