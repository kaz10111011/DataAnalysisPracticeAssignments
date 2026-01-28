import numpy as np
from IPython.display import display, Markdown
from scipy import stats
from matplotlib import pyplot as plt
import parse_data as pa
import descriptive as de
de.define_variables(pa.df)
y, GDP, SS, HE, FR, GE, SS1 = de.define_variables(pa.df)

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s    = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref=='Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display( Markdown(s) )

    

def Regression(y, GDP, SS, HE, FR, GE, SS1):
    mask = (~np.isnan(y) &
        ~np.isnan(GDP) &
        ~np.isnan(SS) &
        ~np.isnan(HE) &
        ~np.isnan(FR) &
        ~np.isnan(GE) )


    
    SSresults = stats.linregress(SS[mask], y[mask])
    SSp = SSresults.pvalue

  
    HEresults = stats.linregress(HE[mask], y[mask])
    HEp = HEresults.pvalue

    
    FRresults = stats.linregress(FR[mask], y[mask])
    FRp = FRresults.pvalue
 
    
    GEresults = stats.linregress(GE[mask], y[mask])
    GEp = GEresults.pvalue


    
    i_low  = GDP <= 1
    i_high = GDP > 1
    mask_low  = i_low  & ~np.isnan(SS) & ~np.isnan(y)
    mask_high = i_high & ~np.isnan(SS) & ~np.isnan(y)
    SS_lowresults  = stats.linregress(SS[mask_low], y[mask_low])
    SS_low_p        = SS_lowresults.pvalue
    SS_highresults  = stats.linregress(SS[mask_high], y[mask_high])
    SS_high_p        = SS_highresults.pvalue


    x = [1,2,3,4,5,6]
    height = [float(SSp), float(HEp), float(FRp), float(GEp), float(SS_low_p), float(SS_high_p) ]
    labels = ["Social support","Health","Freedom","Generosity","Low-GDP","High-GDP"]
    fig, ax = plt.subplots()
    plt.bar(x, height)
    plt.yscale("log")
    plt.xticks(x, labels)
    ax.set_ylabel('p-value')
    
    display_title('P-values of independent variables for happiness', pref='Fig', num=3, center=True)
    plt.show()