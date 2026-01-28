import numpy as np
from scipy import stats
import parse_data as pa
import descriptive as de
de.define_variables(pa.df)
y, GDP, SS, HE, FR, GE, SS1 = de.define_variables(pa.df)

def Regression(y, GDP, SS, HE, FR, GE, SS1):
    mask = (~np.isnan(y) &
        ~np.isnan(GDP) &
        ~np.isnan(SS) &
        ~np.isnan(HE) &
        ~np.isnan(FR) &
        ~np.isnan(GE) )


    print('p value')
    print()
    
    SSresults = stats.linregress(SS[mask], y[mask])
    SSp = SSresults.pvalue
    print('Social support:',SSp)
  
    HEresults = stats.linregress(HE[mask], y[mask])
    HEp = HEresults.pvalue
    print('Health:',HEp)
    
    FRresults = stats.linregress(FR[mask], y[mask])
    FRp = FRresults.pvalue
    print('Freedom:',FRp)
    
    GEresults = stats.linregress(GE[mask], y[mask])
    GEp = GEresults.pvalue
    print('Generosity:',GEp)

    
    i_low  = GDP <= 1
    i_high = GDP > 1
    mask_low  = i_low  & ~np.isnan(SS) & ~np.isnan(y)
    mask_high = i_high & ~np.isnan(SS) & ~np.isnan(y)
    SS_lowresults  = stats.linregress(SS[mask_low], y[mask_low])
    SS_low_p        = SS_lowresults.pvalue
    SS_highresults  = stats.linregress(SS[mask_high], y[mask_high])
    SS_high_p        = SS_highresults.pvalue
    print( 'Low-GDP:　p=', SS_low_p)
    print( 'High-GDP:　p=', SS_high_p)