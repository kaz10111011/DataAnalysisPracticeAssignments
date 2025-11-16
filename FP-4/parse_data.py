import pandas as pd

df0 = pd.read_csv('WHR25_Data_Figure_2.1v3(Data for Figure 2.csv',encoding='latin1')


df = df0[  ['Life evaluation (3-year average)', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity']  ]



df = df.rename( columns={'Life evaluation (3-year average)':'Happiness', 'Explained by: Log GDP per capita':'GDP', 'Explained by: Social support':'Social Support', 'Explained by: Healthy life expectancy':'Health','Explained by: Freedom to make life choices':'Freedom',  'Explained by: Generosity':'Generosity'} )


