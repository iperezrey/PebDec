import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
data = pd.read_excel('responses.xlsx')
data_bp = pd.read_excel('responses.xlsx', skiprows=[1]) # skiprows skips the raw where are the result of SAM
 
 
x_labels = ['6648'] + ['6662'] + ['6666'] + ['6673'] + ['6701'] + ['6710'] + ['6716'] + ['6720'] + ['6750'] + ['6762']
 
# Create the boxplot
plt.figure(figsize=(16,12))
sns.stripplot(x=x_labels,
              y=(data.iat[0,0], data.iat[0,1], data.iat[0,2], data.iat[0,3], data.iat[0,4], data.iat[0,5], data.iat[0,6], data.iat[0,7], data.iat[0,8], data.iat[0,9]),
              marker='o',
              s=8,
              color='Red',
              )

sns.boxplot(data=data_bp,
            showmeans=True,
            meanprops={'marker':'s','markerfacecolor':'k','markeredgecolor':'w',},
            boxprops={'facecolor': (.4, .6, .8, .5)},
            flierprops={'marker': 'x'},
            )

plt.xlabel('Picture code', fontsize=18)
plt.ylabel('Percentage of pebbles observed (%)', fontsize=18)

# Create the legends
legend_SAM =  plt.Line2D([], 
                         [], 
                         linestyle='None',
                         color='red', 
                         marker='o', 
                         markersize=8, 
                         label='SAM_results'
                         )

legend_average = plt.Line2D([], 
                            [], 
                            marker='s', 
                            color='k',
                            markeredgecolor='w', 
                            markersize=8, 
                            linestyle='None', 
                            label='Average'
                            )

handles=[legend_SAM, legend_average]

# Plot the legends
plt.legend(loc='lower right', handles=handles)

plt.show()