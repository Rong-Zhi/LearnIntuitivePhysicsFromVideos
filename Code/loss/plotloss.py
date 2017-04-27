import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

df = pd.DataFrame.from_csv('train_loss_D.csv', parse_dates=False)
x = df.Step
y = df.Value
filtered = lowess(y,x,is_sorted=True,frac=0.025,it=1)
plt.plot(x,y,'sandybrown',label='original loss')
plt.plot(filtered[:,0],filtered[:,1],linewidth=1.3,color='red',label='smooth=0.025')
# ax = df.plot(legend=False,x='Step',y='Value')
plt.legend()
plt.ylabel('D Loss')
plt.xlabel('Step')
plt.show()