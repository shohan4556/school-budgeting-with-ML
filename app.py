import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('TrainingData.csv', index_col = 0)

print(df.head(5))
print(df.info())
print(df.describe())

# Total: Stands for the total cost of the expenditure. This number tells us how much the budget item cost
'''
FTE: Stands for "full-time equivalent". If the budget item is associated to an employee, this number tells us the percentage of full-time that the employee works. A value of 1 means the associated employee works for the school full-time. A value close to 0 means the item is associated to a part-time or contracted employee.
'''

# see the distribution of part-time and full time emplyoe 
plt.hist(df['FTE'].dropna(), bins=None,range=[0,1])
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')
plt.show()

