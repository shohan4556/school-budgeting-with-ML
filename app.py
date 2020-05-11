import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# the ultimate goal is to predict the probability that a certain label is attached to a budget line item

df = pd.read_csv('TrainingData.csv', index_col = 0)


### EDA ###
#print(df.head(5))
print(df.info())
#print(df.describe())

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

### Convert to caterogrycal data
Labels = ['Function', 'Use', 'Sharing', 'Reporting', 'Student_Type', 'Position_Type', 'Object_Type', 'Pre_K', 'Operating_Status']

categorize_label = lambda x : x.astype('category')
#print(df['Function'].head(10))
df[Labels] = df[Labels].apply(categorize_label, axis=0)
#print(df[Labels].dtypes)
# 37 category 
#print(df['Function'].head(10))

# Counting unique labels
num_unique_labels = df[Labels].apply(pd.Series.nunique)
num_unique_labels.plot(kind='bar')
plt.xlabel('Labels')
plt.ylabel('Number of unique values')
plt.show()

# our goal is to minimize the loss function 
# it is better to less less confident than confident and wrong prediction 
# log loss provides a steep penalty for predictions that are both wrong and confident
def compute_log_loss(pred_prob, actual, eps=1e-14):
    """
    pred_prob: predcited probability 
    actual: actual value 
    eps: since log(0) is infinity need to offset our value slightly
    """
    pred = np.clip(pred_prob, eps, 1-eps)
    loss = -1 * np.mean(actual * np.log(pred) + (1-actual) * np.log(1-pred))
    return loss

print('correct and confident : ', compute_log_loss(0.9, 1))
print('correct and not confident : ', compute_log_loss(0.5, 1))  
print('wrong and confident : ', compute_log_loss(0.95, 0))
print('wrong and not confident : ', compute_log_loss(0.4, 0))  
    