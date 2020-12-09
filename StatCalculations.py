import pandas as pd


x = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
y = x.extend(['w', 'd'])
print(x)
y = range(0,10)
z = range(11,20)

x_s = pd.DataFrame(data = zip(x,y,z), columns = ['Group', 'Number1', 'Number2'])

#rmx = x_s.rolling(5).mean().shift(1)
# In[ ]:
x_s
rmx = x_s.groupby('Group').rolling(3).mean().groupby('Group').shift(1)


