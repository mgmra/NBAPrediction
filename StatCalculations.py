import pandas as pd


x = ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
y = range(0,10)

x_s = pd.DataFrame(data = zip(x,y), columns = ['Group', 'Number'])

#rmx = x_s.rolling(5).mean().shift(1)
# In[ ]:
x_s
rmx = x_s.groupby('Group')['Number'].rolling(3).mean().groupby('Group').shift(1)


