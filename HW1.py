



# For creating data tables
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style 
import numpy as np 

# creating a series from players in the Golden State Warriors team 
# notes from slide 41 pp "Pandas" for references 
lastseason = {'Player':['Stephen Curry', 'Klay Thompson', 'Jordan Poole','Andrew Wiggins'],
        'Scored':[26,20,19,17],}

df = pd.DataFrame(lastseason,columns =['Player', 'Scored'])
print(df)



# Add salary as a column slide 42 PP "Pandas"
# Salary from espn 
salaries = np.array(["48070014","40600080","3901399","33616770",])
df['Salaries'] = salaries
print(df)

# The following code will create a bar graph
plt.figure() 
style.use("ggplot")
plt.bar(df.Player,df.Scored,label="Points Scored", color="orange",align="center",alpha=1)
plt.title("Points Scored Last Season")
plt.xlabel("Players")
plt.xticks(rotation=45)
plt.ylabel("Points Scored")
plt.legend()
plt.grid(True, color="g")


#Creating a pie chart from Player's salary 
salaries = ["48070014","40600080","3901399","33616770"]
explode = (0,0,0,0)

plt.figure()
plt.pie(salaries,explode=explode,labels=df.Player,autopct="%.1f%%",shadow=False,startangle=30)
plt.axis("equal")
plt.title("Top 4 Player's Salary who play in Golden State Warriors ")
plt.legend()
plt.show()