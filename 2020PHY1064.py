#UNIVERSITY ROLL NO. - 20068567047 
#NAME - RAVNEET KAUR
#COLLEGE ROLL NO. - 2020PHY1064
#SIMPSONS RULE
print("PRACTICAL EXAM MATHEMATICAL PHYSICS-II")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def sim(f,a,b,n):          #Function to evaluate the value of integral
    h2 = ( b - a )/ (2*n)  # Step size, we have divided the integral to 2n equal steps  
    sum = 0
    s=f(a)+f(b)            # Computing sum of first and last terms in above formula 
    for i in range(1,2*n):
         S_1=f(a+i*h2)      #x1---> a+h , x2----> a+2h and so on
         #Value of f(x) at the nodal points
         
         if (i%2==0):          #if it gives the remainder 0 then use this sum 
          sum = sum + 2 * S_1  #At Even Nodes 
         elif (i%2==1):        #else use the this one
          sum = sum + 4 * S_1  #At Odd Nodes
        
         I_1=h2/3*(s+sum) #The final integral which is to be evaluated 
         # I≈ h/2*[f(x0 = a) + f(xn = b) + 2[f(x1) + f(x2) + ... + f(xn−1)]]
    return(I_1)
def f(y): #Defining the function to be used for evaluation
    x=(25-y**2)**(1/2)
    return x
a =0 #Assigning the lower limit  
b = 5 #Assigning the upper limit
n = int(2.5)
n1 = int(25)
n2 = int(250)
n3 = int(2500)
n4 = int(25000)
print()
#integrals with F,UL,LL,ND
A0=sim(f, a, b, n)  #different values of nodal points/stepsize
A1=sim(f, a, b, n1)
A2=sim(f,a,b,n2)
A3=sim(f, a, b, n3)
A4=sim(f,a,b,n4)
A_ana = 19.63  #  Analytical Value of the integral
A_list = [A0,A1,A2,A3,A4]
h_list = [1,0.1,0.01,0.001,0.001]
delA_list = [A0-A_ana,A1-A_ana,A2-A_ana,A3-A_ana,A4-A_ana]
logh = np.log(h_list)
logdelA=np.log(delA_list)
for i in range(len(h_list)):
    data={"H":h_list,"A_h":A_list,"delA":delA_list}   
df=pd.DataFrame(data)
print(df)
print()
plt.plot(h_list,A_list,c = "red")
plt.xlabel("H (StepSize)",fontweight = 20,c = "deeppink")
plt.ylabel("A_h",fontweight = 20,c = "deeppink")
plt.title("A_h vs H",fontweight = 30,c = "Blue")
plt.legend("H")
plt.grid()
plt.show()
plt.xlabel("Log of H (StepSize)",fontweight = 20,c = "Green")
plt.ylabel("Log of δA_h",fontweight = 20,c = "Green")
plt.title("Log of  δA_h vs Log of H",fontweight = 50,c = "Orange")
plt.grid()
plt.plot(logh,logdelA,c = "navy")
plt.legend("H")
plt.show()
from scipy.stats import linregress    #Calculated slope using Scipy library
print(linregress(h_list, A_list))
print()