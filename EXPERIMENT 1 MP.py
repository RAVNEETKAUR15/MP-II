#3a 
#CURRENT AND VOLTAGE
import numpy as np
import matplotlib.pyplot as plt
print()
print("3(a) Estimation of the value of power delivered to the element using Trapezoidal and Simpson rule ")
print()

x= np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])  #CURRENT
y=np.array([0.0,0.5,2.0,4.05,8.0,12.5,18.0,24.5,32.0,40.5,50.0])   #VOLTAGE
plt.xlabel("CURRENT (mA)",fontsize=15,fontweight='bold')
plt.ylabel("VOLTAGE (V)",fontsize=15,fontweight='bold')
plt.title("VOLTAGE v/s CURRENT PLOT",fontsize=20,fontweight='bold')
plt.grid("true")
plt.plot(x,y,c='red')
plt.show()
#POWER USING TRAPEZOIDAL RULE
h=0.1
a=x[0]    #LOWER LIMIT
b=x[9]    #UPPER LIMIT
n=int((b-a)/h)         #NO OF SUBINTERVALS
S=0.5*(y[0]+y[10])     
for i in range(1,10):
        S=S+ y[i]  #VALUES OF NODAL POINTS
Power = S * h     #INTEGRAL
print("POWER USING TRAPEZOIDAL RULE :", Power)
#POWER USING SIMPSONS RULE
S1 = y[0]+y[10]    
for i in range(1,10):   
    if i%2 == 0:
       S1 = S1 + 2 * y[i]        #AT EVEN NODES
    else:
        S1 = S1 + 4 * y[i]      #AT ODD NODES

Power_S= S1 * h/3
print("POWER USING SIMPSONS RULE :" ,Power_S)
#3b
#TRAPEZOIDAL RULE
def trap(f,a,b,n):    # Function to evaluate the value of integral
    h = ( b - a )/ n   # Step size, we have divided the integral to n equal steps   
    sum = 0
    s = f(a) + f(b)    # Computing sum of first and last terms in above formula 
    for i in range(1,n):    
    
          S = f(a + i*h)        #x1----> a+h , x2----> a+2h and so on
        #Value of f(x) at the nodal points
          sum = sum + S      # Adding middle terms in the formula
          
    I = h/2 * (s + 2*sum)    #The final integral which is to be evaluated 
                        #h/2*[f(x0 = a) + f(xn = b) + 2[f(x1) + f(x2) + ... + f(xn−1)]]
    return(I)

#SIMPSONS RULE
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
print()
print(" 3(a) APPLICATION OF TRAPEZODIAL AND SIMPSONS RULE")
print()

f=eval("lambda x:"+input("Enter the value of the FUNCTION F(x): ")) #Defining the function to be used for evaluation
a = int(input('Enter the value of LOWER LIMIT (a):')) #Assigning the lower limit  
b = int(input("Enter the value of UPPER LIMIT (b):")) #Assigning the upper limit
n = int(input("Enter the value of NO. of intervals (n):"))
print()
I=trap(f,a,b,n)                 #integrals with F,UL,LL,ND
I_1=sim(f,a,b,n)

#Plotting the graph Trapezoidal and Simpson integration evaluate I(h) for given h and plot [h, I(h)]
def plot(f,a,b):
   #FOR TRAPEZOIDAL RULE
    N=np.arange(1,100,1)    #Number of Subintervals
    H=(b-a)/N         #N from 1 to 100 w
    Inte=[]            #integral values
    for i in N: 
        z=trap(f,a,b,i)     
        Inte.append(z)    #Putting the values of integrals one by one in Inte[]
       
    plt.plot(H,Inte,label="TRAPEZOIDAL INTEGRALS",c='deeppink',marker='.') 
    #FOR SIMPSONS RULE
    H2=(b-a)/(2*N)
    Ints=[]
    for j in 2*N:
        z=sim(f,a,b,j) 
        Ints.append(z)    #Putting the values of integrals one by one in Ints[]
    plt.plot(H2,Ints,label="SIMPSON INTEGRALS",c='navy',marker='.')
    plt.legend(loc="upper center")
    plt.xlabel("h",fontsize=15,fontweight='bold')   #Labelling the axis  (x and y)
    plt.ylabel("I(h)",fontsize=15,fontweight='bold')
    plt.xticks(c='red')
    plt.yticks(c='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid("true")
    plt.title("CONVERGENCE OF INTEGRATION METHODS, I(h) v/s h PLOT",fontsize=15,fontweight='bold')  
                                     #TITLE OF THE PLOT
    plt.show()
print("Integral is {:.8} using Trapezoidal Rule".format(float(I)))
print()                        #PRINTING VALUES OF INTEGRALS 
print("Integral is {:.8} using Simpsons Rule".format(float(I_1)))
print()
plot(f,a,b)          #PLOTTING THE GRAPH