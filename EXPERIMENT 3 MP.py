#RAVNEET KAUR       2020PHY1064
#2a       #LAGRANGE INTERPOLATION
import numpy as np
import sympy as sp       #importing the sympy library for math symbols
x0 = sp.symbols('x0')
x1 = sp.symbols('x1')    #used symbols in the Lagrange interpolation formula
y0 = sp.symbols('y0')
y1 = sp.symbols('y1')
Lx = [x0,x1]       #x0,x1,x2.....,xn
Ly = [y0,y1]        #y0,y1,y2,.....,yn
x=sp.symbols('x')       #Interpolating value
def Lagrange(Lx, Ly ,x):       #Defined a function for Lagrange intepolation
    p=0               #Length of x should be equal to Length of y
    for i in range ( len(Lx) ):         #i = 0,......,n
        y=1          #initial value for y
        for k in range ( len(Lx) ):
            if i != k:       #i≠j
                y=y* ( (x-Lx[k]) /(Lx[i]-Lx[k]) )    #Li(y)∀ i = 0 to n
        p+= y*Ly[i]        #pn(x) = y0L0(x) + y1L1(x) + ... + ynLn(x)
    return p
print('Pn(x) = ',Lagrange(Lx,Ly,x)) #Displaying the output
print()     
#2b   INVERSE LAGRANGE INTERPOLATION
def Inverse_Lagrange(Lx, Ly ,x):   #Defined a function for Inverse Lagrange Interpolation
    return Lagrange (Ly, Lx)
print('Qn(x) = ',Lagrange (Ly, Lx ,x))
print()
#2c Inbuilt Function for Lagrange Interpolation
from scipy.interpolate import lagrange
def Inbuilt_LP(Lx,Ly,x1):
    poly = lagrange(Lx, Ly) 
    L=poly(x1)
    return L
     #Displaying the Output
#Defined a function for graph
import matplotlib.pyplot as plt
def graph(x,y,array_x,array_y_f,array_y_in,pt_x,pt_y,xlab,ylab,titl):
    plt.scatter(x,y,marker='*',c="blue",label="DATA POINTS")
    plt.scatter(pt_x,pt_y,marker='o',c='black',label="INTERPOLATED POINT")
    plt.plot(array_x,array_y_in,c='green',label="SCIPY's INBUILT LAGRANGE INTERPOLATION",linestyle="-.")
    plt.plot(array_x,array_y_f,c="hotpink",label="INTERPOLATED LAGRANGE FUNCTION")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(titl)
    plt.grid(True)
    plt.legend()
    plt.show()     
#3a
β= [0.00, 0.2 ,0.4 ,0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]     #β = x
J0_β =[1.0, 0.99, 0.96, 0.91, 0.85, 0.76 ,0.67 ,0.57 ,0.46, 0.34, 0.22, 0.11, 0.00, -0.10, -0.18, -0.26] #J0_β = y
g=Lagrange(Lx, Ly ,x)
print('The value of Bessel Function at β =',x,':',Lagrange(β,J0_β,2.3))    #Displaying the output
print()       #Displaying the Output
print('Inverse Lagrange interpolation of Bessel Function at J0_β = 0.5 : ',Lagrange(J0_β,β,0.5))
print()
#3b
I = [2.81 ,3.24, 3.80 ,4.30, 4.37, 5.29, 6.03]      #x = I
V=[0.5, 1.2, 2.1 ,2.9, 3.6 ,4.5, 5.7]            #y = V
print('Lagrange interpolation of Linear interpolation at V = :',Lagrange(V,I,2.4))
print()
'''
from prettytable import PrettyTable
# Specify the Column Names while initializing the Table
myTable = PrettyTable(["Ques No:", "My function", "Inbuilt" ,"Error"])
# Add rows
myTable.add_row(["3a(i)",   Lagrange(β,J0_β,2.3),         Inbuilt_LP(β,J0_β,2.3) ,Lagrange(β,J0_β,2.3)-Inbuilt_LP(β,J0_β,2.3)])
myTable.add_row(["3a(ii)",  Lagrange(J0_β,β,0.5),          Inbuilt_LP(J0_β,β,0.5) ,Lagrange(J0_β,β,0.5)-Inbuilt_LP(J0_β,β,0.5) ])
myTable.add_row(["3b",      Lagrange(V,I,2.4),             Inbuilt_LP(V,I,2.4) ,Lagrange(V,I,2.4)-Inbuilt_LP(V,I,2.4)])
print(myTable)
'''
#For the Bessel Function
array_β=np.linspace(0,3,1000)
array_J0β_f=[]
array_J0β_i=[]
for i in array_β:
    array_J0β_f.append(Lagrange(β,J0_β,i))
    array_J0β_i.append(Inbuilt_LP(β,J0_β,i))
#For inverse Bessel Function
array_J0β=np.linspace(1,-0.26,1000)
array_β_f=[]
array_β_i=[]
for i in array_J0β:
    array_β_f.append(Lagrange(J0_β,β,i))
    array_β_i.append(Inbuilt_LP(J0_β,β,i))
graph(β,J0_β,array_β,array_J0β_f,array_J0β_i,2.3,Lagrange(β,J0_β,2.3),"β","J0_β","3a. (i) Bessel Function") 
graph(J0_β,β,array_J0β,array_β_f,array_β_i,0.5,Lagrange(J0_β,β,0.5),"J0_β","β","3a. (ii) Inverse Bessel Function")
#For Lagrange Interpolation of Photoelctric effect, the Linear Interpolation   
array_I=np.linspace(2.81,6.03,1000)
array_V_f=[]
array_V_i=[]
for i in array_I:
    array_V_f.append(Lagrange(I,V,i))
    array_V_i.append(Inbuilt_LP(I,V,i))
#For Inverse of Lagrange Interpolation of Photoelectric effect, the Linear Interpolation
array_V=np.linspace(0.5,5.7,1000)
array_I_f=[]
array_I_i=[]
for i in array_V:
    array_I_f.append(Lagrange(V,I,i))
    array_I_i.append(Inbuilt_LP(V,I,i))
graph(I,V,array_I,array_V_f,array_V_i,3.79,Lagrange(I,V,3.79),"I","V","3b. (i) Linear Interpolation Photoelectric Effect") 
graph(V,I,array_V,array_I_f,array_I_i,2.4,Lagrange(V,I,2.4),"V","I","3b. (ii) Inverse of Linear Interpolation Photoelectric effect")
#Plotted the outut for Lagrange Interpolation and Inverse Lagrange Interpolation