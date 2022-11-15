import numpy as np
import sys
def Gauss_Seidal(A):   #Augumented Matrix
    tol = 1e-5   # Reading tolerable error
    a = np.array(A)  #changed to an array
    m = len(A)    #no. of unknowns (rows)
    n = int(np.size(a)/m)  #no. of columns
    N = n-1  #No. of columns in the Coefficient matrix
    x_k1 = np.zeros(N)#array of no. zeros same as that of N
    print("LINEAR EQAUTIONS:")
    print("4I1 - I2 +0I3 = -1")
    print("-I1 + 5I2- I3 = 2")
    print("0I1 - I2 + 3I3 = -1")
    print("*----------------------------------*")
    print("AUGUMENTED MATRIX")
    print(a)
    print("*----------------------------------*")
    # Find diagonal coefficients
    Diag = np.diag(np.abs(a))
    print("THE DIAGONAL ELEMENTS ARE :",Diag)
    print("*----------------------------------*")
    # Find row sum without diagonal
    Diag_new = np.sum(a, axis=1) - Diag
    if np.all(Diag > Diag_new):
        print("MATRIX IS DIAGONALLY DOMINANT")
        print("*----------------------------------*")
    else:
        print("MATRIX IS NOT DIAGONALLY DOMINANT")
        print("*----------------------------------*")
        for i in range(n):
            if a[i][i] == 0: #aii is the pivot element corresponding to ith row
                sys.exit("Division by 0!")
    x = 0
    for i in range(N):
        x_k1[i] = a[i][-1]/a[i][i] #The initial approximations may be taken to be x(0)i = bi/aii.
    while True :
        SOLUTION = np.array(x_k1) #initial value of the loop
        for i in range(N):
            a_x = 0 
            for j in range(N) :
                if i == j :    
                    continue
                else : 
                    a_x += a[i][j] * x_k1[j]
                    continue
            x_k1[i] = (a[i][-1] - a_x)/a[i][i]
        DIFF = abs(x_k1 - SOLUTION)
        sub = abs(max(DIFF))
        x += 1
        if sub <= tol:
            break
    return x_k1,x
AUG_MAT=[[4,-1,0,-1],
         [-1,5,-1,2],
         [0,-1,3,-1]]
print("GAUSS-SEIDAL METHOD")
print("*----------------------------------*")
sol,x = Gauss_Seidal(AUG_MAT)
print("No. of Iterations: ",x)	
print("*------------------------------*")
for i in range(len(sol)):
    print("I"+str(i+1)+" : ", sol[i])
