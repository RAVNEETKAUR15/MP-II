import numpy as np
import sys
def Gauss_eli(a,n):
    for i in range(n):
        if aug[i][0] == 1:
            aug[[0,i]] = aug[[i,0]]
            break
    for i in range(n):
        if aug[i][i] == 0: #aii is the pivot element corresponding to ith row
            sys.exit("Division by 0!")
        for j in range(i+1,n):
            ratio = aug[j][i]/aug[i][i]
            for k in range(n+1):
                aug[j][k] = aug[j][k] - ratio * aug[i][k]
            print()
            print("STEPS FOR SOLVING")
            print(np.array(aug))
        
def Back_sub(a,n):  
    x = np.zeros([n]) 
    if aug[n-1][n-1]==0:
        sys.exit("Divison by 0!")
    x[n-1] = aug[n-1][n]/aug[n-1][n-1]
    for i in range(n-2,-1,-1):
        x[i] = aug[i][n]
        
        for j in range(i+1,n):
            x[i] = x[i] - aug[i][j]*x[j]      
        x[i] = x[i]/aug[i][i]     
    return x
'''4I1 - I2 +0I3 = -1
-I1 + 5I2- I3 = 2
0I1 + I2 - 3I3 = 1'''
 
aug = [[4,-1,0,-1],
       [-1,5,-1,2],
       [0,1,-3,1]]
print("AUGMENTED MATRIX")
print(np.array(aug))
n=3          #no. of unknowns
a=Gauss_eli(aug,n)
x = Back_sub(aug,n)
if aug==0:
    print("Inconsistent")
else:
    print()
    print("REDUCED ECHELON FORM")    
    print(np.array(aug))
    for i in range(len(x)):
     print("I"+str(i+1),":",x[i])    
     
import numpy as np
import sys
def interchange(mat,a,b):
    n=len(mat)
    for i in range(n):
        t = mat[a][i]
        mat[a][i] = mat[b][i]
        mat[a][i] = t
        
    return mat
def check(mat):
    n = len(mat)
    if mat[0][0]==0:
        for i in range(1,n-1):
            interchange(mat,0,i)
            if mat[0][0] !=0:
                break
            else:
                continue
        else:
            sys.exit('All entries of the first column is 0 ')
        if mat[0][0] != 0 and mat[0][0]!=1:
            for k in range(n-1):
                interchange(mat,0,k)
            else:
                return mat
        return mat
def gaussElimination(matrix):
    matrix = matrix.astype(float)
    print("The initial augumented matrix: ")
    print(matrix)
    if matrix[0,0] == 0.0:
        check(matrix)
    n,m = matrix.shape
    print("row: ",n,"Column: ",m)
    
    for i in range(0,n):
        for j in range(i+1,n):
            if matrix[j,i] !=0.0:
                print("using row" ,i, "as pivot and row ",j,"as target")
                multiplier = matrix[j,i]/matrix[i,i]
                matrix[j,i:m] = matrix[j,i:m] - multiplier*matrix[i,i:m]
                print(matrix)
                
    rank = len(matrix)
    print("The rank of the matrix is :", rank)
    return matrix
def backsub(matrix):
    n,m=matrix.shape
    x= np.zeros([n])
    x[n-1] = matrix[n-1][n]/matrix[n-1][n-1]
    for i in range(n-2,-1,-1):
        x[i] = matrix[i][n]
        for j in range(i+1,n):
            x[i]=x[i]-matrix[i][j]*x[j]
        x[i] = x[i]/matrix[i][i]
    return x
A = np.array([[9,3,4,7],
       [4,3,4,8],
       [1,1,1,3]])
b=gaussElimination(A)
c = backsub(b)
print(c)


#submitted
import numpy as np
def Swap_Row(a,row_no):
    n=len(a)
    r=row_no
    a=np.array(a)
    a=a.astype(float)
    m=int(np.size(a)/n)
    count=1
    while (a[r][r]==0):
        if (r==n-1):
            a[[n-1, 0],:]=a[[0, n-1],:]
        else:
            a[[r+count, r],:]=a[[r, r+count],:]
        count=count+1
        if count==n:
            return a,0
    return a,1
def RANK(a):
    n=len(a)
    m=int(np.size(a)/n)
    a=np.array(a)
    no_zero_rows=len(np.where(~a.any(axis=1))[0])
    coeff_mat=np.zeros((n,m-1))
    for i in range(n):
        for j in range(m-1):
            coeff_mat[i][j]=a[i][j]
    no_zero_rows2=len(np.where(~coeff_mat.any(axis=1))[0])
    rank1=n-no_zero_rows
    rank2=n-no_zero_rows2
    return rank1,rank2
def remove_column(a,r_c):
    n=len(a)
    m=int((np.size(a))/n)
    new_a=np.zeros((n,m-1))
    for i in range(n):
        for j in range(1,m,1):
            new_a[i][j-1]=a[i][j]
    return new_a
def reduced_form(a):
    n=len(a) 
    m=int(np.size(a)/len(a))
    flag=1
    for i in range(n):
        if a[i][i]==0:
            print("row interchange")
            a,p=(a,i)
            print(a)
            if p==0:
                print("removing redundant column")
                a=remove_column(a,i)
                m=m-1       
        for j in range(i+1,n): #elements lower than pivot element moving columnwise
            r=a[j][i]/a[i][i]  #ratio according to the columnn
            for k in range(m):  
                a[j][k]=round((a[j][k]-(r*a[i][k])),3) #row operation
            print("intermediate steps")
            print(np.array(a))
            rank1,rank2=RANK(a)
            if rank1<n:
                return a,rank1,rank2
    return a,rank1,rank2
def back_substitution(a,n):
    x=np.zeros([n])
    x[n-1] =round(a[n-1][n]/a[n-1][n-1],4) #solving last row
    for i in range(n-2,-1,-1):     #moving upwards
        x[i] = a[i][n]             #assigning last column value to x
        for j in range(i+1,n):     #solving for each row 
            x[i] = x[i] - a[i][j]*x[j]  #subtracting the values from last column
        x[i] = round(x[i]/a[i][i],4)    #dividing it by appropriate pivot value 
    return x   
aug=[[4,-1,0,-1],[-1,5,-1,2],[0,1,-3,1]]
#aug=[[2,7,4,2],[8,5,7,2],[7,4,0,1],[1,6,4,7]] #overdetermined
#aug=[[9,6,4,4,5],[2,3,9,6,5],[2,1,3,7,5]] #undetermined
n=len(aug)
m=int(np.size(aug)/n)
coeff_mat=np.zeros((n,m-1))
print("coefficient matrix:")
for i in range(n):
    for j in range(m-1):
        coeff_mat[i][j]=aug[i][j]
print(np.array(coeff_mat))
print("AUGMENTED MATRIX")
print(np.array(aug))
ans,r1,r2=reduced_form(aug)
print("REDUCED ECHELON FORM")
print(np.array(ans))
print("RANK OF AUGMENTED MATRIX: ",r1)
print("RANK OF COEFFICIENT MATRIX: ",r2)
if m-1==r1: #no. of variable is equal to rank
    x=back_substitution(ans,r1)
    print("UNIQUE SOLUTION")
    for i in range(len(x)):
        print("x"+str(i+1),":",x[i])
elif(m-1<r1):
    print("OVERDETRMINED (more equations than variables)")
else:
    print("UNDERDETERMINED (less equations than variables)")
     
