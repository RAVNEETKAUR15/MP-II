import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def frwd_euler(dx,IC,T_half,h):
    a,xn = IC;b = 5*T_half;N= int((b-a)/h);t = [a]; x = [xn];x1=[];s = [dx(a,xn)]
    for i in range(N):
        xn = xn+h*dx(a,xn)
        x0 = xn
        a = a+h
        slope = dx(a,xn)
        x.append(x0)
        t.append(a)
        s.append(slope)
        x1.append(x0)  
    x1.append("-")  
    l = [t,x,h,s,x1,b]
    return l
def RK2(dx,IC,T_half,h):
    a,xn = IC;b = 5*T_half;N= int((b-a)/h); t = [a];x2 = [];x = [xn];s1 = [dx(a,xn)]
    K1 = [h*dx(a,xn)];K2 = [h*dx(a +h , xn + dx(a,xn))];KRK2 = [(h*dx(a,xn) + h*dx(a +h , xn + dx(a,xn)))/2]
    for i in range(N):
        k1 = h*dx(a,xn)     #slope 1: k1 = hf(tn, xn)
        k2 = h*dx(a +h , xn +k1) #slope 2 : k2 = hf(tn + h, xn + k1)
        slope = dx(a,xn)
        kRK2 =(k1+k2)/2  #where (k1 + k2)/2 is the slope of RK2
        xn = xn + kRK2         #xn+1 = xn + (k1 + k2)/2
        a = a + h          #ti = a + ih 
        x0 = xn
        x2.append(x0)
        x.append(x0)
        t.append(a)
        s1.append(slope)
        K1.append(k1)
        K2.append(k2)
        KRK2.append(kRK2)
    x2.append("-")
    l1 = [t,x,h,K1,K2,KRK2,s1,x2]
    return l1
def Anal(x0,a,b,h,tau):  
    t_arr=[]
    X_arr=[]
    e=np.arange(a,b+h,h)
    for t in e:
        D=x0*np.exp(-1*t/tau)
        X_arr.append(D)
        t_arr.append(t)
    return t_arr,X_arr
def graph(t,x,t1,x1,L,C1,C2,Canl1,Canl12,E1,E2,k,Q,eu,rk,C11,C22,Rk1,Rk2,k1,Y,Z,H1,H2,Z1,H11,H22,Z2,H12,H21,Z3,H24,H44,Z4):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t, x,marker="D",label="EULER",c='deeppink') 
    axs[0, 0].plot(t1, x1,marker="o",label="RK2")
    axs[0, 0].plot(C1,C2,marker="v",label="ANALYTIC",c='orange')
    axs[0,0].set_xlabel(L[0],fontweight = 'bold',c = 'red')
    axs[0,0].set_ylabel(L[1],fontweight = 'bold',c='red')
    axs[0, 0].set_title("COMPARISON OF 3 METHODS, EULERS,RUNGE KUTTA & ANALYTICAL",fontweight = 'bold',c="darkgreen")
    axs[0, 1].plot(Canl1,Canl12,marker="*",label="ANALYTIC")
    axs[0, 1].plot(E1, E2,marker="v", label="h ={}".format(k))
    axs[0, 1].plot(H1,H2,marker="D", label="h ={}".format(Z1))
    axs[0, 1].plot(H11,H22,marker="o", label="h ={}".format(Z2))
    axs[0,1].set_xlabel("t",fontweight = 'bold',c = 'Deeppink')
    axs[0,1].set_ylabel(Y[0],fontweight = 'bold',c = 'Deeppink')
    axs[0, 1].set_title("EULER METHOD with Variation of h",fontweight = 'bold',c="indigo")
    axs[1,0].plot(C11,C22,marker="H",label="ANALYTIC")
    axs[1,0].plot(H12,H21,marker="o", label="h ={}".format(Z3))
    axs[1,0].plot(Rk1, Rk2,marker="v", label="h ={}".format(k1))
    axs[1,0].plot(H24,H44,marker="D", label="h ={}".format(Z4))
    axs[1,0].set_xlabel("t",fontweight = 'bold',c = 'Green')
    axs[1,0].set_ylabel(Z[0],fontweight = 'bold',c = 'Green')
    axs[1,0].set_title("RUNGE KUTTA'S METHOD (ORDER 2) with Variation of h",fontweight = 'bold',c="brown")
    axs[1, 1].scatter(Q,eu , label="ERROR IN EULER")
    axs[1, 1].scatter(Q,rk , label="ERROR IN RK2")
    axs[1,1].set_xlabel("Time",fontweight = 'bold',c = 'darkblue')
    axs[1,1].set_ylabel("ABSOLUTE ERROR",fontweight = 'bold',c = 'Darkblue')
    axs[1,1].set_title("RELATIVE ERRORS IN EULER AND RUNGE KUTTA METHOD",fontweight = 'bold',c="Deeppink")
    axs[0,0].grid();axs[1,0].grid();axs[0,1].grid();axs[1,1].grid()
    axs[0,0].legend();axs[1,0].legend();axs[0,1].legend();axs[1,1].legend()
    plt.show()
def Q3a():
    print("Ques 3a RADIOACTIVE DECCAY USING EULER'S METHOD")
    dx = lambda lam,N1: -N1/tau
    t_half = 4;lam = 0.693/t_half; tau = 1/lam;a = 0;xn = 20000    
    IC =np.array([a,xn])
    q = frwd_euler(dx,IC,t_half,t_half/10)
    q1=frwd_euler(dx,IC,t_half,t_half/20)
    q2=frwd_euler(dx,IC,t_half,t_half/50)
    q[0]= np.array(q[0])/t_half
    q[1]= np.array(q[1])/xn
    print("τ for Radioactive Deccay :",tau)
    L1=["Number of Half lives ","N / N0"] 
    Y1 = ['N']
    IC =np.array([a,xn])
    Data1 = {"tn":q[0],"xn":q[1],"slope":q[3],"xn+1":q[4]}
    print("The value of step size h(Euler)=%.5f"%q[2])
    print("------------------------------------------------")
    print(pd.DataFrame(Data1))
    print("------------------------------------------------")
    a3 = RK2(dx,IC,t_half,t_half/10)
    A = RK2(dx,IC,t_half,t_half/20)
    A1 = RK2(dx,IC,t_half,t_half/50)
    a3[0]= np.array(a3[0])/t_half;a3[1]= np.array(a3[1])/xn
    print("Ques 3a RADIOACTIVE DECCAY USING RUNGE KUTTA'S METHOD")
    Data2 = {"tn":a3[0],"xn":a3[1],"slope":a3[6],"m1":a3[4],"m2":a3[5],"mrk2":a3[6],"xn+1":a3[7]}
    print("The value of step size h(Euler)=%.5f"%a3[2])
    print("-----------------------------------------------------------------------------------------")
    print(pd.DataFrame(Data2))
    print("-----------------------------------------------------------------------------------------")
    pd.set_option('display.expand_frame_repr',False)
    c=Anal(xn,a,q[5],q[2],tau)
    eu=[]
    rk=[]
    for i in range(len(q[1])):  
        eu.append((c[1][i]-q[1][i])/c[1][i])
        rk.append((c[1][i]-a3[1][i])/c[1][i])
    graph(q[0],q[1],a3[0],a3[1],L1,np.array(c[0])/t_half,np.array(c[1])/xn,np.array(c[0])/t_half,np.array(c[1])/xn,q[0],q[1],q[2],q[0],eu,rk,np.array(c[0])/t_half,np.array(c[1])/xn,a3[0],a3[1],a3[2],Y1,Y1,np.array(q1[0])/t_half,np.array(q1[1])/xn,q1[2],np.array(q2[0])/t_half,np.array(q2[1])/xn,q2[2],np.array(A[0])/t_half,np.array(A[1])/xn,A[2],np.array(A1[0])/t_half,np.array(A1[1])/xn,A1[2])
def Q3b():
    print("Ques 3b RC CIRCUIT USING EULER'S METHOD")
    dx = lambda t,V : -V/tau
    R = 1000;C = 10**(-6);tau = R*C
    print("τ for RC CIRCUIT :",tau)
    t_half = tau*0.693;a = 0;xn = 10
    L2=["t / R*C ","V / V0"] ; Y1=["V"]
    IC = np.array([a,xn])  #The inital conditions
    a1 = frwd_euler(dx,IC,t_half,t_half/10);a1[0] = np.array(a1[0])/t_half;a1[1] = np.array(a1[1])/xn
    a11 = frwd_euler(dx,IC,t_half,t_half/20)
    a12 = frwd_euler(dx,IC,t_half,t_half/50)
    Data = {"tn":a1[0],"xn":a1[1],"slope":a1[3],"xn+1":a1[4]}
    print("The value of step size h(RC) =%.5f"%a1[2])
    print("------------------------------------------")
    print(pd.DataFrame(Data))
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    print("-------------------------------------------")
    a4 = RK2(dx,IC,t_half,t_half/10);a4[0] = np.array(a4[0])/(t_half);a4[1] = np.array(a4[1])/xn
    a41 = RK2(dx,IC,t_half,t_half/20)
    a42 = RK2(dx,IC,t_half,t_half/50)
    print("Ques 3b RC CIRCUIT USING RUNGE KUTTA'S METHOD")
    Data = {"tn":a4[0],"xn":a4[1],"slope":a4[6],"m1":a4[4],"m2":a4[5],"mrk2":a4[6],"xn+1":a4[7]}
    print("The value of step size h(RK2) =%.5f"%a4[2])
    print("-------------------------------------------------------------------------------------")
    print("τ for RC CIRCUIT :",tau)
    print(pd.DataFrame(Data))
    pd.set_option('display.expand_frame_repr',False)
    print("--------------------------------------------------------------------------------------")
    c=Anal(xn,a,a1[5],a1[2],tau)
    eu=[];rk=[]
    for i in range(len(a1[1])):  
        eu.append((c[1][i]-a1[1][i])/c[1][i])
        rk.append((c[1][i]-a4[1][i])/c[1][i])
    graph(a1[0],a1[1],a4[0],a4[1],L2,np.array(c[0])/t_half,np.array(c[1])/xn,np.array(c[0])/t_half,np.array(c[1])/xn,a1[0],a1[1],a1[2],a1[0],eu,rk,np.array(c[0])/t_half,np.array(c[1])/xn,a4[0],a4[1],a4[2],Y1,Y1,np.array(a11[0])/t_half,np.array(a11[1])/xn,a11[2],np.array(a12[0])/t_half,np.array(a12[1])/xn,a12[2],np.array(a41[0])/t_half,np.array(a41[1])/xn,a41[2],np.array(a42[0])/t_half,np.array(a42[1])/xn,a42[2])
def Q3c():
    print("Ques 3c STOKES' LAW USING EULER'S METHOD")
    dx = lambda t,V : -V/tau
    a = 2;m = 2;η = 15;tau = m/6*np.pi*η*a;t_half = tau*0.693
    print("τ for Stokes Law :",tau)
    a = 0;xn = 10; L3 = ["t / m/6*np.pi*η*a","V / V0"];Y1=["V"]
    IC = np.array([a,xn])  #The inital conditions
    a2 = frwd_euler(dx,IC,t_half,t_half/10);a21 = frwd_euler(dx,IC,t_half,t_half/20);a22 = frwd_euler(dx,IC,t_half,t_half/50)
    a2[0] = np.array(a2[0])/t_half;a2[1] = np.array(a2[1])/xn
    Data = {"tn":a2[0],"xn":a2[1],"slope":a2[3],"xn+1":a2[4]}
    print("The value of step size h(RC) =%.5f"%a2[2])
    print("------------------------------------------")
    print(pd.DataFrame(Data))
    print("------------------------------------------")
    a5 = RK2(dx,IC,t_half,t_half/10);a5[0] = np.array(a5[0])/t_half;a5[1] = np.array(a5[1])/xn
    a51 = RK2(dx,IC,t_half,t_half/20)
    a52 = RK2(dx,IC,t_half,t_half/50)
    print("Ques 3c STOKES' LAW USING RUNGE KUTTA'S METHOD")
    Data = {"tn":a5[0],"xn":a5[1],"slope":a5[6],"m1":a5[4],"m2":a5[5],"mrk2":a5[6],"xn+1":a5[7]}
    print("The value of step size h(RC) =%.5f"%a5[2])
    print("-----------------------------------------------------------------------")
    print(pd.DataFrame(Data))
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    print("-----------------------------------------------------------------------")
    pd.set_option('display.expand_frame_repr',False)
    c=Anal(xn,a,a2[5],a2[2],tau)
    eu=[]
    rk=[]
    for i in range(len(a2[1])):  
        eu.append((c[1][i]-a2[1][i])/c[1][i])
        rk.append((c[1][i]-a5[1][i])/c[1][i])
    graph(a2[0],a2[1],a5[0],a5[1],L3,np.array(c[0])/t_half,np.array(c[1])/xn,np.array(c[0])/t_half,np.array(c[1])/xn,a2[0],a2[1],a2[2],a2[0],eu,rk,np.array(c[0])/t_half,np.array(c[1])/xn,a5[0],a5[1],a5[2],Y1,Y1,np.array(a21[0])/t_half,np.array(a21[1])/xn,a21[2],np.array(a22[0])/t_half,np.array(a22[1])/xn,a22[2],np.array(a51[0])/t_half,np.array(a51[1])/xn,a51[2],np.array(a52[0])/t_half,np.array(a52[1])/xn,a52[2])
Q3a()
Q3b()
Q3c()       