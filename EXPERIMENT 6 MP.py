import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def damped_oscillator(t,Var,cons,b): 
    k,m= cons
    x,v = Var
    dx_dt = v    
    dv_dt = -(b/m)*v -(k/m)*x                         
    return np.array([dx_dt,dv_dt])
def simple_pendulum(t,Var,cons1,θ):
    g,L=cons1
    θ,v = Var
    dθ_dt = v    
    dv_dt = -g/L*θ
    return np.array([dθ_dt,dv_dt])
def coupled_pendulum(t,Var,cons2,wo):
    xa,va,xb,vb=Var
    xb=np.radians(xb)
    xa=np.radians(xa)
    k,m=cons
    dxa_dt=va
    dva_dt=-(wo**2)*xa-(k/m)*(xa-xb)
    dxb_dt=vb
    dvb_dt=-(wo**2)*xb+(k/m)*(xa-xb)
    return np.array([dxa_dt,dva_dt,dxb_dt,dvb_dt])
def RK_2(tn,IC,func,h,cons,b):        #IC= [0,0,-1] or [0,0] where t=0,x=0 and y =-1 & t=0,i=0, tn is the final value
    time_vect = np.array([IC[0]])   #[0,0,-1] where t=0
    a = []
    for i in range(1,len(IC)):
        a.append([IC[i]])
    y_vect = np.array(a)
    N = int((tn - IC[0])/h)  
    for i in range(N):
        m1_vect = h*func(time_vect[i],y_vect[:,i],cons,b)   #slope 1: m1 = hf(tn, xn);l1 = hg(tn,xn,yn)
        m2_vect = h*func(time_vect[i] + h ,y_vect[:,i] + m1_vect,cons,b)  #slope 2 : m2 = hf(tn + h, xn + m1);l2 = hg(tn + (h/2), xn + l1) 
        t = time_vect[i] + h        #ti = a + ih
        t_vect = np.append(time_vect,t)
        time_vect = t_vect 
        y_next = y_vect[:,i] + ((m1_vect+m2_vect)/2)   #x(t + h) = x(t) + ((m1+m2)/2 , (l1+l2)/2)
        Y = []
        for j in range(len(y_vect)):
            y = np.append(y_vect[j],y_next[j]);Y.append(y)
        y_vect = np.array(Y)
    return [time_vect,y_vect]
if __name__ == "__main__" :
    y1=[0,2,0];k=4;m=0.5;cons=k,m   #SI units
    b1=[0,0.5,np.sqrt(4*k*m),9]
    T=2*np.pi*(np.sqrt(m/k))
    RK2_1 = RK_2(5*T,y1,damped_oscillator,0.01,cons,b1[0]) #simple harmonic oscillator
    RK2_2 = RK_2(20,y1,damped_oscillator,0.1,cons,b1[1])#underdamped 
    RK2_3 = RK_2(20,y1,damped_oscillator,0.1,cons,b1[2])#critical
    RK2_4 = RK_2(20,y1,damped_oscillator,0.1,cons,b1[3]) #overdamped
    g=9.8;L=1;cons1=g,L
    tp=2*np.pi*(np.sqrt(L/g))
    y3=[0,1,0]
    RK2=RK_2(10*tp,y3,simple_pendulum,tp/50,cons1,y3[1]) #theta
    y2=[0,10,0,-10,0];k1=90;m1=10;cons2=k1,m1
    g1=9.8;L1=10;wo=np.sqrt(g1/L1)
    RK_A=RK_2(80,y2,coupled_pendulum,0.01,cons2,wo) #omega
    print("Table for Simple Harmonic Oscillator")
    data={"Time/Time Period":RK2_1[0]/T,"Displacement":RK2_1[1][0],"Velocity":RK2_1[1][1]}
    print(pd.DataFrame(data))
    print("Table for Damped Harmonic Oscillator(Underdamped)")
    data1={"Time":RK2_2[0],"Displacement":RK2_2[1][0],"Velocity":RK2_2[1][1]}
    print(pd.DataFrame(data1))
    print("Table for Simple Pendulum")
    data2={"Time":RK2[0]/tp,"Angular Displacement":RK2[1][0],"Angular Velocity":RK2[1][1]}
    print(pd.DataFrame(data2))
    print("Table for Coupled Pendulum")
    data3={"Time":RK_A[0],"Angular Displacement(xa)":RK_A[1][0],"Angular Velocity(va)":RK_A[1][1],"Angular Displacement(xb)":RK_A[1][2],"Angular Velocity(vb)":RK_A[1][3]}
    print(pd.DataFrame(data3))
    pd.set_option("display.max_rows",None)
    pd.set_option('display.expand_frame_repr',False)
def graph():
    fig,ax = plt.subplots(2) 
    ax[0].plot(RK2_1[0]/T,RK2_1[1][0],c='indianred',marker="*",label='Displacement')
    ax[0].set(xlabel = "Time/Time Period",ylabel ='Displacement',title="Displacement vs Time")
    ax[0].legend(loc='best');ax[0].grid()
    ax[1].plot(RK2_1[0]/T,RK2_1[1][1],marker="*",label = 'Velocity',c='lightseagreen')
    ax[1].set(xlabel = "Time/Time Period",ylabel ='Velocity',title="Velocity vs Time")
    ax[1].legend(loc='best');ax[1].grid()
    fig.suptitle("Simple Harmonic osicallator (b=0)")
graph()
def graph1():
     fig,ax = plt.subplots(2) 
     ax[0].scatter(RK2_2[0],RK2_2[1][0],c='deeppink',marker="*",label='Underdamped')
     ax[0].set(xlabel = "Time",ylabel ='Displacement')
     ax[0].legend(loc='best');ax[0].grid()
     ax[1].scatter(RK2_2[0],RK2_2[1][1],marker="*",label = 'Underdamped',c='green')
     ax[1].set(xlabel = "Time",ylabel ='Velocity')
     ax[1].legend(loc='best');ax[1].grid()
     ax[0].scatter(RK2_3[0],RK2_3[1][0],c='violet',marker="*",label='Critical')
     ax[0].set(xlabel = "Time",ylabel ='Displacement')
     ax[0].legend(loc='best');ax[0].grid()
     ax[1].scatter(RK2_3[0],RK2_3[1][1],marker="*",label = 'Critical',c='crimson')
     ax[1].set(xlabel = "Time",ylabel ='Velocity')
     ax[1].legend(loc='best');ax[1].grid()
     ax[0].scatter(RK2_4[0],RK2_4[1][0],c='purple',marker="*",label='Overdamped')
     ax[0].set(xlabel = "Time",ylabel ='Displacement')
     ax[0].legend(loc='best');ax[0].grid()
     ax[1].scatter(RK2_4[0],RK2_4[1][1],marker="*",label = 'Overdamped',c='chocolate')
     ax[1].set(xlabel = "Time",ylabel ='Velocity')
     ax[1].legend(loc='best');ax[1].grid()
     fig.suptitle("Damped Harmonic osicallator")
graph1()
def graph2():
    fig,ax = plt.subplots(2) 
    ax[0].plot(RK2[0]/tp,RK2[1][0],marker="*",c='green',label='Displacement')
    ax[0].set(xlabel = "Time/Time Period",ylabel ='Angular Displacement',title="Displacement vs Time")
    ax[0].legend(loc='best');ax[0].grid()
    ax[1].plot(RK2[0]/tp,RK2[1][1],marker="*",label = 'Velocity',c='deeppink')
    ax[1].set(xlabel = "Time/Time Period",ylabel ='Angular Velocity',title="Velocity vs Time")
    ax[1].legend(loc='best');ax[1].grid()
    fig.suptitle("Simple Pendulum")
graph2()
def graph3():
    fig,ax = plt.subplots(2,2) 
    ax[0,0].plot(RK_A[0],RK_A[1][0],c='limegreen',label='Displacement')
    ax[0,0].set(xlabel = "Time",ylabel ='Angular Displacement',title="Angular Displacement ($\dfrac {d^2xa}{dt}$)")
    ax[0,0].legend(loc='best');ax[0,0].grid()
    ax[0,1].plot(RK_A[0],RK_A[1][1],label = 'Velocity',c='magenta')
    ax[0,1].set(xlabel = "Time",ylabel ='Angular Velocity',title="Angular Velocity vs Time ($\dfrac {d^2xa}{dt}$)")
    ax[0,1].legend(loc='best');ax[0,1].grid()
    ax[1,0].plot(RK_A[0],RK_A[1][2],c='dodgerblue',label='Displacement')
    ax[1,0].set(xlabel = "Time",ylabel ='Angular Displacement',title="Angular Displacement vs Time ($\dfrac {d^2xb}{dt}$)")
    ax[1,0].legend(loc='best');ax[1,0].grid()
    ax[1,1].plot(RK_A[0],RK_A[1][3],label = 'Velocity',c='peru')
    ax[1,1].set(xlabel = "Time",ylabel ='Angular Velocity',title="Angular Velocity vs Time ($\dfrac {d^2xb}{dt}$)")
    ax[1,1].legend(loc='best');ax[1,1].grid()
    fig.suptitle("Coupled Pendulum")
graph3()
'''http://hyperphysics.phy-astr.gsu.edu/hbase/oscda.html#c2
https://scipy-lectures.org/intro/scipy/auto_examples/plot_odeint_damped_spring_mass.html
https://www.myphysicslab.com/pendulum/pendulum-en.html'''