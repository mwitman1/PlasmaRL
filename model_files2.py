import numpy as np
from casadi import *
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt

## define the model and the integrator
# NEED TO DEFINE delta = timestep, nominaly 1.3s
def thermal_int(delta):
    nx = 1 #outputs
    nu = 1 #inputs
    np = 3 #parameters
    
    u = SX.sym('u',nu)
    x = SX.sym('y',nx)
    q = SX.sym('q',np)
    
    #Surface glass
    propSurf={'rho':2.8e3, 
              'cp':795.00, 
              'k':1.43, 
              'd':0.20e-3} #surface thickness
    
    # #Surface metal
    #propSurf={'rho':2.710e3, 
    #          'cp':0.91e3, 
    #          'k':5, 
    #          'd':0.20e-3} #surface thi2ckness
                        
    #system dimensions
    dim={'r':1.5e-3};
    dim['vol']=3.1416*1e-2*dim['r']**2.0 #m^3 volume of plasma chamber 
    dim['Ac']=3.1416*dim['r']**2.0 #m^2 flow crossectional area
    
    P=u[0]*5
    T=x[0]*300
    
    a1=q[0]*38
    a2=q[1]*0.003
    Tinf=q[2]+273
    #Parameters
    dsep=4e-3 #separation distance
    # Tinf = 293.00 # K ambient temperature
    eta  = 0.4+0.07*dsep/4.00e-3 #power deposition efficiency
    # a1=38.9
    # a2=0.003

    dTs_maxdt=(a2*eta*P-a1*dim['r']*2.0*3.1416*propSurf['d']*propSurf['k']*(T-Tinf))/(propSurf['rho']*propSurf['cp']*dim['Ac']*propSurf['d']);

    xdot = vertcat(dTs_maxdt/300.0)
    jet_dae = {'x':x, 'p':vertcat(u,q), 'ode':xdot}
    opts = {'tf':delta}
    
    I = integrator('I', 'idas', jet_dae, opts)
    
    return I , jet_dae

# function to give next time step values
def thermal_model(I,y0,Pow,param):
    
    y0_n=(y0+273.)/300. #normalized initial T
    P_n=Pow/5. #normalized Power
    
    y=I(x0=y0_n,p=vertcat(P_n,param))

    return y['xf'].full()*300.-273

# function to give min and max achieavable temperature step values    
def thermal_model_minmax(jet_dae,param):
    
    y0_n=(param[2]-273.)/300. #normalize initial T
    
    opts = {'tf':1e5}
    I = integrator('I', 'idas', jet_dae, opts)

    y_min=I(x0=y0_n,p=vertcat(1.5/5.,param))
    y_max=I(x0=y0_n,p=vertcat(5./5.,param))

    return y_min['xf'].full()*300.-273, y_max['xf'].full()*300.-273

# function to give SS temperature for a step power   
def thermal_model_inf(jet_dae,param,power):
    
    y0_n=(param[2]-273.)/300. #normalize initial T
    
    opts = {'tf':1e5}
    I = integrator('I', 'idas', jet_dae, opts)

    y_inf=I(x0=y0_n,p=vertcat(power/5.,param))

    return y_inf['xf'].full()*300.-273

######################### USAGE ########################

if __name__ == "__main__":
    #load the model
    I, jet_dae=thermal_int(1.3)
    y0=30. #initial temperature in C
    y0_Al=30. #initial temperature in C
    Pow=3. #applied power in W

    #nominal values for a1 and a2
    #glass a1=2.39, a2=0.8177
    #metal a1=0.932621, a2=0.505957
    #new metal a1=1.1236, 0.71131
    param=[2.39,0.8177,26] #parameters a1,a2 and Tinf in C
    param=[0.932621,0.505957,26] #parameters a1,a2 and Tinf in C

    # the test ones
    #param=[2.39,0.8177,23] #parameters a1,a2 and Tinf in C
    #param_Al=[0.3,0.08,27] # low time constant
    #param_Al=[3.7, 1.2,27] #parameters a1,a2 and Tinf in C

   
    my_data = [] 
    my_data_Al = [] 
    for t in range(0, 500):  
    #calculate next time step
        y_next=thermal_model(I,y0,Pow,param)
        my_data.append([t, y_next[0][0]])
        print(my_data[-1])
        y0 = y_next

        y_next_Al=thermal_model(I,y0_Al,2.5,param_Al)
        my_data_Al.append([t, y_next_Al[0][0]])
        y0_Al = y_next_Al
    
    #find min and max possible temperatures in deg C
    ymin, ymax = thermal_model_minmax(jet_dae,param)
    print('ymin',ymin)
    print('ymax',ymax)
    
    ymin, ymax = thermal_model_minmax(jet_dae,param_Al)
    print('ymin',ymin)
    print('ymax',ymax)

    ## predictions against data
    #my_data = genfromtxt('glass_data_out.csv', delimiter=',')
    my_data = np.array(my_data)
    my_data_Al = np.array(my_data_Al)
    
    plt.plot(my_data[:,0],my_data[:,1])
    plt.plot(my_data_Al[:,0],my_data_Al[:,1])
    plt.tight_layout()
    plt.show()

