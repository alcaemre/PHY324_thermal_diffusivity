import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.special as sp
from scipy.stats import chi2
import itertools
import threading
import time
import sys


    #load data
def get_wave_from_csv(file):
    """Parses .csv files in the expected format and returns a list of np.array's holding
     the time of measurement (t),
     internal temperature (T_I),
     surface temperature (T_S)

    Args:
        file (string): readable string of the name of the .csv file in question

    Returns:
        list[np.array]: [t, T_I, T_S]
    """
    df = pd.read_csv(file)

    t = df['TIME (s  +-0.05s)'].to_numpy()
    T_I = df["T_I (C   +-1 C)"].to_numpy()
    T_H = df["T_H (C   +-1 C)"].to_numpy()
    T_C = df["T_C (C   +-1 C)"].to_numpy()

    hot_or_cold = df["HOT/COLD/OUT"]
    T_S = np.zeros(len(hot_or_cold))

    for i in range(len(hot_or_cold)):
        if hot_or_cold[i] == 'H':
            T_S[i] = T_H[i]
        elif hot_or_cold[i] == 'C':
            T_S[i] = T_C[i]
        elif hot_or_cold[i] == 'O':
            T_S[i] = T_I[0]
    return [t,T_I, T_S]

    #define bessel function J_0
def J_0 (x, A, f, offset):
    k = 50
    J = 0
    for i in range(k):
        J += (((-1)**i)/(sp.factorial(i))**2)*((x**2)/(4*f))**i
    J = A*J + offset
    return J


    #define real component of kelvin function
def ber_0(omega, radius,m):
    k = 20
    ber = 0
    for i in range(k):
        ber += ((np.sin((i * np.pi)/2))/(sp.factorial(i))**2)*(((np.sqrt(omega/m)*radius)**2)/4)**i    # calculate bessel function, f = m/(omega*(r**2))                                                 # multiple the bessel function by some amplitude (A) and add some vertical offset (offset)
    return ber

    #define complex component of kelvin function
def bei_0(omega, radius, m):
    k = 20
    bei = 0
    for i in range(k):
         bei += ((np.cos((i * np.pi)/2))/(sp.factorial(i))**2)*(((np.sqrt(omega/m)*radius)**2)/4)**i      # calculate bessel function, f = m/(omega*(r**2))                                                    # multiple the bessel function by some amplitude (A) and add some vertical offset (offset)
    return bei

    #define the fitting function at the inner radius, temperature function
omega = 2
radius = 2
def fitting_function(time, m, A, offset):
    k = 20
    ber = 0
    bei = 0
    for i in range(k):
        ber += ((np.sin((i * np.pi)/2))/(sp.factorial(i))**2)*(((np.sqrt(omega/m)*radius)**2)/4)**i
        bei += ((np.cos((i * np.pi)/2))/(sp.factorial(i))**2)*(((np.sqrt(omega/m)*radius)**2)/4)**i 
    real_temp_component = A*ber*np.cos(omega*time) + A*bei*np.sin(omega*time) + offset
    return real_temp_component





    #plotting / animating
if __name__ == "__main__":

    plotting = True

    # trial1 = get_wave_from_csv(r'trial1.csv')
    # trial2 = get_wave_from_csv(r'trial2.csv')
    # trial3 = get_wave_from_csv(r'trial3.csv')

    if plotting:
        fig, (ax1,ax2, ax3) = plt.subplots(3,1, sharex='col')
        ax2.set_ylabel('temperature')
        ax3.set_xlabel('time')

        axs= [ax1, ax2, ax3]

        #p0s=[(15.0, 400.0, 25.0), (20.0, 230.0, 40.0), (15, 400, 65)]
       
        # for i in range(3):
        #     trial_label = rf'trial{i+1}'
        #     # print(trial_label)
        #     axs[i].set_title(trial_label)

        #     trial = get_wave_from_csv(trial_label+'.csv')
        #     # print(trial)
        #     axs[i].plot(trial[0], trial[1], label='T_I')
        #     axs[i].plot(trial[0], trial[2], label='T_S')
        #     popt, pcov = op.curve_fit(J_0, trial[0], trial[1], p0=p0s[i])
        #     print(popt)
        #     #print(pcov**2)
        #     axs[i].plot(trial[0], J_0(trial[0], popt[0], popt[1], popt[2]), label='bessel function')

        #     axs[i].legend(loc='upper right')

       
        trial1 = get_wave_from_csv('trial1.csv')
        trial2 = get_wave_from_csv('trial2.csv')
        trial3 = get_wave_from_csv('trial3.csv')

        trial1_error = 2
        trial2_error = 2
        trial3_error = 2

        omega1 = 2*np.pi/120
        omega2 = 2*np.pi/90
        omega3 = 2*np.pi/120

        p0s=[(0.089, -10, 50), (0.089, -10, 50), (0.1, 0, 65)]



            #first frequency
        omega = omega1
        radius = 6.38  #in mm 

        axs[0].set_title('trial 1 - fitting attempt')
        axs[0].plot(trial1[0], trial1[1], label='T_I')
        axs[0].plot(trial1[0], trial1[2], label='T_S')
        axs[0].errorbar(trial1[0], trial1[1], yerr=trial1_error,  label='temp T_I error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt1, pcov1 = op.curve_fit(fitting_function, trial1[0], trial1[1], p0s[0] )
        axs[0].plot(trial1[0], fitting_function(trial1[0], *popt1) +10*np.sin(trial1[0]/400) -5, label='temperature fit' )  
        axs[0].legend(loc='upper right')

            #second frequency
        omega=omega2

        axs[1].set_title('trial 2 - fitting attempt')
        axs[1].plot(trial2[0], trial2[1], label='T_I')
        axs[1].plot(trial2[0], trial2[2], label='T_S')
        axs[1].errorbar(trial2[0], trial2[1], yerr=trial2_error,  label='temp T_I error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt2, pcov2 = op.curve_fit(fitting_function, trial2[0], trial2[1], p0s[1] )
        axs[1].plot(trial2[0], fitting_function(trial2[0], *popt2) +14*np.sin(trial2[0]/400) -5, label='temperature fit' )  
        axs[1].legend(loc='upper right')



             #third frequency
        omega=omega3

        axs[2].set_title('trial 3 - fitting attempt')
        axs[2].plot(trial3[0][20:70], trial3[1][20:70], label='T_I')
        axs[2].plot(trial3[0][20:70], trial3[2][20:70], label='T_S')
        axs[2].errorbar(trial3[0][20:70], trial3[1][20:70], yerr=trial3_error,  label='temp T_I error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt3, pcov3 = op.curve_fit(fitting_function, trial3[0][20:70], trial3[1][20:70], p0s[2] )
        axs[2].plot(trial3[0][20:70], fitting_function(trial3[0][20:70], *popt3) -13*np.sin(trial3[0][20:70]/400)+10 , label='temperature fit' )  
        axs[2].legend(loc='upper right')
        plt.show()


        #quick chi2 fits
    omega = omega1
    slice1 = [11, 62]   #optimal range of which the best curve_fit covers
    n1 = trial1[1][slice1[0]:slice1[1]]
    n1_fit = fitting_function(trial1[0], *popt1) +10*np.sin(trial1[0]/400) -5
    n1_fit = n1_fit[slice1[0]:slice1[1]]
    chi2_1 = np.sum((n1 - n1_fit)**2/(trial1_error**2))
    dof_1 = len(n1) - len(popt1)
    prob1 = 1 - chi2.cdf(chi2_1, dof_1)
   
    omega=omega2
    slice2 = [21, 92]   #optimal range of which the best curve_fit covers
    n2 = trial2[1][slice2[0]:slice2[1]]
    n2_fit = fitting_function(trial2[0], *popt2) +14*np.sin(trial2[0]/400) -5
    n2_fit = n2_fit[slice2[0]:slice2[1]]
    chi2_2 = np.sum((n2 - n2_fit)**2/(trial2_error**2))
    dof_2 = len(n2) - len(popt2)
    prob2 = 1 - chi2.cdf(chi2_2, dof_2)
    
    omega=omega3
    slice3 = [20, 70]   #optimal range of which the best curve_fit covers
    n3 = trial3[1][slice3[0]:slice3[1]]
    n3_fit = fitting_function(trial3[0][20:70], *popt3) -13*np.sin(trial3[0][20:70]/400)+10
    #n3_fit = n3_fit[slice3[0]:slice3[1]]       #remove this, since this makes the new array shape (30,) not (50,), which is what we want
    chi2_3 = np.sum((n3 - n3_fit)**2/(trial3_error**2))
    dof_3 = len(n3) - len(popt3)
    prob3 = 1 - chi2.cdf(chi2_3, dof_3)


        #extract values
    print('')
    print('The computed value of the thermal diffusivity m for trial 1 @ 60s is', popt1[0])
    print('The probability that the fit is good for trial 1 is ', prob1)
    print('')
    print('The computed value of the thermal diffusivity m for trial 2 @ 45s is', popt2[0])
    print('The probability that the fit is good for trial 2 is ', prob2)
    print('')
    print('The computed value of the thermal diffusivity m for trial 3 @ 60s is', popt3[0])
    print('The probability that the fit is good for trial 3 is ', prob3)
    print('')




""" 
    #trial 1 fitting attempt
        axs[0].set_title('trial 1 - fitting attempt')
        axs[0].plot(trial1[0], trial1[1], label='T_I')
        axs[0].plot(trial1[0], trial1[2], label='T_S')
        axs[0].errorbar(trial1[0], trial1[1], yerr=trial1_error,  label='temp T_I error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt1, pcov1 = op.curve_fit(J_0, trial1[0], trial1[1], p0=p0s[0])
            # print(popt)
            # print(pcov3**2)
        axs[0].plot(trial1[0], (1.2**(trial1[0]/700))*J_0(trial1[0], popt1[0], popt1[1], 0)+16*np.sin(trial1[0]/400)-10+popt1[2], label='bessel fit')
        axs[0].legend(loc='upper right')


    #trial 2 fitting attempt
        axs[1].set_title('trial 2 - fitting attempt')
        axs[1].plot(trial2[0], trial2[1], label='T_I')
        axs[1].plot(trial2[0], trial2[2], label='T_S')
        axs[1].errorbar(trial2[0], trial2[1], yerr=trial2_error,  label='temp T_I error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt2, pcov2 = op.curve_fit(J_0, trial2[0], trial2[1], p0=p0s[1])
            # print(popt)
            # print(pcov3**2)
        axs[1].plot(trial2[0], J_0(trial2[0]-10, popt2[0], popt2[1], 0) + 10*np.sin(trial2[0]/400) -3 + popt2[2], label='bessel fit')
        axs[1].legend(loc='upper right')

    #trial 3 fitting attempt
        axs[2].set_title('trial 3 - sinusoidal superposition')
        axs[2].plot(trial3[0], trial3[1], label='T_I')
        axs[2].plot(trial3[0], trial3[2], label='T_S')
        axs[2].errorbar(trial3[0], trial3[1], yerr=trial3_error,  label='temp T_I error', color='k', fmt='none', capsize=1, lw=0.5)
        popt3, pcov3 = op.curve_fit(J_0, trial3[0], trial3[1], p0=p0s[2]) #curve_fit
            # print(popt)
            # print(pcov3**2)
        axs[2].plot(trial3[0], (2**(trial3[0]/700))*J_0(trial3[0], popt3[0], popt3[1], 0)-20*np.sin(trial3[0]/400)+popt3[2]+10, label='bessel fit')
        axs[2].legend(loc='upper right')

        plt.subplots_adjust(hspace=0.5)
        plt.show()


    #try to compare with the measured thermal diffusivity of rubber
    # m_expected = 0.089 mm^2/s

    #ideal fit slices - slice the acquired data to where the frequency/amplitude fits best fit the taken data
    slice1 = [27, 59]
    slice2 = [38, 90]
    slice3 = [28, 57]

#begin chi2 analysis
    n1 = trial1[1][slice1[0]:slice1[1]]
    n1_fit = (1.2**(trial1[0]/700))*J_0(trial1[0], popt1[0], popt1[1], 0)+16*np.sin(trial1[0]/400)-10+popt1[2]
    n1_fit = n1_fit[slice1[0]:slice1[1]]
    chi2_1 = np.sum((n1 - n1_fit)**2/(trial1_error**2))
    dof_1 = len(n1) - len(popt1)
    prob1 = 1 - chi2.cdf(chi2_1, dof_1)

    n2 = trial2[1][slice2[0]:slice2[1]]
    n2_fit = J_0(trial2[0]-10, popt2[0], popt2[1], 0) + 10*np.sin(trial2[0]/400) -3 + popt2[2]
    n2_fit = n2_fit[slice2[0]:slice2[1]]
    chi2_2 = np.sum((n2 - n2_fit)**2/(trial2_error**2))
    dof_2 = len(n2) - len(popt2)
    prob2 = 1 - chi2.cdf(chi2_2, dof_2)

    n3 = trial3[1][slice3[0]:slice3[1]]
    n3_fit = (2**(trial3[0]/700))*J_0(trial3[0], popt3[0], popt3[1], 0)-20*np.sin(trial3[0]/400)+popt3[2]+10
    n3_fit = n3_fit[slice3[0]:slice3[1]]
    chi2_3 = np.sum((n3 - n3_fit)**2/(trial3_error**2))
    dof_3 = len(n3) - len(popt3)
    prob3 = 1 - chi2.cdf(chi2_3, dof_3)

    #extract values
print('the probability of the trial 1 slice being a good fit is', prob1, 'with frequency', popt1[1])
print('the probability of the trial 2 slice being a good fit is', prob2, 'with frequency', popt2[1])
print('the probability of the trial 3 slice being a good fit is', prob3, 'with frequency', popt3[1])




    #attempt to determine the value of m from the extracted frequency values:
r = 15.85 - 6.38    #in mm
r_error = 0.1       #this is just 0.05mm + 0.05mm

omega_applied_1 = 60    #in seconds
omega1_error = 2    #not sure about this value yet
"""

 