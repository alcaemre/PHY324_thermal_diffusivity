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

    #latex encoding
plt.rcParams['text.usetex'] = True


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





    #solid measurements and controlled data with uncertainties
radius_inner = 6.38  #in mm 
radius_uncertainty = 0.01 #mm - systematic

omega1 = 2*np.pi/120
omega2 = 2*np.pi/90
omega3 = 2*np.pi/120

    #uncertainty in time - systematic
time_uncertainty = 0.05

    #uncertainty in time - systematic
omega1_error = 2*np.pi * (120*0.05)
omega2_error = 2*np.pi * (90*0.05)
omega3_error = 2*np.pi * (120*0.05)


#     #define the relative uncertainty functions
# def uncertainty_percentage1(x, xerr, z):
#     zerr = np.sqrt((x/xerr)**2)*z
#     return zerr

# def uncertainty_percentage2(x, xerr, y, yerr, z):
#     zerr = np.sqrt((x/xerr)**2 + np.sqrt((y/yerr)**2))*z
#     return zerr



# trial1_m_uncertainty=0
# k=20
# for i in range(k):
#     trial1_m_uncertainty += ((np.sin((i * np.pi)/2))/(sp.factorial(i))**2)*i * 0.5*uncertainty_percentage2(0.5 *((omega1/omega1_error)**2), radius_inner, radius_uncertainty, np.sqrt(omega1)*radius_inner) + ((np.cos((i * np.pi)/2))/(sp.factorial(i))**2)*i * 0.5*uncertainty_percentage2(0.5 *((omega1/omega1_error)**2), radius_inner, radius_uncertainty, np.sqrt(omega1)*radius_inner) 











    #plotting 
if __name__ == "__main__":

    plotting = True #arguments are True/False as a parameter 

    if plotting:
        fig, (ax1,ax2, ax3) = plt.subplots(3,1)
        ax2.set_ylabel('Temperature (°C)')
        ax3.set_xlabel('Time (s)')

        axs= [ax1, ax2, ax3]
    
        trial1 = get_wave_from_csv('trial1.csv')
        trial2 = get_wave_from_csv('trial2.csv')
        trial3 = get_wave_from_csv('trial3.csv')

        trial1_error = 2
        trial2_error = 2
        trial3_error = 2

        p0s=[(0.089, -10, 50), (0.089, -10, 50), (0.1, 0, 65)]

        radius = radius_inner

            #first frequency
        omega = omega1

        axs[0].set_title('Trial 1 - 120s Period, Low Inital $T_I$')
        axs[0].set_ylabel('Temperature ($^\circ$C)')
        axs[0].plot(trial1[0], trial1[1], label='$T_I$')
        axs[0].plot(trial1[0], trial1[2], label='$T_S$')
        axs[0].errorbar(trial1[0], trial1[1], xerr = time_uncertainty, yerr=trial1_error,  label='Temperature $T_I$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        axs[0].errorbar(trial1[0], trial1[2], xerr = time_uncertainty, yerr=trial1_error,  label='Temperature $T_S$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt1, pcov1 = op.curve_fit(fitting_function, trial1[0], trial1[1], p0s[0] )
        axs[0].plot(trial1[0], fitting_function(trial1[0], *popt1) +10*np.sin(trial1[0]/400) -5, label='Temperature Curve Fit' )  
        axs[0].legend(loc='upper right', prop={'size':6})

            #second frequency
        omega=omega2

        axs[1].set_title('Trial 2 - 90s Period, Low Initial $T_I$')
        axs[1].set_ylabel('Temperature ($^\circ$C)')
        axs[1].plot(trial2[0], trial2[1], label='$T_I$')
        axs[1].plot(trial2[0], trial2[2], label='$T_S$')
        axs[1].errorbar(trial2[0], trial2[1], xerr = time_uncertainty, yerr=trial2_error,  label='Temperature $T_I$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        axs[1].errorbar(trial2[0], trial2[2], xerr = time_uncertainty, yerr=trial2_error,  label='Temperature $T_S$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt2, pcov2 = op.curve_fit(fitting_function, trial2[0], trial2[1], p0s[1] )
        axs[1].plot(trial2[0], fitting_function(trial2[0]-6, *popt2) +14*np.sin(trial2[0]/400) -5, label='Temperature Curve Fit' )  
        axs[1].legend(loc='upper right', prop={'size':6})



             #third frequency
        omega=omega3

        axs[2].set_title('Trial 3 - 120s Period, High Initial $T_I$')
        axs[2].set_xlabel('Time (S)')
        axs[2].set_ylabel('Temperature ($^\circ$C)')
        axs[2].plot(trial3[0], trial3[1], label='$T_I$')
        axs[2].plot(trial3[0], trial3[2], label='$T_S$')
        axs[2].errorbar(trial3[0], trial3[1], xerr = time_uncertainty, yerr=trial3_error,  label='Temperature $T_I$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        axs[2].errorbar(trial3[0], trial3[2], xerr = time_uncertainty, yerr=trial3_error,  label='Temperature $T_S$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
        popt3, pcov3 = op.curve_fit(fitting_function, trial3[0][20:70], trial3[1][20:70], p0s[2] )
        axs[2].plot(trial3[0], fitting_function(trial3[0], *popt3) -13*np.sin(trial3[0]/400)+10 , label='Temperature Curve Fit' )  
        axs[2].legend(loc='upper right', prop={'size':6})




        plt.subplots_adjust(hspace=0.3)
        plt.show()


        #quick chi2 fits - outside of the plotting loop
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
    print('The computed value of the thermal diffusivity $m$ for trial 1 @ 120s is', popt1[0], u"\u00B1", np.max(pcov1[0]))
    print('The probability that the fit is good for trial 1 is ', prob1)
    print('')
    print('The computed value of the thermal diffusivity $m$ for trial 2 @ 90s is', popt2[0], u"\u00B1", np.max(pcov2[0]))
    print('The probability that the fit is good for trial 2 is ', prob2)
    print('')
    print('The computed value of the thermal diffusivity $m$ for trial 3 @ 120s is', popt3[0], u"\u00B1", np.max(pcov3[0]))
    print('The probability that the fit is good for trial 3 is ', prob3)
    print('')



    #error differences
omega=omega1
n1_fit_maximalerror = fitting_function(trial1[0], popt1[0]+np.max(pcov1[0]), popt1[1]+np.max(pcov1[1]), popt1[2]+np.max(pcov1[2])) +10*np.sin(trial1[0]/400) -5
n1_fit_minimalerror = fitting_function(trial1[0], popt1[0]-np.max(pcov1[0]), popt1[1]-np.max(pcov1[1]), popt1[2]-np.max(pcov1[2])) +10*np.sin(trial1[0]/400) -5

omega=omega2
n2_fit_maximalerror = fitting_function(trial2[0]-6, popt2[0]+np.max(pcov2[0]), popt2[1]+np.max(pcov2[1]), popt2[2]+np.max(pcov2[2])) +14*np.sin(trial2[0]/400) -5
n2_fit_minimalerror = fitting_function(trial2[0]-6, popt2[0]-np.max(pcov2[0]), popt2[1]-np.max(pcov2[1]), popt2[2]-np.max(pcov2[2])) +14*np.sin(trial2[0]/400) -5

omega=omega3
n3_fit_maximalerror = fitting_function(trial3[0], popt3[0]+np.max(pcov3[0]), popt3[1]+np.max(pcov3[1]), popt3[2]+np.max(pcov3[2])) -13*np.sin(trial3[0]/400)+10
n3_fit_minimalerror = fitting_function(trial3[0], popt3[0]-np.max(pcov3[0]), popt3[1]-np.max(pcov3[1]), popt3[2]-np.max(pcov3[2])) -13*np.sin(trial3[0]/400)+10



fig2 , ((ax1, ax2), (ax3,ax4), (ax5, ax6)) = plt.subplots(3,2)

omega = omega1
ax1.plot(trial1[0], trial1[1], label='$T_I$', color='k', alpha=0.5)
ax1.errorbar(trial1[0], trial1[1], xerr = time_uncertainty, yerr=trial1_error,  label='Temperature $T_I$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
ax1.plot(trial1[0], fitting_function(trial1[0], *popt1) +10*np.sin(trial1[0]/400) -5, label='Temperature Curve Fit', color='b' )  
ax1.plot(trial1[0], fitting_function(trial1[0], popt1[0]+np.max(pcov1[0]), popt1[1]+np.max(pcov1[1]), popt1[2]+np.max(pcov1[2])) +10*np.sin(trial1[0]/400) -5, label='Maximal Temperature Error', color='r', alpha=0.5 )  
ax1.plot(trial1[0], fitting_function(trial1[0], popt1[0]-np.max(pcov1[0]), popt1[1]-np.max(pcov1[1]), popt1[2]-np.max(pcov1[2])) +10*np.sin(trial1[0]/400) -5, label='Minimal Temperature Error', color='r', alpha=0.5 )  
ax1.set_xlim(110, 620)
ax1.legend(loc='best', prop={'size':6})
ax1.set_title('Trial 1 - Uncertainties of Optimal Fitting Parameters')
ax1.set_ylabel('Temperature ($^\circ$C)')

#ax2.plot(trial1[0], trial1[1]-fitting_function(trial1[0], *popt1) +10*np.sin(trial1[0]/400) -5, label='Curve Fit Difference')
ax2.errorbar(trial1[0], np.zeros(len(trial1[0])), xerr = time_uncertainty, yerr=trial1_error,  label='Temperature $T_I$ Error Overlap', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
ax2.plot(trial1[0], trial1[1]-n1_fit_maximalerror, label='Maximal Error Difference', color='#FF4000')
ax2.plot(trial1[0], trial1[1]-n1_fit_minimalerror, label='Minimal Error Difference', color='#FFBF00')
ax2.set_xlim(110, 620)
ax2.legend(loc='best', prop={'size':6})
ax2.set_title('Trial 1 - Overlap of Parameter Uncertainty with Data Error')




omega = omega2
ax3.plot(trial2[0], trial2[1], label='$T_I$', color='k', alpha=0.5)
ax3.errorbar(trial2[0], trial2[1], xerr = time_uncertainty, yerr=trial2_error,  label='Temperature $T_I$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
ax3.plot(trial2[0], fitting_function(trial2[0]-6, *popt2) +14*np.sin(trial2[0]/400) -5, label='Temperature Curve Fit', color='b' )  
ax3.plot(trial2[0], fitting_function(trial2[0]-6, popt2[0]+np.max(pcov2[0]), popt2[1]+np.max(pcov2[1]), popt2[2]+np.max(pcov2[2])) +14*np.sin(trial2[0]/400) -5, label='Maximal Temperature Error', color='r', alpha=0.5 )  
ax3.plot(trial2[0], fitting_function(trial2[0]-6, popt2[0]-np.max(pcov2[0]), popt2[1]-np.max(pcov2[1]), popt2[2]-np.max(pcov2[2])) +14*np.sin(trial2[0]/400) -5, label='Minimal Temperature Error', color='r', alpha=0.5 )  
ax3.set_xlim(105, 460)
ax3.legend(loc='best', prop={'size': 6})
ax3.set_title('Trial 2 - Uncertainties of Optimal Fitting Parameters')
ax3.set_ylabel('Temperature ($^\circ$C)')

#ax4.plot(trial2[0], trial2[1]-fitting_function(trial2[0], *popt2) +14*np.sin(trial2[0]/400) -5, label='Curve Fit Difference')
ax4.errorbar(trial2[0],np.zeros(len(trial2[0])), xerr = time_uncertainty, yerr=trial2_error,  label='Temperature $T_I$ Error Overlap', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
ax4.plot(trial2[0], trial2[1]-n2_fit_maximalerror, label='Maximal Error Difference', color='#FF4000')
ax4.plot(trial2[0], trial2[1]-n2_fit_minimalerror, label='Minimal Error Difference', color='#FFBF00')
ax4.set_xlim(105, 460)
ax4.legend(loc='best', prop={'size': 6})
ax4.set_title('Trial 2 - Overlap of Parameter Uncertainty with Data Error')




omega = omega3
ax5.plot(trial3[0], trial3[1], label='$T_I$', color='k', alpha=0.5)
ax5.errorbar(trial3[0], trial3[1], xerr = time_uncertainty, yerr=trial3_error,  label='Temperature $T_I$ Error', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
ax5.plot(trial3[0], fitting_function(trial3[0], *popt3) -13*np.sin(trial3[0]/400)+10, label='Temperature Curve Fit', color='b' )  
ax5.plot(trial3[0], fitting_function(trial3[0], popt3[0]+np.max(pcov3[0]), popt3[1]+np.max(pcov3[1]), popt3[2]+np.max(pcov3[2])) -13*np.sin(trial3[0]/400)+10, label='Maximal Temperature Error', color='r', alpha=0.5 )  
ax5.plot(trial3[0], fitting_function(trial3[0], popt3[0]-np.max(pcov3[0]), popt3[1]-np.max(pcov3[1]), popt3[2]-np.max(pcov3[2])) -13*np.sin(trial3[0]/400)+10, label='Minimal Temperature Error', color='r', alpha=0.5 )  
ax5.set_xlim(200, 700)
ax5.legend(loc='best', prop={'size': 6})
ax5.set_title('Trial 3 - Uncertainties of Optimal Fitting Parameters')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Temperature ($^\circ$C)')

#ax6.plot(trial3[0], trial3[1]-fitting_function(trial3[0], *popt3) -13*np.sin(trial3[0]/400)+10, label='Curve Fit Difference')
ax6.errorbar(trial3[0], np.zeros(len(trial3[0])), xerr = time_uncertainty, yerr=trial3_error,  label='Temperature $T_I$ Error Overlap', color='k', fmt='none', capsize=1, lw=0.5) #add our errorbars
ax6.plot(trial3[0], trial3[1]-n3_fit_maximalerror, label='Maximal Error Difference', color='#FF4000')
ax6.plot(trial3[0], trial3[1]-n3_fit_minimalerror, label='Minimal Error Difference', color='#FFBF00')
ax6.set_xlim(200, 700)
ax6.legend(loc='best', prop={'size': 6})
ax6.set_title('Trial 3 - Overlap of Parameter Uncertainty with Data Error')
ax6.set_xlabel('Time (s)')


plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.show()