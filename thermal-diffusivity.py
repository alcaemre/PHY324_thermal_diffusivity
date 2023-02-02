import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.special as sp
import itertools
import threading
import time
import sys



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


def J_0 (x, A, f, offset):
    """real componant of the bessel function

    Args:
        x (np.array): x data
        A (float): amplitude
        f (float): frequency
    """
    k = 50
    J = 0
    for i in range(k):
        J += (((-1)**i)/(sp.factorial(i))**2)*((x**2)/(4*f))**i
    J = A*J + offset
    return J

def ber_0(x, A, f, offset):
    """real componant of the bessel function

    Args:
        x (np.array): x data
        A (float): amplitude
        f (float): frequency
    """
    k = 10
    ber = 0
    for i in range(k):
        ber += ((np.cos((i * np.pi)/2))/(sp.factorial(i))**2)*((x**2)*f)**i     # calculate bessel function, f = m/(omega*(r**2))
    ber = (A*ber) + offset                                                      # multiple the bessel function by some amplitude (A) and add some vertical offset (offset)
    return ber



def ber_i(x, A, f, offset):
    """real componant of the bessel function

    Args:
        x (np.array): x data
        A (float): amplitude
        f (float): frequency
    """
    k = 10
    ber = 0
    for i in range(k):
        ber += ((np.sin((i * np.pi)/2))/(sp.factorial(i))**2)*((x**2)*f)**i     # calculate bessel function, f = m/(omega*(r**2))
    ber = (A*ber) + offset                                                      # multiple the bessel function by some amplitude (A) and add some vertical offset (offset)
    return ber


def find_optimal_params(trial):
    A_s = np.linspace(0, 20, 20)
    f_s = np.linspace(10, 100000, 20)
    offset_s = np.linspace(20, 70, 10)
    chars = "/—\|"
    chars_i = 0
    
    p0s = {}
    perrs = []
    for A in A_s:
        for f in f_s:
            for offset in offset_s:

                sys.stdout.write('\r'+'calculating . . .  '+chars[chars_i % 4]) # animating output to show that function is running
                chars_i += 1
                
                popt, pcov = op.curve_fit(J_0, trial[0], trial[1])
                perr = np.sqrt(np.diag(pcov))[0]
                # print(perr)
                perrs.append(perr)
                p0s[str(perr)]=(A, f, offset)

                sys.stdout.flush() # clearing animation
    
    return p0s[str(np.min(perrs))]


if __name__ == "__main__":

    plotting = True
    optimizing = False

    # trial1 = get_wave_from_csv(r'trial1.csv')
    # trial2 = get_wave_from_csv(r'trial2.csv')
    # trial3 = get_wave_from_csv(r'trial3.csv')

    if plotting:
        fig, (ax1,ax2, ax3) = plt.subplots(3,1)
        ax2.set_ylabel('temperature')
        ax3.set_xlabel('time')

        axs= [ax1, ax2, ax3]

<<<<<<< HEAD
        p0s=[(15.0, 400.0, 25.0), (20.0, 45.0, 70.0), (15, 400, 65)]

        for i in range(3):
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

        axs[0].set_title('trial 1')
        axs[0].plot(trial1[0], trial1[1], label='T_I')
        axs[0].plot(trial1[0], trial1[2], label='T_S')
        popt1, pcov1 = op.curve_fit(J_0, trial1[0], trial1[1], p0=p0s[0])
            # print(popt)
            # print(pcov3**2)
        axs[0].plot(trial1[0], (1.2**(trial1[0]/700))*J_0(trial1[0], popt1[0], popt1[1], 0)+20*np.sin(trial1[0]/400)-14+popt1[2], label='bessel fit')
        axs[0].legend(loc='upper right')
=======
        p0s=[(20.0, 60000.0, 70.0), (20.0, 45.0, 70.0), (10, 60, 50)]

        for i in range(3):
            trial_label = rf'trial{i+1}'
            # print(trial_label)
            axs[i].set_title(trial_label)

            trial = get_wave_from_csv(trial_label+'.csv')
            # print(trial)
            axs[i].plot(trial[0], trial[1], label='T_I')
            axs[i].plot(trial[0], trial[2], label='T_S')
            popt, pcov = op.curve_fit(J_0, trial[0], trial[1], p0=p0s[i])
            # print(popt)
            print(pcov**2)
            axs[i].plot(trial[0], J_0(trial[0], popt[0], popt[1], popt[2]), label='bessel function')
>>>>>>> changes to text and py fitting


<<<<<<< HEAD
    #sinusoidal fitting attempt
        axs[2].set_title('trial 3 - sinusoidal superposition')
        axs[2].plot(trial3[0], trial3[1], label='T_I')
        axs[2].plot(trial3[0], trial3[2], label='T_S')
        popt3, pcov3 = op.curve_fit(J_0, trial3[0], trial3[1], p0=p0s[2]) #curve_fit
            # print(popt)
            # print(pcov3**2)
        #plotting the curve fit bessel function
        axs[2].plot(trial3[0], (2**(trial3[0]/700))*J_0(trial3[0], popt3[0], popt3[1], 0)-20*np.sin(trial3[0]/400)+popt3[2]+10, label='bessel fit')
        axs[2].legend(loc='upper right')




=======
>>>>>>> changes to text and py fitting
        plt.subplots_adjust(hspace=0.5)
        plt.show()




    #try to initialize input from another phase difference

<<<<<<< HEAD
# xx=np.linspace(30, 300, 100)
# fig, (ax1) = plt.subplots(1,1)
# # ax1.plot(xx, J_0(xx, 2, 60), label='bessel function')
# ax1.legend(loc='upper right')
# ax1.set_xlim(0, 300)
=======




xx=np.linspace(30, 300, 100)
fig, (ax1) = plt.subplots(1,1)
ax1.plot(xx, J_0(xx, 2, 60), label='bessel function')
ax1.legend(loc='upper right')
ax1.set_xlim(0, 300)
>>>>>>> changes to text and py fitting

plt.show()