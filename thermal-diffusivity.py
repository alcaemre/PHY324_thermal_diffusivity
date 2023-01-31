import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.special as sp



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


def J_0 (x, A, f):
    """real componant of the bessel function

    Args:
        x (np.array): x data
        A (float): amplitude
        f (float): frequency
    """
    k = 50
    J = 0
    for i in range(k):
        J = A*(((-1)**i)/(sp.factorial(i))**2)*((x**2)/(4*f))**i + ber
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
        ber += ((np.cos((i * np.pi)/2))/(sp.factorial(i))**2)*((x**2)/(4*f))**i     # calculate bessel function, f = omega*r**3/m
    ber = (A*ber) + offset                                                        # multiple the bessel function by some amplitude (A) and add some vertical offset (offset)
    return ber


if __name__ == "__main__":

    trial1 = get_wave_from_csv(r'trial1.csv')
    # trial2 = get_wave_from_csv(r'trial2.csv')
    # trial3 = get_wave_from_csv(r'trial3.csv')
    # print(trial1)

    fig, (ax1,ax2, ax3) = plt.subplots(3,1)
    ax2.set_ylabel('temperature')
    ax3.set_xlabel('time')

    axs= [ax1, ax2, ax3]

    p0s=[()]

    for i in range(3):
        trial_label = rf'trial{i+1}'
        # print(trial_label)
        axs[i].set_title(trial_label)

        trial = get_wave_from_csv(trial_label+'.csv')
        # print(trial)
        axs[i].plot(trial[0], trial[1], label='T_I')
        axs[i].plot(trial[0], trial[2], label='T_S')
        popt, pcov = op.curve_fit(ber_0, trial[0], trial[1], p0=(10, 600, 50))
        print(popt)
        axs[i].plot(trial[0], ber_0(trial[0], popt[0], popt[1], popt[2]), label='bessel function')

        axs[i].legend(loc=1)

plt.show()


xx=np.linspace(0, 30, 100)
fig, (ax1) = plt.subplots(1,1)
ax1.plot(xx, ber_0(xx, 1, 20, 50), label='bessel function')
ax1.legend(loc='upper right')
plt.show()