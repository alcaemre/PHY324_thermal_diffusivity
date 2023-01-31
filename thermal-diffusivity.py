import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    trial1 = get_wave_from_csv(r'trial1.csv')
    # trial2 = get_wave_from_csv(r'trial2.csv')
    # trial3 = get_wave_from_csv(r'trial3.csv')
    # print(trial1)

    fig, (ax1,ax2, ax3) = plt.subplots(3,1)
    ax2.set_ylabel('temperature')
    ax3.set_xlabel('time')

    axs= [ax1, ax2, ax3]

    for i in range(3):
        trial_label = rf'trial{i+1}'
        # print(trial_label)
        axs[i].set_title(trial_label)

        trial = get_wave_from_csv(trial_label+'.csv')
        # print(trial)
        axs[i].plot(trial[0], trial[1], label='T_I')
        axs[i].plot(trial[0], trial[2], label='T_S')
        axs[i].legend(loc=1)
        

plt.show()
