import numpy as np
import matplotlib.pyplot as plt

#funkcja napięcia baterii w zależności od stanu naładowania
def battery_voltage(soc):

    #przykladowa funkcja sigmoidalna
    a = 12
    b = 0.2
    c = 50
    return a / (1 + np.exp( -b * (soc -c)))

#zakres wartosci stanu naladowania (0 - 100)
soc_values = np.linspace(0, 100, 100)

#obliczenie napiecia baterii dla kazdej wartosci soc
voltage_values = [battery_voltage(soc) for soc in soc_values]

#wykres
plt.figure(figsize=(8,6))
plt.plot(soc_values, voltage_values, label='Napiecie baterii')
plt.title('Wykres napiecia baterii w zaleznosci od stanu naladowania')
plt.xlabel('Stan naladowania(%)')
plt.ylabel('Napiecie baterii (V)')
plt.grid(True)
plt.legend()
plt.show()