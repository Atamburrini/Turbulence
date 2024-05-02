#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.special import hermite
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq



# Definir los nombres de las columnas
#column_names = ['velocity']

# Leer el archivo 
df = pd.read_csv('ACE_merge64s.dat',delim_whitespace=True, header=0)

print(df)

#definir paso de tiempo y calcular una serie con este paso de tiempo y todos los elementos de velocidad
#dt = 1/37500
dt = 74
tiempo = pd.Series([i*dt for i in range(len(df))])

# Agregar la columna de tiempo al DataFrame
df['tiempo'] = tiempo

# Mostrar el DataFrame resultante
#print(df)

std_total= df['V'].std()

#df['velocity']=df['velocity']/std_total

 ###GRAFICO SERIE DE TIEMPO DE VELOCIDAD#######
plt.plot(df['tiempo'], df['V'])
plt.xlabel('t')
plt.ylabel('velocity')
plt.title('Veolicity in turbulent exp')
plt.legend()
plt.grid(True)
plt.show()

 ##########################################

### STATIONARITY CHECK ###################

tamaño_ventana = len(df) // 100

# Calcular la media y la desviación estándar en cada ventana móvil
media_ventana = df['V'].rolling(window=tamaño_ventana, min_periods=1,step=1).mean()
desviacion_ventana = df['V'].rolling(window=tamaño_ventana, min_periods=1,step=1).std()

# Agregar las medias y desviaciones estándar al DataFrame original
df['media_ventana'] = media_ventana
df['desviacion_ventana'] = desviacion_ventana

# Mostrar el DataFrame resultante
#print(df)

 ###GRAFICO SERIE DE TIEMPO DE VELOCIDAD#######
plt.plot(df['tiempo'], df['media_ventana'],label='media',color='red')
plt.plot(df['tiempo'], df['desviacion_ventana'],label='std',color='blue')
plt.xlabel('t')
plt.ylabel('Media, Std')
plt.title('Stationary check')
plt.legend(loc = 'upper right')
plt.grid(True)
plt.show()
 ##########################################
### STATIONARITY CHECK END ###################

###### AUTOCORRELACION ###########

tamaño_ventana = len(df) // 100  # Tamaño de la ventana móvil
dt_original = 74  # Intervalo de tiempo original

delta_t_values = [n * dt_original for n in range(0, tamaño_ventana + 1, 1)]
#print('termina de realizar la lista de deltas')
#print(len(delta_t_values))
autocorrelaciones = []
# normalizacion
#########TRASLADAR TODA LA FUNCION AL CERO 
#print(std_total)

# Calcular la función de autocorrelación para diferentes valores de Delta t

print('entra al for')
for delta_t in delta_t_values:
   # Desplazar la serie de velocidad en Delta t
    velocidad_desplazada = df['V'].shift(-int(delta_t / dt_original))
    
  #  Calcular el producto punto entre la serie de velocidad original y la desplazada
    producto_punto = df['V'] * velocidad_desplazada
    
    # Calcular la media sobre todo el tiempo t
    autocorrelacion = producto_punto.mean()/(std_total**2)
    autocorrelaciones.append(autocorrelacion)


autocorrelaciones_df = pd.DataFrame({'Delta_t': delta_t_values, 'Autocorrelacion': autocorrelaciones})

# Mostrar el DataFrame de autocorrelaciones
#print(autocorrelaciones_df)

# Graficar la autocorrelación versus el intervalo de tiempo Delta_t
plt.plot(autocorrelaciones_df['Delta_t'], autocorrelaciones_df['Autocorrelacion'])
plt.xlabel('Delta_t')
plt.ylabel('Autocorrelacion')
plt.title('Funcion de Autocorrelacion vs Delta_t')
plt.grid(True)
plt.show()

###### AUTOCORRELACION END ###########

####### POWER LAW ########

# Extraer la serie de velocidad
velocidad = df['V'].values

###SCIPY.SIGNAL.WELCH ### EXPLORAR ESTO PARA EL ESPECTRO 

# Calcular la Transformada Rápida de Fourier (FFT) de la velocidad
fft_velocidad = fft(velocidad)
fft_velocidad_cuadrado= np.abs(fft_velocidad) **2
# Calcular la frecuencia correspondiente a cada punto en la FFT
frecuencia = fftfreq(len(velocidad), d=dt)


# Trama de la magnitud al cuadrado de la FFT vs la frecuencia
plt.loglog(frecuencia, fft_velocidad_cuadrado)
plt.xlabel('Frecuencia')
plt.ylabel('Magnitud al Cuadrado de FFT')
plt.title('Transformada de Fourier al Cuadrado vs Frecuencia')
plt.grid(True)
plt.show()


###VISUAL STUDIO CODES WITH NOTEBOOK 