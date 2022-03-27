from numpy import poly1d,polyfit,array,arange
from scipy.signal import butter,lfilter

def BaselineCorrection(acceleration,dt,order):
    time = arange(0,len(acceleration)*dt,dt)
    constants = polyfit(time,acceleration,order)
    f = poly1d(constants)
    predicted = f(time)
    return array(acceleration) - predicted

def Filtering(data,dt=0.0005,lowcut=0,highcut = 0,order=4):
    #filter_configuration : low, high, bandpass, bandstop
    nyq = 0.5 / dt
    low = lowcut / nyq
    high = highcut / nyq
    cutoff = [low,high]

    b,a = butter(order, cutoff, btype="bandpass")

    y = lfilter(b, a, data)

    return y
