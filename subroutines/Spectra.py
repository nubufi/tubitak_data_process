from numpy import array,fft,argmax,abs,argmin

def peakFrequency(f,fa,F):
    index_min = (abs(f-(F-5))).argmin()
    index_max = (abs(f-(F+5))).argmin()
    peak = argmax(fa[index_min:index_max])
    if F<=30:
        return F
    else:
        return f[index_min:index_max][peak]

def FourierAmplitude(data, dt=0.0005):
    n = 1
    while n < len(data[0]): n *= 2
    delf = 1/(n*dt)
    f = array(range(1,int(n/2+2)))*delf
    idx = (abs(f - 200)).argmin()
    fa_list = [f[:idx]]
    for d in data:
        fftx = abs(fft.fft(d,n))
        fa = (fftx[0:int(n/2+1)])*dt
        fa_list.append(fa[:idx])
    
    return fa_list
