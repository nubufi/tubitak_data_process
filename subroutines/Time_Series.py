from scipy.integrate import cumtrapz
from numpy import insert,arange,diff

def fromAcc(acceleration,dt=0.0005):
    vels= []
    for acc in acceleration:
        vel = cumtrapz(acc*981,dx=dt)
        vel = insert(vel,0,vel[0])
        vels.append(vel)
    return vels

def fromVel(velocity, dt=0.0005):
    accs = []
    for vel in velocity:
        time = arange(0, len(vel) * dt, dt)
        acc = diff(vel) / diff(time)
        acc = insert(acc, 0, acc[0]) / 981
        accs.append(acc)
    return accs
