import numpy as np
import matplotlib.pyplot as plt

from kf import KF

kf = KF(initialX0=0, initialV0=1, accel_var=0.3)

DT = 0.1
STEPS = 1000
MEASUREMENT_STEP = 100
time = np.linspace(0, STEPS*DT, STEPS)

posReal =0
velocityReal = 0.9
mesVariance = 0.1**2
noise = np.random.randn()*np.sqrt(mesVariance)
mus = []
covs = []
positionsReal = []
velocitiesReal = []


for step in range(STEPS):
    mus.append(kf.mean)
    covs.append(kf.covariance)
    kf.predict(dt=DT)
    # Compute real values for plotting
    posReal = posReal + velocityReal*DT
    positionsReal.append(posReal)
    velocitiesReal.append(velocityReal)
    # update with measurements
    if step !=0 & step % MEASUREMENT_STEP == 0:
        kf.update(stateMeas=posReal + noise, measVar=mesVariance)


plt.ion()
plt.figure()


plt.subplot(2,1,1)
plt.grid()
plt.title('Position')
plt.ylabel('m')
plt.plot(time, [mu[0] for mu in mus], 'r')
plt.plot(time, [mu[0] + 2*np.sqrt(cov[0,0]) for mu,cov in zip(mus,covs)], 'r--')
plt.plot(time, [mu[0] - 2*np.sqrt(cov[0,0]) for mu,cov in zip(mus,covs)], 'r--')
# plot real position
plt.plot(time, [pos for pos in positionsReal], 'b--')

plt.subplot(2,1,2)
plt.grid()
plt.title('Velocity')
plt.xlabel('time [s]')
plt.ylabel('m/s')
plt.plot(time, [mu[1] for mu in mus], 'r')
plt.plot(time, [mu[1] + 2*np.sqrt(cov[1,1]) for mu,cov in zip(mus,covs)], 'r--')
plt.plot(time, [mu[1] - 2*np.sqrt(cov[1,1]) for mu,cov in zip(mus,covs)], 'r--')
# plot real velocity
plt.plot(time, [vel for vel in velocitiesReal], 'b--')

plt.show()
#plt.ginput(1)
