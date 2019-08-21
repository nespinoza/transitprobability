import numpy as np
import utils

nsamples = 1000

# Mutual inclination:
DeltaI,DeltaI_sigma_up,DeltaI_sigma_down = 2.0,4.0,2.0 

# Inclination of transiting planet, in degrees:
inc,inc_sigma_up,inc_sigma_down = 89.12,0.37,0.31

# Semi-major axis (AU) of planet we know does not transit (planet c):
ac,ac_sigma_up,ac_sigma_down = 0.061,0.004,0.004

# Semi-major axis of non-transiting planet (AU):
a,a_sigma_up,a_sigma_down = 0.204,0.015,0.015

# Radius of star (solar radii):
R,R_sigma_up,R_sigma_down = 0.337,0.015,0.015

ptransit = np.array([])


for i in range(1000):
    DeltaI_samples = utils.sample_from_errors(DeltaI,DeltaI_sigma_up,DeltaI_sigma_down, nsamples, low_lim = 0., up_lim = 90.)
    inc_samples = utils.sample_from_errors(inc, inc_sigma_up,inc_sigma_down, nsamples, low_lim = 0., up_lim = 90.)
    a_samples = utils.sample_from_errors(a,a_sigma_up,a_sigma_down, nsamples, low_lim = 0.,up_lim = 1000.)
    R_samples = utils.sample_from_errors(R,R_sigma_up,R_sigma_down, nsamples, low_lim = 0.,up_lim = 1000.)
    ac_samples = utils.sample_from_errors(ac,ac_sigma_up,ac_sigma_down, nsamples, low_lim = 0.,up_lim = 1000.)
    # Compute R*/a:
    Ra = (R_samples * 6.957e5)/(a_samples * 149597870.7)
    # Same for c:
    Rac = (R_samples * 6.957e5)/(ac_samples * 149597870.7)
    # Generate possible target inclinations:
    target_inc = np.zeros(nsamples)
    incc = np.zeros(nsamples)
    for j in range(nsamples):
        target_inc[j],incc[j] = np.random.normal(inc_samples[j],DeltaI_samples[j],size=2)
    # Identify inclinations larger than 90, those are the same as inclinations 180 - i:
    idx = np.where(target_inc>90.)[0]
    target_inc[idx] = 180. - target_inc[idx]
    # Same for c:
    idx = np.where(incc>90.)[0]
    incc[idx] = 180. - incc[idx]
    # Compute probability of transit. d has to transit, but c does not:
    ntransit = np.where((np.cos(target_inc*np.pi/180.)<Ra)&(np.cos(incc*np.pi/180.)>Rac))[0]
    ptransit = np.append(ptransit,(np.double(len(ntransit))/np.double(nsamples))*100)

# Print probability:
p,pu,pl = utils.get_quantiles(ptransit)
print('Probability if in 0.5 systems with +- 2 degrees:',p,'+',pu-p,'-',p-pl)

# Cut by half to account for the possibility that the system is part of the 50% of large-mutual-inclination 
# systems:
ptransit = ptransit/2.

# Get quantiles to report in paper:
p,pu,pl = utils.get_quantiles(ptransit)
print('Final probability:',p,'+',pu-p,'-',p-pl)


# Plot:
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc,rcParams

# Set seaborn contexts:
sns.set_context("talk")
sns.set_style("ticks")

# Fonts:
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size':12})
plt.rc('legend', **{'fontsize':7})

# Ticks to the outside:
rcParams['axes.linewidth'] = 1.2
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

plt.hist(ptransit,bins=15,normed=True,color = "cornflowerblue", lw=0, rwidth=0.9)
plt.xlabel('Transit probability')
plt.ylabel('PDF')
plt.tight_layout()
plt.savefig("fig.pdf", dpi=300)
plt.savefig("fig.eps", dpi=300)
plt.savefig("fig.png", dpi=300)
