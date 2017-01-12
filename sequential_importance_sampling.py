from scipy.stats import beta
from scipy.stats import norm
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import random
import bisect
from math import *
import collections
from matplotlib.pyplot import *
def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

#weights=[0.3, 0.4, 0.3]
#population = 'ABC'
#counts = collections.defaultdict(int)
#for i in range(10000):
    #counts[choice(population, weights)] += 1
#print(counts)
####sample from the proposal distribution at time t=1
N=100
resample_particles=[]
particles=np.random.normal(1,2,N)## proposal distribution
target=mlab.normpdf(particles,-1,1)## calculate the target pdf
proposal=mlab.normpdf(particles,1,2)## calculate the proposal pdf
particles=particles.tolist()
w_t=target/proposal ## calculate the weights of the particles
#newparticles=np.random.choice(particles,N,w_t)
for i in range(100):
    particles_index=choice(particles, w_t)
    #print particles_index
    resample_particles.append(particles_index)
#print index
#print newparticles
#print w_t
## at time t=1:50 random walk and sample from new distributions
#print len(w_t)
a_update=[]

a=np.ones(N)*w_t
#print suma
#suma=w_t
#print len(a)
for t in range(5):
    particles_update=[]
    w=[]
    tar=[]
    #a=[]
    a_new=[]
    for i in range(N):
        #regenerate particles
        newparticles=np.random.normal(resample_particles[i],2,1)## generate new particles
        particles_update.extend(newparticles)
        proposal_update=mlab.normpdf(newparticles,resample_particles[i],2)
        target_update=mlab.normpdf(newparticles,-1,1)
        tar.extend(target_update)
    #print particles_update
        a[i]=a[i]*(target_update/(target[i]*proposal_update))
        #print a_update
        #a_update=w_t[i]*a
        #print a[i]
        a_new.append(a[i])

    #### reform the varibles
    #target=tar
    particles=particles_update
    target=tar
    a=a_new
    #w_t=w
#print a_update[1]



#print len(suma)
#print len(w_t)
#w=w.__iter__()
#w=w.tolist()
#sumw=w[99]

#print w
#w=w/sum(w)*1.0
#print w_t
#w=w/sumw
#print w
a=a/sum(a)
#print len(a)
mean=sum(a*particles)
x=np.linspace(-20,20,100)
plt.plot(x,mlab.normpdf(x,-6,1),'r',label='target distribution')
plt.plot(x,mlab.normpdf(x,1,2),'b',label='proposal distribution')
plt.legend()
vlines(particles,[0],a)
plt.show()
#samples=np.random.choice(particles,N,w_t)
#plt.hist(samples)
#list=[1.1412, 4.3453, 5.8709, 0.1314]
#k=w.index(min(w))
#print k
#print particles[k]
#l=[]
#for i in range(N):
    #l.append(w[i]*particles[i])
#s=sum(l)/N*1.0
#print s
