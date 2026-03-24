import jax
import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cuda")
# %env XLA_PYTHON_CLIENT_MEM_FRACTION=.95

jax.clear_caches()

num_ensemble = 100

num_agents = 100
num_buildings = 3
num_dorms = 3

num_days    = 100             # total number of days simulated
schedule    = [5,5,5,5,5,0,0] # Mon-Sun number of class periods per day
num_classes = 3               # number of classes an agent attends per day
off_campus  = 0.5             # portion of agents living off campus
bs          = [0.2,0.3]       # range of building environmental risk factors
ds          = [0.2,0.3]       # range of dorm environmental risk factors
init_inf    = 0.05            # probability of initial agents being infected
p           = 0.05            # infectious susceptibility
q           = 1/5             # probability of infectivity (E -> I)
r           = 1/10            # probability of recovery (I -> R)
off_inf     = 0.001           # probability of off-campus students to get infected
spn_inf     = 0.0001          # spontaneous infection rate

num_tests = 10   # number of tests available per day
num_bits  = 2    # number of bits tested; 0 for neither, 1 for first bit, 2 for both
alpha     = 0.05 # probability of first bit false positive
beta      = 0.05 # probability of first bit false negative
gamma     = 0.05 # probability of second bit false positive
delta     = 0.05 # probability of second bit false negative
N_p   = int(4e6) # number of particles

key = random.key(0)
key2 = random.split(key)[0]
keys = random.split(key2,[N_p])

# variables for data graphs
sus_plt = jnp.zeros((num_ensemble,num_days + 1))
exp_plt = jnp.zeros((num_ensemble,num_days + 1))
inf_plt = jnp.zeros((num_ensemble,num_days + 1))
rec_plt = jnp.zeros((num_ensemble,num_days + 1))
Sus = jnp.zeros((num_ensemble,num_days + 1))
Exp = jnp.zeros((num_ensemble,num_days + 1))
Inf = jnp.zeros((num_ensemble,num_days + 1))
Rec = jnp.zeros((num_ensemble,num_days + 1))
MSE = jnp.zeros((num_ensemble,num_days + 1))
emp_test_err = jnp.zeros((num_ensemble,num_days))
bkf_test_err = jnp.zeros((num_ensemble,num_days))
bkf_total_err = jnp.zeros((num_ensemble,num_days))
bkf_count_err = jnp.zeros((num_ensemble,num_days))

# simulation tools
if num_bits not in {0,1,2}:
  raise ValueError("num_bits must be in {0,1,2}")
num_tests = min(num_tests,num_agents)
num_time = sum(schedule)
schedule_vec = jnp.concat((jnp.zeros(1,dtype=int),jnp.cumsum(jnp.array(schedule))))
test_vec = jnp.arange(num_agents)
part_vec = jnp.arange(N_p) / N_p
selection_vec = jnp.array([[0,0],[1,0],[1,1],[0,1]],dtype=jnp.uint8)

# ground truth update function
@jax.jit
def f(x: jax.Array, bs: jax.Array, key: jax.Array) -> jax.Array:
    # extract agent states
    sus = (1-x[:,0]) * (1-x[:,1])
    exd =    x[:,0]  * (1-x[:,1])
    inf =    x[:,0]  *    x[:,1]
    rec = (1-x[:,0]) *    x[:,1]

    # compute P(S -> E)
    bp = c * (1-(1-p)**(bs @ inf))
    pi = jnp.prod(1-jnp.multiply(jnp.reshape(bs,(-1,num_agents)).T,bp.flatten()),-1)
    dp = e * (1-(1-p)**(d @ inf))
    pi *= jnp.prod(1-jnp.multiply(d.T,dp),-1)
    pi *= 1 - jnp.multiply(o,off_inf)
    pi *= 1 - spn_inf
    pi = 1 - pi

    # compute P(S -> E), P(E -> I), P(I -> R)
    P0 = (1 - pi) * sus
    P1 = pi * sus + (1 - q) * exd
    P2 = q * exd + (1 - r) * inf
    P3 = r * inf + rec
    P  = jnp.column_stack((P0, P1, P2, P3))

    # return updated agent states
    key = random.split(key)[0]
    return selection_vec[random.categorical(key,jnp.log(P))], key

# particle first stage weight function
@jax.jit
def Vk(x: jax.Array, bs: jax.Array, tests: jax.Array, y1: jax.Array, y2: jax.Array) -> jax.Array:
    # extract agent states
    sus = (1-x[:,0]) * (1-x[:,1])
    exd =    x[:,0]  * (1-x[:,1])
    inf =    x[:,0]  *    x[:,1]
    rec = (1-x[:,0]) *    x[:,1]

    # compute P(S -> E)
    bp = c * (1-(1-p)**(bs @ inf))
    pi = jnp.prod(1-jnp.multiply(jnp.reshape(bs,(-1,num_agents)).T,bp.flatten()),-1)
    dp = e * (1-(1-p)**(d @ inf))
    pi *= jnp.prod(1-jnp.multiply(d.T,dp),-1)
    pi *= 1 - jnp.multiply(o,off_inf)
    pi *= 1 - spn_inf
    pi = 1 - pi

    # compute P(S -> E), P(E -> I), P(I -> R)
    P0 = (1 - pi) * sus
    P1 = pi * sus + (1 - q) * exd
    P2 = q * exd + (1 - r) * inf
    P3 = r * inf + rec
    P  = jnp.column_stack((P0, P1, P2, P3))

    # compute likelihood
    ny1 = 1-y1
    ny2 = 1-y2
    if num_bits == 2:
      L0 = ny1*ny2*(1-alpha)*(1-gamma) + y1*ny2*alpha*(1-gamma) + ny1*y2*(1-alpha)*gamma + y1*y2*alpha*gamma
      L1 = ny1*ny2*beta*(1-gamma) + y1*ny2*(1-beta)*(1-gamma) + ny1*y2*beta*gamma + y1*y2*(1-beta)*gamma
      L2 = ny1*ny2*beta*delta + y1*ny2*(1-beta)*delta + ny1*y2*beta*(1-delta) + y1*y2*(1-beta)*(1-delta)
      L3 = ny1*ny2*(1-alpha)*delta + y1*ny2*alpha*delta + ny1*y2*(1-alpha)*(1-delta) + y1*y2*alpha*(1-delta)
      L  = jnp.column_stack((L0, L1, L2, L3))
    elif num_bits == 1:
      L0 = ny1*(1-alpha) + y1*alpha
      L1 = ny1*beta + y1*(1-beta)
      L  = jnp.column_stack((L0, L1, L1, L0))
    else:
      L = jnp.ones((num_tests,4))

    # compute particle update posterior and first stage weight
    Q = P.at[tests].set(P[tests] * L)
    V = jnp.sum(jnp.log(jnp.sum(Q[tests],1)))

    return Q, V

# vectorize first stage weight function
Vk_vec = jax.jit(jax.vmap(Vk,(0,None,None,None,None)))

# particle state selection function
@jax.jit
def f_samp(x: jax.Array, Q, key):
    key = random.split(key)[0]
    return selection_vec[random.categorical(key,jnp.log(Q))], key

# vectorized particle state selection function
f_samp_vec = jax.jit(jax.vmap(f_samp,(0,0,0)))

# particle state update function
@jax.jit
def f_k(x: jax.Array, bs: jax.Array, tests: jax.Array, y1: jax.Array, y2: jax.Array, keys: jax.Array, key: jax.Array) -> jax.Array:
    Q_k, V_k1 = Vk_vec(x, bs, tests, y1, y2)
    mx = jnp.max(V_k1)
    V_k = jnp.exp(V_k1 - mx)
    V_k /= jnp.sum(V_k)
    key = random.split(key)[0]
    zeta_k = jnp.digitize(random.uniform(key) / N_p + part_vec, bins = jnp.cumsum(V_k))
    X, keys = f_samp_vec(x[zeta_k], Q_k[zeta_k], keys)
    return X, keys, key

# noisy observations
@jax.jit
def observation(tests: jax.Array, x: jax.Array, key: jax.Array) -> jax.Array:
    key = random.split(key)[0]
    r1 = random.uniform(key,[num_tests]) < alpha

    key = random.split(key)[0]
    r2 = random.uniform(key,[num_tests]) > beta

    key = random.split(key)[0]
    r3 = random.uniform(key,[num_tests]) < gamma

    key = random.split(key)[0]
    r4 = random.uniform(key,[num_tests]) > delta

    t1 = x[tests,0]
    t2 = x[tests,1]

    return (1-t1) * r1 + t1 * r2, (1-t2) * r3 + t2 * r4, key

for en in tqdm(range(num_ensemble), position=0, leave=True):
    # initialize agents
    agents = jnp.zeros((num_agents,2),dtype=jnp.uint8)
    key = random.split(key)[0]
    indx = random.choice(key,test_vec,shape=(int(init_inf * num_agents),),replace=False)
    agents = agents.at[indx].set(1)
    sus_plt = sus_plt.at[en,0].set(jnp.sum((1-agents[:,0]) * (1-agents[:,1])))
    exp_plt = exp_plt.at[en,0].set(jnp.sum(   agents[:,0]  * (1-agents[:,1])))
    inf_plt = inf_plt.at[en,0].set(jnp.sum(   agents[:,0]  *    agents[:,1]))
    rec_plt = rec_plt.at[en,0].set(jnp.sum((1-agents[:,0]) *    agents[:,1]))

    # generate schedule
    b = jnp.zeros((num_time,num_buildings,num_agents),dtype=jnp.uint8)
    @jax.jit
    def attendence(key, s):
        key = random.split(key)[0]
        indx = random.choice(key,s,shape=(num_classes,),replace=False)
        a = jnp.zeros(schedule[j]*num_buildings,dtype=jnp.uint8)
        a = a.at[indx].set(1).reshape(schedule[j],num_buildings)
        return a
    attendence_vec = jax.jit(jax.vmap(attendence,(0,None),2))
    for j in range(len(schedule)):
      if schedule[j] != 0:
        s = attendence_vec(random.split(key,[num_agents]),jnp.arange(int(schedule[j]*num_buildings)))
        b = b.at[schedule_vec[j]:schedule_vec[j+1],:].set(s)
    key = random.split(key)[0]
    c = random.uniform(key,minval=bs[0],maxval=bs[1],shape=(num_buildings,))
    key = random.split(key)[0]
    d = jnp.eye(num_dorms,dtype=jnp.uint8)[random.choice(key,jnp.arange(num_dorms),shape=(num_agents,))]
    key = random.split(key)[0]
    e = random.uniform(key,minval=ds[0],maxval=ds[1],shape=(num_dorms,))
    key = random.split(key)[0]
    o = jnp.zeros(num_agents,dtype=jnp.uint8)
    indx = random.choice(key,test_vec,shape=(int(off_campus*num_agents),),replace=False)
    o = o.at[indx].set(1)
    d = d.at[jnp.where(o)[0]].set(0).T

    #initialize particles, 0,0 = S, 1,0 = E, 1,1 = I, 0,1 = R
    @jax.jit
    def sample(vec, key):
      key = random.split(key)[0]
      indx = random.choice(key,test_vec,shape=(int(init_inf * num_agents),),replace=False)
      x = jnp.zeros((num_agents,2),dtype=jnp.uint8)
      x = x.at[indx].set(1)
      return x, key
    sample_vec = jax.jit(jax.vmap(sample,(None,0)))
    x_k, keys = sample_vec(test_vec, keys)
    z_k = jnp.mean(x_k,0)
    Sus = Sus.at[en,0].set(jnp.round(jnp.mean(jnp.sum((1-x_k[:,:,0]) * (1-x_k[:,:,1]),1))))
    Exp = Exp.at[en,0].set(jnp.round(jnp.mean(jnp.sum(   x_k[:,:,0]  * (1-x_k[:,:,1]),1))))
    Inf = Inf.at[en,0].set(jnp.round(jnp.mean(jnp.sum(   x_k[:,:,0]  *    x_k[:,:,1] ,1))))
    Rec = Rec.at[en,0].set(jnp.round(jnp.mean(jnp.sum((1-x_k[:,:,0]) *    x_k[:,:,1] ,1))))
    MSE = MSE.at[en,0].set(jnp.sum(jnp.max(jnp.minimum(z_k,1-z_k),1)))

    indx1 = 0
    indx2 = 0
    for i in tqdm(range(num_days),position=1,leave=False):
        builds = b[indx1:indx1+schedule[indx2]]
        agents, key = jax.block_until_ready(f(agents,builds,key))
        sus_plt = sus_plt.at[en,i+1].set(jnp.sum((1-agents[:,0]) * (1-agents[:,1])))
        exp_plt = exp_plt.at[en,i+1].set(jnp.sum(   agents[:,0]  * (1-agents[:,1])))
        inf_plt = inf_plt.at[en,i+1].set(jnp.sum(   agents[:,0]  *    agents[:,1]))
        rec_plt = rec_plt.at[en,i+1].set(jnp.sum((1-agents[:,0]) *    agents[:,1]))

        key2 = random.split(key2)[0]
        tests = random.choice(key2,test_vec,shape=(num_tests,),replace=False)
        y_k1, y_k2, key2 = jax.block_until_ready(observation(tests,agents,key2))
        x_k, keys, key2 = jax.block_until_ready(f_k(x_k, builds, tests, y_k1, y_k2, keys, key2))

        # update estimator values
        z_k = jnp.mean(x_k,0)
        Z_k = jnp.round(z_k)
        Sus = Sus.at[en,i+1].set(jnp.round(jnp.mean(jnp.sum((1-x_k[:,:,0]) * (1-x_k[:,:,1]),1))))
        Exp = Exp.at[en,i+1].set(jnp.round(jnp.mean(jnp.sum(   x_k[:,:,0]  * (1-x_k[:,:,1]),1))))
        Inf = Inf.at[en,i+1].set(jnp.round(jnp.mean(jnp.sum(   x_k[:,:,0]  *    x_k[:,:,1] ,1))))
        Rec = Rec.at[en,i+1].set(jnp.round(jnp.mean(jnp.sum((1-x_k[:,:,0]) *    x_k[:,:,1] ,1))))
        MSE = MSE.at[en,i+1].set(jnp.sum(jnp.max(jnp.minimum(z_k,1-z_k),1)))

        # update error vectors
        emp_test_err = emp_test_err.at[en,i].set(1 - jnp.sum(jnp.prod(jnp.column_stack((y_k1,y_k2)) == agents[tests],1)) / num_tests)
        bkf_test_err = bkf_test_err.at[en,i].set(1 - jnp.sum(jnp.prod(Z_k[tests] == agents[tests],1)) / num_tests)
        bkf_total_err = bkf_total_err.at[en,i].set(1 - jnp.sum(jnp.prod(Z_k == agents,1)) / num_agents)
        bkf_count_err = bkf_count_err.at[en,i].set(jnp.abs(exp_plt[en,i+1] + inf_plt[en,i+1] - Exp[en,i+1] - Inf[en,i+1]))

        # update schedule indices
        indx1 += schedule[indx2]
        indx2 += 1
        if indx2 % len(schedule) == 0:
          indx1 = 0
          indx2 = 0

sus_mean = jnp.mean(sus_plt,0)
sus_std = jnp.std(sus_plt,0)
inf_mean = jnp.mean(exp_plt+inf_plt,0)
inf_std = jnp.std(exp_plt+inf_plt,0)
rec_mean = jnp.mean(rec_plt,0)
rec_std = jnp.std(rec_plt,0)

print()
print('Average Outbreak Size = ', jnp.mean(inf_mean))
print('Average Outbreak Peak = ', jnp.mean(jnp.max(exp_plt+inf_plt,1)))
print('Average Ground Truth test error = ', jnp.mean(emp_test_err))
print('Average BKF test error = ', jnp.mean(bkf_test_err))
print('Average BKF total error = ', jnp.mean(bkf_total_err))
print('Average E+I BKF relative L1 error = ', jnp.mean(jnp.mean(bkf_count_err,1) / jnp.mean(exp_plt + inf_plt,1)))
print('Ground Truth within MSE = ', jnp.mean(jnp.logical_and(Exp + Inf - MSE <= exp_plt + inf_plt, exp_plt + inf_plt <= Exp + Inf + MSE)))

# generate SIR plot
fig = plt.figure()
plt.plot(sus_mean, color='g', label='S')
plt.fill_between(jnp.arange(101),sus_mean-2*sus_std,sus_mean+2*sus_std,alpha=0.2,color='g')
plt.plot(inf_mean, color='r', label='E+I')
plt.fill_between(jnp.arange(101),inf_mean-2*inf_std,inf_mean+2*inf_std,alpha=0.2,color='r')
plt.plot(rec_mean, color='b', label='R')
plt.fill_between(jnp.arange(101),rec_mean-2*rec_std,rec_mean+2*rec_std,alpha=0.2,color='b')
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.legend()
plt.show()
