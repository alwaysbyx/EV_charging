import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
import random


def charging_session(n): #Simulate Charging session from distribution
    norm_params = np.array([[5, 1],
                            [1, 1.3],
                            [9, 1.3]])
    n_components = norm_params.shape[0]
    # Weight of each component, in this case all of them are 1/3
    weights = np.ones(n_components, dtype=np.float64) / 3.0
    # A stream of indices from which to choose the component
    mixture_idx = numpy.random.choice(len(weights), size=n, replace=True, p=weights)
    print(np.shape(mixture_idx))
    # y is the mixture sample
    y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                       dtype=np.float64)

    # Theoretical PDF plotting -- generate the x and y plotting positions
    xs = np.linspace(y.min(), y.max(), 200)
    ys = np.zeros_like(xs)
    print(np.shape(ys))

    for (l, s), w in zip(norm_params, weights):
        ys += ss.norm.pdf(xs, loc=l, scale=s) * w

    plt.plot(xs, ys)
    plt.hist(y, normed=True, bins="fd")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    arr_time=ys
    dep_time=xs

    return arr_time, dep_time

def charging_session_individual(lambda_val, vehicle_capacity): #Simulate Charging session from distribution
    arr_time=(np.random.poisson(lam=lambda_val)+5.0)*12.0+np.random.randint(0.0, 12.0)
    sess_time=np.random.randint(1.0, 15.0)*12.0+np.random.randint(0.0, 12.0)
    init_cap=np.random.uniform(0.0, vehicle_capacity-1.0)
    req_energy=np.random.uniform(init_cap+0.5, vehicle_capacity)
    return arr_time, sess_time, init_cap, req_energy

def generate_dataset(num_step=96, total_vehicles=100, battery_capacity = 8, lam=9, seed=0, soc=0, arrival_flag='poisson'):
    np.random.seed(seed)
    if arrival_flag == 'poisson':
        arrival_time= np.random.poisson(lam=lam, size=(total_vehicles,))
        arrival_time = np.sort(arrival_time)*4.0
        arrival_time = arrival_time + np.random.randint(0,4, size=(total_vehicles,))
        arrival_time = np.sort(arrival_time)
        depart_time = np.random.randint(6, 36, size=(total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)
    elif arrival_flag == 'uniform':
        arrival_time = np.random.randint(1,50,(total_vehicles,))
        arrival_time = np.sort(arrival_time)
        depart_time = np.random.randint(6,36, size=(total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)
    elif arrival_flag == 'fix':
        arrival_time = np.random.randint(1,2,(total_vehicles,))
        arrival_time = np.sort(arrival_time)
        depart_time = np.random.randint(6,36, size=(total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)

    if soc==0:
        initial_state=np.random.uniform(0.8, 4.0, size=(total_vehicles,))
    else:
        initial_state=np.random.uniform(soc, soc+0.2, size=(total_vehicles,))
    required_energy=np.random.uniform(2.0, 6.0, size=(total_vehicles,))
    final_energy = np.min((initial_state+required_energy, np.ones_like(initial_state)*battery_capacity), axis=0)
    required_energy = np.round(final_energy-initial_state, 2)
    
    dataset = {'arrival_time':arrival_time, 'depart_time':depart_time, 'initial_state':initial_state, 'required_energy': required_energy, 'final_energy':final_energy}
    
    return dataset


def generate_dataset_test(num_step=96, total_vehicles=100, battery_capacity = 8, lam=12, seed=0, soc=0, arrival_flag='poisson'):
    np.random.seed(seed)
    if arrival_flag == 'poisson':
        arrival_time= np.random.poisson(lam=lam, size=(total_vehicles,))
        arrival_time = np.sort(arrival_time)*4.0
        arrival_time = arrival_time + np.random.randint(0,4, size=(total_vehicles,))
        arrival_time = np.sort(arrival_time)
        depart_time = 30*np.ones((total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)
    elif arrival_flag == 'normal':
        arrival_time = np.array([i for i in range(1,11)] + [i for i in range(1,11)]+ [i for i in range(2,8)] + [i for i in range(2,8)] + [i for i in range(3,7)] + [i for i in range(3,7)]+ [i for i in range(3,7)] + [5 for _ in range(6)])
        #arrival_time = np.array([i for i in range(1,26)] + [i for i in range(8,19)] + [i for i in range(10,17)] + [i for i in range(11,16)] + [12 for _ in range(2)])
        arrival_time = np.sort(arrival_time)
        depart_time = 30*np.ones((total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)
    elif 'uniform' in arrival_flag:
        arrival_time = np.array([i for i in range(1,11) for t in range(total_vehicles//10)])
        arrival_time = np.sort(arrival_time)
        depart_time = 30*np.ones((total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)
    elif arrival_flag == 'fix2':
        arrival_time = np.array([1 for _ in range(total_vehicles//2)] + [10 for _ in range(total_vehicles//2)])
        arrival_time = np.sort(arrival_time)
        depart_time = 30*np.ones((total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)
    elif arrival_flag == 'fix1':
        arrival_time = np.array([1 for _ in range(total_vehicles)])
        arrival_time = np.sort(arrival_time)
        depart_time = 30*np.ones((total_vehicles,))
        depart_time = np.min((arrival_time+depart_time, np.ones_like(arrival_time)*num_step), axis=0)

    if soc==0: # fix 50%
        initial_state = battery_capacity/2 * np.ones((total_vehicles,))
    else:
        low, high = 50 - 5*soc, 50 + 5*soc
        initial_state = list(np.linspace(low/100*battery_capacity, high/100*battery_capacity, total_vehicles))
        random.shuffle(initial_state)
    #required_energy=np.random.uniform(2.0, 6.0, size=(total_vehicles,))
    final_energy = battery_capacity * np.ones((total_vehicles))
    required_energy = np.round(final_energy-np.array(initial_state), 2)
    
    dataset = {'arrival_time':arrival_time, 'depart_time':depart_time, 'initial_state':initial_state, 'required_energy': required_energy, 'final_energy':final_energy}
    
    return dataset

def generate_soc(low, high, num, battery_capacity):
    count = num // (high-low+1)
    soc = []
    if count != 0:
        for i in range(low, high+1):
            soc.extend([i/100*battery_capacity for _ in range(count)])
        leave = num-len(soc)
        soc.extend([0.5*battery_capacity for _ in range(leave)])
        return np.array(soc)
    else:
        soc = np.linspace(low/100*battery_capacity, high/100*battery_capacity, num)
        return soc