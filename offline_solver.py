from mimetypes import init
from xmlrpc.client import Boolean
import numpy as np
import csv
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import generate_dataset, generate_dataset_test
import argparse
import time



def aware_MPC(num_of_vehicles, timesteps, initial_states, max_power, terminal_states, arrival_time, dept_time, power_capacity, decay_rate):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal') # Requested end SoC for all vehicles
    x0 = cp.Parameter(num_of_vehicles, name='x0') # Initial SoC for all vehicles
    max_sum_u = cp.Parameter(name='max_sum_u') # Peak charging power for the infrastructure
    u_max = cp.Parameter(name='u_max') # Maximum charging power for each vehicle at each time step
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x') # charging power at each time step for each vehicle
    u = cp.Variable((num_of_vehicles, timesteps), name='u') # charging power at each time step for each vehicle

    x_terminal.value=terminal_states
    x0.value = initial_states
    max_sum_u.value = power_capacity
    u_max.value = max_power

    constr = [x[:,0] == x0,  x[:,-1] <= x_terminal]

    for i in range(num_of_vehicles):
        constr += [x[i,1:] == x[i,0:timesteps] + u[i,:], u[i,:] >= 0,]
        for t in range(timesteps):
            constr += [u[i, t] <= (u_max-decay_rate*x[i, t-1])*(t>=arrival_time[i]),
                    u[i, t] <= (u_max-decay_rate*x[i, t-1])*(t<dept_time[i])]
    obj = 0.
    for t in range (timesteps):
        constr += [
                cp.sum(u[0:num_of_vehicles,t]) <= power_capacity,
        ]
        obj += cp.sum(cp.log(u[:,t] + 0.0000001))

    obj -= cp.norm(x[:, -1] - x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()

    plt.plot(x.value[10])
    plt.plot(u.value[10])
    print(f'arrival time:{arrival_time[10]}')
    plt.show()

    return x.value, u.value

# invariant u: umax = umax
def standard_MPC(num_of_vehicles, timesteps, initial_states, max_power, terminal_states, arrival_time, dept_time, power_capacity, decay_rate,aware=False,log=True):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal') # Requested end SoC for all vehicles
    x0 = cp.Parameter(num_of_vehicles, name='x0') # Initial SoC for all vehicles
    max_sum_u = cp.Parameter(name='max_sum_u') # Peak charging power for the infrastructure
    if aware: 
        u_max = cp.Parameter(num_of_vehicles, name='u_max')
        initial_u_max = np.zeros((num_of_vehicles,))
        for i in range(num_of_vehicles):
            initial_u_max[i] = max_power - decay_rate*initial_states[i]
        u_max.value = initial_u_max
    else: 
        u_max = cp.Parameter(name='u_max') # Maximum charging power for each vehicle at each time step
        u_max.value = max_power 

    x = cp.Variable((num_of_vehicles, timesteps+1), name='x') # charging power at each time step for each vehicle
    u = cp.Variable((num_of_vehicles, timesteps), name='u') # charging power at each time step for each vehicle

    x_terminal.value=terminal_states
    x0.value = initial_states
    max_sum_u.value = power_capacity
    obj = 0
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for i in range(num_of_vehicles):
        constr += [x[i,1:] == x[i,0:timesteps] + u[i,:], u[i,:] >= 0]
        for t in range(timesteps):
            if aware: constr += [u[i, t] <= (u_max[i])*(t>=arrival_time[i])+1e-5, u[i, t] <= (u_max[i])*(t<dept_time[i])+1e-5]
            else: constr += [u[i, t] <= (u_max)*(t>=arrival_time[i])+1e-5, u[i, t] <= (u_max)*(t<dept_time[i])+1e-5]

    for t in range(timesteps):
        constr += [cp.sum(u[0:num_of_vehicles,t]) <= power_capacity]
        if log: obj += cp.sum(cp.log(u[:,t]+1e-5))
        else:obj += cp.sum(u[:,t])

    obj -= cp.norm(x[:, -1] - x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve(verbose=False)

    x_mpc = x.value
    u_mpc = u.value
    x_mpc_proj = np.zeros_like(x_mpc)
    x_mpc_proj[:, 0] = x_mpc[:, 0]
    u_mpc_proj = np.zeros_like(u_mpc)

    # validate feasibility
    for i in range(num_of_vehicles):
        for t in range(timesteps):
            u_hat = max_power - decay_rate*x_mpc_proj[i, t]
            if(u_mpc[i, t] >= u_hat + 1e-5):
                u_mpc_proj[i, t] = u_hat
                #print(f'at time{t}, {i} vehicle is violating')
            else:
                u_mpc_proj[i, t] = u_mpc[i, t]
            x_mpc_proj[i, t+1] = x_mpc_proj[i, t] + u_mpc_proj[i, t]
    return x_mpc_proj, u_mpc_proj


#Online algorithm, do not know SOC
def MPC_Solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, dept_time, power_capacity, plot_fig):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(num_of_vehicles, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value=initial_states
    max_sum_u.value = power_capacity
    u_max.value=max_power*np.ones((num_of_vehicles, ))

    obj = 0
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for t in range(timesteps): # need -5??
        constr += [x[:,t+1] == x[:,t] + u[:,t],
                   u[:,t] <= u_max,
                   u[:,t] >= 0,
                   cp.sum(u[:,t]) <= max_sum_u,
                   u[:,t] <= (t*np.ones_like(dept_time)<dept_time)*100.0+0.000001]
        obj += cp.sum(cp.log(u[:,t]))
    #constr+=[u[5,9]<=0.1]
    obj -= cp.norm(x[:, -1]-x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()
    #print(x.value[:,-1])
    #print(u.value)

    if plot_fig==True:
        plt.plot(x.value[10])
        plt.plot(u.value[10])
        print(f'arrival time:{arrival_time[10]}')
        plt.show()
    #print("DOne")


    return x.value, u.value


#Consider SOC level for the maximum charging power
def MPC_BIG_Solver(num_of_vehicles, timesteps, initial_states,
                   max_power, terminal_states, arrival_time, dept_time,
                   power_capacity, decay_rate, plot_fig):

    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(timesteps, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value = initial_states
    max_sum_u.value = power_capacity
    u_max.value = max_power * np.ones((timesteps, ))

    obj = 0
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for i in range(num_of_vehicles):
        constr +=[x[i,1:] == x[i,0:timesteps] + u[i,:],u[i,:] >= 0]
        #for t in range(timesteps):
        #    constr += [u[i, t] <= max_power*(t>=arrival_time[i]), u[i, t] <= max_power*(t<dept_time[i])]

    constr += [u <= (max_power - decay_rate * x[:, :-1])]

    for t in range (timesteps):
        constr += [cp.sum(u[:,t]) <= max_sum_u, 
                   u[:,t] <= (t*np.ones_like(dept_time) < dept_time)*100.0+1e-5,
                   u[:,t] <= (t*np.ones_like(arrival_time) > arrival_time)*100.0+1e-5,      
        ]
        obj += cp.sum(cp.log(u[:,t]))

    obj -= cp.norm(x[:, -1] - x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()

    if plot_fig==True:
        plt.plot(x.value[10])
        plt.plot(u.value[10])
        print(f'arrival time:{arrival_time[10]}')
        plt.show()

    return x.value, u.value



#Consider SOC level for the maximum charging power
#Haven't finished
def MPC_SOC_Solver(num_of_vehicles, timesteps, initial_states, max_power, terminal_states, dept_time, power_capacity):
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    max_sum_u = cp.Parameter(name='max_sum_u')
    u_max = cp.Parameter(num_of_vehicles, name='u_max')
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value=initial_states
    max_sum_u.value = power_capacity
    u_max.value=max_power*np.ones((num_of_vehicles, ))

    obj = 0
    print((0 * np.ones_like(dept_time) < dept_time)*10000)
    constr = [x[:,0] == x0, x[:,-1] <= x_terminal]

    for t in range(timesteps):
        constr += [x[:,t+1] == x[:,t] + u[:,t],
                   u[:,t] <= u_max,
                   u[:,t]>=0,
                   cp.sum(u[:,t]) <= max_sum_u,
                   u[:,t] <= ((t*np.ones_like(dept_time) < dept_time)*100.0+0.000001)
                   ]
        obj += cp.sum(cp.log(u[:,t]))
        # A bad grammar error here
        u_max.value=max_power*np.ones((num_of_vehicles, ))#-x[:,t]*0.5
        print("T", t)
        print(u_max.value)
    #constr+=[u[5,9]<=0.1]
    constr += [u<=x[:,:-1]]
    obj -= cp.norm(x[:, -1]-x_terminal, 1)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve()
    #print(x.value[:,-1])
    #print(u.value)


    #plt.plot(x.value[0])
    #plt.show()

    return x.value, u.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay", type=float, default=0.06, help="the decay rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed to generate the data")
    parser.add_argument("--soc", type=float, default=0, help="if not equal to zero, fix SoC to that value")
    parser.add_argument("--solver", type=str, default="standardaware", help="whether it is SoC aware")
    parser.add_argument("--power_capacity", type=int, default=4, help="fix SoC")
    parser.add_argument("--arrival_flag",type=str, default="poisson")
    
    opts = parser.parse_args()
    print(f'solver={opts.solver}')
    #One day has 24*12=288 slots
    seed = opts.seed
    np.random.seed(seed)

    #n = 10000
    num_steps=96
    decay_rate=opts.decay
    max_power=0.6 #denote \overline{u}_i(t)
    total_vehicles=50
    battery_capacity=8.0
    power_capacity=opts.power_capacity #charging station's capacity
    soc = opts.soc
    arrival_flag = opts.arrival_flag
    print(f'decay={decay_rate},total_vehicles={total_vehicles},power_capacity={power_capacity},soc={soc},arrival={arrival_flag}')

    dataset = generate_dataset_test(num_step=num_steps, total_vehicles=total_vehicles, battery_capacity=battery_capacity, seed=seed, soc=soc,arrival_flag=arrival_flag)
    arrival_time = dataset['arrival_time']
    depart_time = dataset['depart_time']
    initial_state = dataset['initial_state']
    required_energy = dataset['required_energy']
    final_energy = dataset['final_energy']

    start = time.time()
    if opts.solver == "aware":
        x,u = aware_MPC(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power,terminal_states=final_energy, dept_time=depart_time, power_capacity=power_capacity, arrival_time=arrival_time, decay_rate=decay_rate)
    elif opts.solver == 'standard':
        x,u = standard_MPC(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power,terminal_states=final_energy, dept_time=depart_time, power_capacity=power_capacity, arrival_time=arrival_time, decay_rate=decay_rate, aware=False,log=False)
    elif opts.solver == 'standardaware':
        x,u = standard_MPC(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power,terminal_states=final_energy, dept_time=depart_time, power_capacity=power_capacity, arrival_time=arrival_time, decay_rate=decay_rate, aware=True,log=False)
    end = time.time()
    
    print(f'takes {end-start} s.')
    print(np.sum(u))
    np.savez(f'result/MPC/{decay_rate}_soc{soc}_solver{opts.solver}_power{opts.power_capacity}_arrival{arrival_flag}_log.npz',x=x,u=u,initial=initial_state, require=required_energy, final_plan=final_energy)

