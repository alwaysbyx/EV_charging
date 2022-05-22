from mimetypes import init
from tkinter.font import families
from xmlrpc.client import Boolean
from matplotlib.style import available
import numpy as np
import csv
import cvxpy as cp
import matplotlib.pyplot as plt
from utils import generate_dataset, generate_dataset_test
import argparse
import time
from tqdm import tqdm

def MPC_Solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, arrival_time, dept_time, power_capacity, log=True, aware=False):
    
    # Requested end SoC for all vehicles
    x_terminal=cp.Parameter(num_of_vehicles, name='x_terminal')
    # Initial SoC for all vehicles
    x0 = cp.Parameter(num_of_vehicles, name='x0')
    # Peak charging power for the infrastructure
    max_sum_u = cp.Parameter(name='max_sum_u')
    # Maximum charging power for each vehicle at each time step
    if aware: 
        u_max = cp.Parameter(num_of_vehicles, name='u_max')
        initial_u_max = np.zeros((num_of_vehicles,))
        for i in range(num_of_vehicles):
            initial_u_max[i] = max_power - decay_rate*initial_states[i]
        u_max.value = initial_u_max
    else:
        u_max = cp.Parameter(name='u_max')
        u_max.value = max_power
    # SoC at each time step for each vehicle
    x = cp.Variable((num_of_vehicles, timesteps+1), name='x')
    # charging power at each time step for each vehicle
    u = cp.Variable((num_of_vehicles, timesteps), name='u')

    x_terminal.value=terminal_states
    x0.value = initial_states
    max_sum_u.value = power_capacity

    obj = 0
    constr = [x[:,0] == x0, 
              x[:,-1] <= x_terminal]

    for i in range(num_of_vehicles):
        constr += [x[i,1:] == x[i,0:timesteps] + u[i,:],
            u[i,:] >= 0,
        ]
        for t in range(timesteps):
            if aware:
                constr += [u[i, t] <= (u_max[i])*(t>=arrival_time[i]),
                    u[i, t] <= (u_max[i])*(t<dept_time[i]),
                    ]
            else:
                constr += [u[i, t] <= (u_max)*(t>=arrival_time[i]),
                    u[i, t] <= (u_max)*(t<dept_time[i]),
                    ]
    for t in range (timesteps):
        constr += [
                  cp.sum(u[0:num_of_vehicles,t]) <= power_capacity,
        ]
        if log: obj += cp.sum(cp.log(u[:,t] + 1e-5))
        else: obj += cp.sum(u[:,t])

    obj -= cp.norm(x[:, -1] - x_terminal, 2)
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve(verbose=False)

    return x.value, u.value


def online_solver(num_of_vehicles, timesteps, initial_states, max_power,
               terminal_states, arrival_time, dept_time, power_capacity, decay_rate, log=True, aware=False):
    initial_state_MPC = np.copy(initial_states)
    terminal_state_MPC = np.copy(terminal_states)
    u_mpc_online = np.zeros((num_of_vehicles, timesteps), dtype=float)
    x_mpc_online = np.zeros((num_of_vehicles, timesteps+1), dtype=float)
    x_mpc_online[:, :] = initial_state_MPC.reshape((-1, 1))
    print(np.sum(terminal_state_MPC-initial_state_MPC))
    for t in tqdm(range(int(arrival_time[0]), timesteps-5)):
        #Change this for variable case
        power_budget = power_capacity 

        vehicle_ending_index = (arrival_time <= t).sum()
        available_vehicle_idx = np.array([i for i in range(vehicle_ending_index) if dept_time[i] > t and x_mpc_online[i,t] <= terminal_states[i] - 1e-3])
        step_num_of_vehicles = len(available_vehicle_idx)
        #print(available_vehicle_idx, step_num_of_vehicles)
        if len(available_vehicle_idx) == 0:
            continue
        #print("current number of arrived cars", vehicle_ending_index, "current number of charging cars", step_num_of_vehicles)
        
        ## Obtain initial energy level of all cars
        step_initial_SOC = np.copy(x_mpc_online[available_vehicle_idx, t])
        ## Obtain required energy level of all cars
        step_terminal_SOC = np.copy(terminal_state_MPC[available_vehicle_idx])
        
        ## Obtain arrival departure time for all current cars
        arrival_schedule=np.copy(arrival_time[available_vehicle_idx])
        depart_schedule=np.copy(dept_time[available_vehicle_idx])
        # print('initial soc', step_initial_SOC)
        # print('terminal soc', step_terminal_SOC)
        # print('arrival_schedule', arrival_schedule)
        # print('depart_schedule', depart_schedule)

        x_index, u_index = MPC_Solver(num_of_vehicles = step_num_of_vehicles, timesteps = timesteps-t, 
               initial_states = step_initial_SOC, terminal_states = step_terminal_SOC, 
               max_power = max_power, arrival_time = arrival_schedule-t, dept_time = depart_schedule-t, 
               power_capacity = power_budget, log=log, aware=aware)
        

        # compute the maximum charging rate (soc curve)
        u_hat = max_power - decay_rate*x_mpc_online[available_vehicle_idx, t]
        u_hat2 = terminal_state_MPC[available_vehicle_idx] - x_mpc_online[available_vehicle_idx,t]

        u_mpc_online[available_vehicle_idx, t] = np.minimum(np.minimum(u_index[:, 0], u_hat), u_hat2)
        
        x_mpc_online[available_vehicle_idx, t+1] = x_mpc_online[available_vehicle_idx, t] + u_mpc_online[available_vehicle_idx, t]
    return x_mpc_online, u_mpc_online


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
    if opts.solver == 'standardaware':
        x,u = online_solver(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power,terminal_states=final_energy, dept_time=depart_time, power_capacity=power_capacity, arrival_time=arrival_time, decay_rate=decay_rate, aware=True, log=False)
    elif opts.solver == 'standard':
        x,u = online_solver(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power,terminal_states=final_energy, dept_time=depart_time, power_capacity=power_capacity, arrival_time=arrival_time, decay_rate=decay_rate, aware=False, log=False)
    end = time.time()
    
    print(f'takes {end-start} s to solve, delievery: {np.sum(u)}')
    np.savez(f'result/online_MPC/{decay_rate}_soc{soc}_solver{opts.solver}_power{opts.power_capacity}_arrival{arrival_flag}_nolog.npz',x=x,u=u,initial=initial_state, require=required_energy, final_plan=final_energy)


