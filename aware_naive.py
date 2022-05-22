import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
import csv
import pandas as pd
from utils import generate_dataset, generate_dataset_test
import argparse

def naive(num_of_vehicles, timesteps, initial_states, max_power, terminal_states, arrival_time, dept_time, power_capacity, decay_rate,aware=False):
    
    initial_states = np.copy(initial_states)
    u_mpc_online = np.zeros((num_of_vehicles, timesteps), dtype=float)
    x_mpc_online = np.zeros((num_of_vehicles, timesteps+1), dtype=float)
    x_mpc_online[:, :] = initial_states.reshape((-1, 1))
    if aware: max_u = max_power * np.ones((num_of_vehicles,)) - decay_rate * initial_states
    else: max_u = max_power

    for t in range(int(arrival_time[0]), timesteps-5):
        power_budget = power_capacity
        vehicle_ending_index = (arrival_time <= t).sum()
        available_vehicle_idx = np.array([i for i in range(vehicle_ending_index) if dept_time[i] > t and x_mpc_online[i,t] < terminal_states[i]])
        if len(available_vehicle_idx) == 0:
            continue
        u_val = np.zeros((len(available_vehicle_idx),))
        charging_sesson = 0
        while power_budget >= 0:
            vehicle_idx = available_vehicle_idx[charging_sesson]
            if aware: available_charging = np.minimum(max_u[vehicle_idx], power_budget)
            else: available_charging = np.minimum(max_u, power_budget)
            #print(f'at time {t}, vehicle{vehicle_idx} can charge {available_charging}')
            available_charging = np.minimum(terminal_states[vehicle_idx]-x_mpc_online[vehicle_idx,t], available_charging)
            #print(f'at time {t}, vehicle{vehicle_idx} can charge {available_charging}')
            u_val[charging_sesson] = np.maximum(available_charging, 0)
            power_budget -= u_val[charging_sesson]
            # check availability
            u_hat = max_power - decay_rate*x_mpc_online[vehicle_idx, t]
            u_mpc_online[vehicle_idx,t] = np.minimum(u_hat, u_val[charging_sesson])
            x_mpc_online[vehicle_idx,t+1:] = x_mpc_online[vehicle_idx,t] + u_mpc_online[vehicle_idx,t]
            charging_sesson += 1
            if charging_sesson >= len(available_vehicle_idx):
                break
        #u_hat = max_power - decay_rate*x_mpc_online[available_vehicle_idx, t]
        #print(f'at time {t}, vehicle{vehicle_idx} can charge {u_hat}, the previous SoC is {x_mpc_online[available_vehicle_idx, t]}')
        #u_mpc_online[available_vehicle_idx,t] = np.minimum(u_hat[:], u_val[:])
        #print(u_val[:], u_hat, u_mpc_online[available_vehicle_idx,t])
        #x_mpc_online[available_vehicle_idx, t+1] = x_mpc_online[available_vehicle_idx, t] + u_mpc_online[available_vehicle_idx, t]


    return x_mpc_online, u_mpc_online

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay", type=float, default=0.06, help="the decay rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed to generate the data")
    parser.add_argument("--soc", type=float, default=0, help="fix SoC")
    parser.add_argument("--power_capacity", type=int, default=4, help="fix SoC")
    parser.add_argument("--arrival_flag",type=str, default="poisson")
    parser.add_argument("--solver",type=str, default="aware")
    
    opts = parser.parse_args()
    #One day has 24*12=288 slots
    seed = opts.seed
    np.random.seed(seed)

    #n = 10000
    num_steps=96
    decay_rate=opts.decay
    max_power=0.6 #denote \overline{u}_i(t)
    total_vehicles=50
    battery_capacity = 8.0
    power_capacity = opts.power_capacity #charging station's capacity
    soc = opts.soc
    arrival_flag = opts.arrival_flag

    dataset = generate_dataset_test(num_step=num_steps, total_vehicles=total_vehicles, battery_capacity=battery_capacity, seed=seed, soc=soc,arrival_flag=arrival_flag)
    arrival_time = dataset['arrival_time']
    depart_time = dataset['depart_time']
    initial_state = dataset['initial_state']
    required_energy = dataset['required_energy']
    final_energy = dataset['final_energy']

    if opts.solver == 'aware':
        x,u = naive(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power, terminal_states=final_energy, arrival_time=arrival_time, dept_time=depart_time, power_capacity=power_capacity, decay_rate=decay_rate,aware=True)
    elif opts.solver == 'no':
        x,u = naive(num_of_vehicles=total_vehicles, timesteps=num_steps, initial_states=initial_state, max_power=max_power, terminal_states=final_energy, arrival_time=arrival_time, dept_time=depart_time, power_capacity=power_capacity, decay_rate=decay_rate,aware=False)
    
    print(f'required: {np.sum(required_energy)}, delivery: {np.sum(u)}.')

    file_name = f'result/naive/{decay_rate}_solver{opts.solver}_soc{soc}_power{opts.power_capacity}_arrival{arrival_flag}.npz'
    print(file_name)
    np.savez(file_name, u=u, x=x, initial=initial_state, require=required_energy, final_plan=final_energy)