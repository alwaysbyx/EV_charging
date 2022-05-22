import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt
from offline_solver import MPC_BIG_Solver, MPC_Solver
import csv
import pandas as pd
from utils import generate_dataset, generate_dataset_test
import argparse



def check_SOC_update(initial_states, decay_rate, charging_rate, max_power):
    #print("Initial states", initial_states)
    updated_state=np.round(initial_states + np.minimum(charging_rate[:,0], max_power-decay_rate*initial_states), 2)
    #print(updated_state)
    return updated_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay", type=float, default=0.05, help="the decay rate")
    parser.add_argument("--seed", type=int, default=0, help="random seed to generate the data")
    parser.add_argument("--soc", type=float, default=0, help="fix SoC")
    parser.add_argument("--power_capacity", type=int, default=4, help="fix SoC")
    parser.add_argument("--arrival_flag",type=str, default="poisson")
    
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

    # print("Arrival time", arrival_time)
    # print("Depart time", depart_time)
    # print("initial State", initial_state)
    # print("Required energy", required_energy)
    # print("Final state", final_energy)

    initial_state_EDF=np.copy(initial_state)
    u_mat=np.zeros((total_vehicles, num_steps), dtype=float)
    sum_delivery = 0.

    #-5 to avoid computation infeasibility at this time
    for t in range(int(arrival_time[0])+1, num_steps-5):
        power_budget=power_capacity #Change this for variable case

        #print("Current time", t)

        #Firstly get the states
        #print("current number of arrived cars", (arrival_time < t).sum())
        vehicle_ending_index = (arrival_time < t).sum()
        step_initial_SOC = np.copy(initial_state_EDF[:vehicle_ending_index])
        depart_schedule=np.copy(depart_time[:vehicle_ending_index])
        u_val=np.zeros_like(step_initial_SOC) #record available cars charging rate
        index=np.argsort(depart_schedule) #sort the departure time
        charging_sessions=0

        while power_budget>=0:
            if depart_schedule[index[charging_sessions]] >= t: #not departed yet
                available_charging=np.minimum(max_power, power_budget)
                u_val[index[charging_sessions]] = np.maximum(np.minimum(available_charging, final_energy[index[charging_sessions]]-step_initial_SOC[index[charging_sessions]]),0)

            power_budget -= u_val[index[charging_sessions]]
            charging_sessions+=1

            if charging_sessions>=vehicle_ending_index:
                break

        #print("SUM EDF", np.sum(u_val))
        updated_val=np.minimum(u_val, np.ones_like(u_val)*max_power - decay_rate * step_initial_SOC) 
        #print("U after MPC cut", updated_val)
        #print("SUM EDF Cut", np.sum(updated_val))
        initial_state_EDF[:vehicle_ending_index] += updated_val
        #print("SOC_states", np.round(initial_state_SOC[:vehicle_ending_index],2))
        u_mat[:vehicle_ending_index, t]=updated_val

    file_name = f'result/EDF/{decay_rate}_soc{soc}_power{opts.power_capacity}_arrival{arrival_flag}.npz'
    print(file_name)
    np.savez(file_name, u=u_mat, final=initial_state_EDF, initial=initial_state, require=required_energy, final_plan=final_energy)
