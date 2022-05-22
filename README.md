# EV_charging

## Fixed Charging rate allocation
See utils.py ```generate_dataset_test``` for more details.
If soc=n, then the initial of SoC among vehicles are Uniform(50-5*n, 50+5*n)(percents).
```python 
# proportional to initial SoC
python aware_naive.py --power_capacity 4 --solver aware --arrival_flag fix1 --soc 0 
# allocate at the maximum charging rate
python aware_naive.py --power_capacity 4 --solver no --arrival_flag fix1 --soc 0  
```

## MPC for charging rate

```python 
# fixed upper bound u_i that is proportional to initial SoC
python offline_solver.py --solver standardaware --arrival_flag fix1
# fixed upper bound u_i
python offline_solver.py --solver standard --arrival_flag fix1
# fixed upper bound u_i that is u_i(x_i(t))
python offline_solver.py --solver aware --arrival_flag fix1 
```

Check ```result_analysis.ipynb``` to quick generate results.
