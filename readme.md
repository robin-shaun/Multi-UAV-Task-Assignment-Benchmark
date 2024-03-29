# Multi-UAV Task Assignment Benchmark
## 多无人机任务分配算法测试基准

## Introduction
A benchmark for multi-UAV task assignment is presented in order to evaluate different algorithms. An extended Team Orienteering Problem is modeled for a kind of multi-UAV task assignment problem. Three intelligent algorithms, i.e., Genetic Algorithm, Ant Colony Optimization and Particle Swarm Optimization are implemented to solve the problem. A series of experiments with different settings are conducted to evaluate three algorithms. The modeled problem and the evaluation results constitute a benchmark, which can be used to evaluate other algorithms used for multi-UAV task assignment problems.

Notice that three algorithms run at three CPU cores respectively, which means that there is no parallel optimization in this benchmark.

<img src="./task_pic/large/ACO-1-1.png" width="640" height="368" />  

<img src="./mean_reward_large.png" width="640" height="368" />  

<img src="./mean_time_large.png" width="640" height="368" />  

Please refer to the paper to see more detail.

K. Xiao, J. Lu, Y. Nie, L. Ma, X. Wang and G. Wang, "A Benchmark for Multi-UAV Task Assignment of an Extended Team Orienteering Problem," 2022 China Automation Congress (CAC), Xiamen, China, 2022, pp. 6966-6970, doi: 10.1109/CAC57257.2022.10054991.

ArXiv preprint **[ arXiv:2003.09700](https://arxiv.org/abs/2009.00363)** 


## Usage

### 1. Algorithm input and output

Algorithm input includes vehicle number (scalar),  speeds of vehicles ($n\times1$ array), target  number (scalar $n$),  targets ($(n+1)\times4$ array, the first line is depot, the first column is x position, the second column is y position, the third column is reward and the forth column is time consumption to finish the mission), time limit (scalar).  The code below is the initialization of the class GA in `ga.py`.

```python
def __init__(self, vehicle_num, vehicles_speed, target_num, targets, time_lim)
```

There should be a function called `run()` in the algorithm class, and the function should return task assignment plan(array, e.g. [[28, 19, 11], [25, 22, 7, 16, 17, 23], [21, 26, 12, 9, 6, 3], [5, 15, 1], [18, 20, 29]], each subset is a vehicle path) and computational time usage (scalar). 

### 2. Evaluate

You can replace one algorithm  below with another algorithm in `evaluate.py`, and then `python evaluate.py`. If you don't want to evaluate three algorithm together, you should modify the code properly( this is easy).    

```python
ga = GA(vehicle_num,env.vehicles_speed,target_num,env.targets,env.time_lim)
aco = ACO(vehicle_num,target_num,env.vehicles_speed,env.targets,env.time_lim)
pso = PSO(vehicle_num,target_num ,env.targets,env.vehicles_speed,env.time_lim)
ga_result=p.apply_async(ga.run)
aco_result=p.apply_async(aco.run)
pso_result=p.apply_async(pso.run)
p.close()
p.join()
ga_task_assignmet = ga_result.get()[0]
env.run(ga_task_assignmet,'GA',i+1,j+1)
re_ga[i].append((env.total_reward,ga_result.get()[1]))
env.reset()
aco_task_assignmet = aco_result.get()[0]
env.run(aco_task_assignmet,'ACO',i+1,j+1)
re_aco[i].append((env.total_reward,aco_result.get()[1]))
env.reset()
pso_task_assignmet = pso_result.get()[0]
env.run(pso_task_assignmet,'PSO',i+1,j+1)
re_pso[i].append((env.total_reward,pso_result.get()[1]))
```

### 3. About reinforcement learning

In `Env()` in `evaluate.py`, function `step` is used for reinforcement learning. Because this is still being developed, we cannot supply a demo. If your algorithm is reinforcement learning, you can try to train it with `Env()`. Your pull request and issue are welcome.

## Enhancement

This [repository](https://github.com/dietmarwo/Multi-UAV-Task-Assignment-Benchmark) does great enhancement and you can use it for high performance. Thanks to [dietmarwo](https://github.com/dietmarwo) for the nice work.

1) GA uses [numba](https://numba.pydata.org/) for a dramatic speedup. Parameters are adapted so that the
    execution time remains the same: popsize 50 -> 300, iterations 500 -> 6000
    For this reason GA now performs much better compared to the original version.

2) Experiments are configured so that wall time for small size is balanced. This means:
    increased effort for GA, decreased effort for ACO. For medium / large 
    problem size you see which algorithms scale badly: Increase execution time superlinear
    in relation to the problem size. Avoid these for large problems. 

3) Adds a standard continuous optimization algorithm: [BiteOpt](https://github.com/avaneev/biteopt) 
    from Aleksey Vaneev - using the same fitness function as GA.py. 
    BiteOpt is the only algorithm included which works well with a large problem size. 
    It is by far the simplest implementation, only the fitness function needs
    to be coded, since we can apply a continuous optimization library 
    [fcmaes](https://github.com/dietmarwo/fast-cma-es). Execute "pip install fcmaes" to use it. 

4) Uses NestablePool to enable BiteOpt multiprocessing: Many BiteOpt optimization runs
   are performed in parallel and the best result is returned. Set workers=1 
   if you want to test BiteOpt single threaded. 
   
5) All results are created using an AMD 5950x 16 core processor
    utilizing all cores: 29 parallel BiteOpt threads, the other 3 algorithms remain single threaded. 

6) Added test_bite.py where you can monitor the progress of BiteOpt applied to the problem.

7) Added test_mode.py where you can monitor the progress of fcmaes-MODE applied to the problem and compare it
   to BiteOpt for the same instance. fcmaes-MODE is a multi-objective optimizer applied to a 
   multi-objective variant of the problem.
   Objectives are: reward (to be maximized), maximal time (to be minimized), energy (to be minimized).
   The maximal time constraint from the single objective case is still valid.
   Energy consumption is approximated by `sum(dt*v*v)`


 

