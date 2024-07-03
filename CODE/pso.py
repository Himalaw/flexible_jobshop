# coding=utf-8
import random
from math import sqrt
from copy import deepcopy
import numpy as np

from Operateur import *

def read_table2D_doc(filePath):
    with open(filePath, "r") as file:
        lines = file.read().split("\n")
    table2D = [line.split() for line in lines if line.strip() != ""]
    return table2D

def process_data(table2D):
    nb_jobs = int(table2D[0][0])
    nb_machines = int(table2D[0][1])
    avg_ops_per_machine = float(table2D[0][2])

    ops_per_job = [int(table2D[i + 1][0]) for i in range(nb_jobs)]
    job_machine_times = []

    for job_idx in range(nb_jobs):
        job_data = table2D[job_idx + 1]
        job_ops = []
        idx = 1
        for op_idx in range(ops_per_job[job_idx]):
            machine_count = int(job_data[idx])
            idx += 1
            task = [
                [job_data[idx + 2 * m], job_data[idx + 2 * m + 1]]
                for m in range(machine_count)
            ]
            idx += 2 * machine_count
            job_ops.append(task)
        job_machine_times.append(job_ops)
    
    return table2D, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job, job_machine_times

def available_speeds():
    return np.array([1.0, 1.5, 2.0])

def generate_random_schedule(nb_jobs, job_machine_times, ops_per_job):
    speeds = available_speeds()
    total_ops = sum(ops_per_job)
    schedule = np.zeros((total_ops, 4))
    current_ops = np.zeros(nb_jobs, int)
    final_ops = np.array(ops_per_job)
    h = 0

    while (current_ops != final_ops).any():
        available_jobs = [j for j in range(nb_jobs) if current_ops[j] < final_ops[j]]
        job = random.choice(available_jobs)
        machines = job_machine_times[job][current_ops[job]]
        if isinstance(machines[0], list):
            machine = random.choice([m[0] for m in machines])
        else:
            machine = machines[0]
        speed = random.choice(speeds)
        schedule[h] = [job + 1, current_ops[job] + 1, machine, speed]
        current_ops[job] += 1
        h += 1
    
    return schedule

def schedule_operations(schedule, nb_jobs, nb_machines, job_machine_times, ops_per_job):
    total_ops = sum(ops_per_job)
    results = np.zeros((total_ops, 7))
    job_times = np.zeros(nb_jobs)
    machine_times = np.zeros(nb_machines)
    j = 0

    while j < total_ops:
        job = int(schedule[j][0])
        op = int(schedule[j][1])
        mach = int(schedule[j][2])
        speed = float(schedule[j][3])
        machines = job_machine_times[job - 1][op - 1]
        if isinstance(machines[0], str):
            proc_time = float(machines[1])
        else:
            proc_time = float(next(m[1] for m in machines if int(m[0]) == mach))
        proc_time /= speed
        start_time = max(job_times[job - 1], machine_times[mach - 1])
        end_time = start_time + proc_time
        job_times[job - 1] = end_time
        machine_times[mach - 1] = end_time

        results[j] = [job, op, mach, speed, proc_time, start_time, end_time]
        j += 1
    
    return results

def save_results_by_machine(results, nb_machines):
    results_by_machine = [[] for _ in range(nb_machines)]
    for result in results:
        machine_idx = int(result[2]) - 1
        results_by_machine[machine_idx].append(result)
    return results_by_machine

def save_results_by_job(results, nb_jobs):
    results_by_job = [[] for _ in range(nb_jobs)]
    for result in results:
        job_idx = int(result[0]) - 1
        results_by_job[job_idx].append(result)
    return results_by_job

def calculate_cmax(results):
    return max(result[6] for result in results)

def calculate_standby_time(results_by_machine, cmax):
    standby_times = np.zeros(len(results_by_machine))
    for i, machine_results in enumerate(results_by_machine):
        if not machine_results:
            standby_times[i] = cmax
        else:
            standby_times[i] = sum(
                machine_results[j + 1][5] - machine_results[j][6]
                for j in range(len(machine_results) - 1)
            )
            standby_times[i] += machine_results[0][5]
            standby_times[i] += cmax - machine_results[-1][6]
    return standby_times

def calculate_energy(results_by_machine, cmax):
    standby_times = calculate_standby_time(results_by_machine, cmax)
    tec_standby = standby_times
    tec_on = [
        sum(4 * result[4] * result[3] ** 2 for result in machine_results)
        for machine_results in results_by_machine
    ]
    total_tec = [standby + on for standby, on in zip(tec_standby, tec_on)]
    return total_tec, sum(total_tec)

def calculate_workload_balance(results_by_machine):
    workloads = [sum(result[4] for result in machine_results) for machine_results in results_by_machine]
    average_workload = sum(workloads) / len(results_by_machine)
    return sqrt(sum((workload - average_workload) ** 2 for workload in workloads))

def population_initialization(N, job_machine_times, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job):
    total_population = []
    all_schedules = []
    for _ in range(N):
        random_schedule = generate_random_schedule(nb_jobs, deepcopy(job_machine_times), ops_per_job)
        all_schedules.append(random_schedule)
        results = schedule_operations(random_schedule, nb_jobs, nb_machines, job_machine_times, ops_per_job)
        results_by_machine = save_results_by_machine(results, nb_machines)
        cmax = calculate_cmax(results)
        tec_by_machine, total_tec = calculate_energy(results_by_machine, cmax)
        wb = calculate_workload_balance(results_by_machine)
        total_population.append([results, total_tec, wb])
    return total_population, all_schedules

def update_crowding_distance(omega, new_solution):
    omega.append(new_solution)
    omega = sorted(omega, key=lambda x: x[1][1])
    distances = [100000] + [
        (abs(omega[i + 1][1][1] - omega[i - 1][1][1]) / (abs(omega[0][1][1] - omega[-1][1][1]) + 0.00001)) +
        (abs(omega[i + 1][1][2] - omega[i - 1][1][2]) / (abs(omega[0][1][2] - omega[-1][1][2]) + 0.00001))
        for i in range(1, len(omega) - 1)
    ] + [100000]
    min_distance = min(distances)
    omega.pop(distances.index(min_distance))
    return omega

def is_non_dominated(solution, omega):
    return all(solution[1][1] < other[1][1] or solution[1][2] < other[1][2] for other in omega)

def pso_algorithm():
    N = 100
    max_non_dominated = 25
    max_iterations = 1000
    alpha = 0.90
    beta = 0.5
    gamma = 0.000001
    delta = 0.000001

    file_path = "Dataprojet/Mk01.fjs"
    table2D = read_table2D_doc(file_path)
    table2D, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job, job_machine_times = process_data(table2D)

    primary_swarm, all_schedules = population_initialization(N, job_machine_times, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job)
    personal_best = [[sched, sol] for sched, sol in zip(all_schedules, primary_swarm)]
    global_best = []

    for solution in primary_swarm:
        if is_non_dominated([[], solution], global_best):
            global_best = [sol for sol in global_best if not (solution[1] < sol[1][1] and solution[2] < sol[1][2])]
            global_best.append([[], solution])
    
    for t in range(max_iterations):
        for i in range(N):
            rand_val = random.random()
            if rand_val < alpha:
                continue
            elif rand_val < alpha + beta:
                personal_best[i][0] = generate_random_schedule(nb_jobs, deepcopy(job_machine_times), ops_per_job)
                personal_best[i][1] = schedule_operations(personal_best[i][0], nb_jobs, nb_machines, job_machine_times, ops_per_job)
            else:
                global_best_solution = random.choice(global_best)
                personal_best[i][0] = generate_random_schedule(nb_jobs, deepcopy(job_machine_times), ops_per_job)
                personal_best[i][1] = schedule_operations(personal_best[i][0], nb_jobs, nb_machines, job_machine_times, ops_per_job)
            
            if is_non_dominated(personal_best[i], global_best):
                if len(global_best) > max_non_dominated:
                    global_best = update_crowding_distance(global_best, personal_best[i])
                else:
                    global_best.append(personal_best[i])
            
            if is_non_dominated(personal_best[i], personal_best):
                if len(personal_best[i]) > max_non_dominated:
                    personal_best[i] = update_crowding_distance(personal_best[i], personal_best[i])
                else:
                    personal_best.append(personal_best[i])
            
            if random.random() < delta:
                for u in range(len(personal_best[i][0])):
                    mutation_prob = random.random()
                    if mutation_prob < 0.1:
                        personal_best[i][0][u][3] = random.choice(available_speeds())
                    elif mutation_prob < 0.5:
                        personal_best[i][0][u][2] = random.choice([int(m[0]) for m in job_machine_times[int(personal_best[i][0][u][0]) - 1][int(personal_best[i][0][u][1]) - 1]])
                    else:
                        personal_best[i][0] = np.roll(personal_best[i][0], shift=random.randint(1, len(personal_best[i][0]) - 1), axis=0)
                
                personal_best[i][1] = schedule_operations(personal_best[i][0], nb_jobs, nb_machines, job_machine_times, ops_per_job)
                
                if is_non_dominated(personal_best[i], global_best):
                    if len(global_best) > max_non_dominated:
                        global_best = update_crowding_distance(global_best, personal_best[i])
                    else:
                        global_best.append(personal_best[i])
                
                if is_non_dominated(personal_best[i], personal_best):
                    if len(personal_best[i]) > max_non_dominated:
                        personal_best[i] = update_crowding_distance(personal_best[i], personal_best[i])
                    else:
                        personal_best.append(personal_best[i])

    for i, solution in enumerate(global_best):
        print(f"\nGlobal Best {i + 1}: TEC = {solution[1][1]}, WB = {solution[1][2]}")

pso_algorithm()