# coding=utf-8
from math import sqrt, ceil
from random import randint, choice, random
from copy import deepcopy

# Read data from a file and return a 2D table
def read_2D_table(file_path):
    with open(file_path, "r") as file:
        lines = file.read().split("\n")
    return [line.split() for line in lines if line.strip()]

# Process data from the table and extract relevant information
def extract_data(table2D):
    num_jobs = int(table2D[0][0])
    num_machines = int(table2D[0][1])
    avg_ops_per_machine = float(table2D[0][2])
    
    ops_per_job = [int(table2D[line][0]) for line in range(1, num_jobs + 1)]
    job_machine_times = []

    for line in range(1, num_jobs + 1):
        machine_times = []
        position = 1
        for op in range(ops_per_job[line - 1]):
            num_options = int(table2D[line][position])
            position += 1
            machine_ops = [table2D[line][position + 2 * i:position + 2 * (i + 1)] for i in range(num_options)]
            position += 2 * num_options
            machine_times.append(machine_ops)
        job_machine_times.append(machine_times)

    return table2D, num_jobs, num_machines, avg_ops_per_machine, ops_per_job, job_machine_times

# Define possible speeds
def get_possible_speeds():
    return [1.0, 1.5, 2.0]

# Generate a random schedule
def generate_random_schedule(num_jobs, job_machine_times):
    speeds = get_possible_speeds()
    ops_per_job = [0] * num_jobs
    schedule = []

    while any(job_machine_times):
        job = choice([i for i, job in enumerate(job_machine_times) if job])
        ops_per_job[job] += 1
        machines = job_machine_times[job][0]
        machine = choice(machines)[0]
        speed = choice(speeds)
        schedule.append([job + 1, ops_per_job[job], machine, speed])
        del job_machine_times[job][0]

    return schedule

# Calculate processing times and create a schedule
def create_schedule(schedule, num_jobs, num_machines, job_machine_times):
    job_times = [0] * num_jobs
    machine_times = [0] * num_machines
    result = []

    while schedule:
        job, op, machine, speed = schedule.pop(0)
        machine = int(machine)
        machine_ops = job_machine_times[job - 1][op - 1]
        proc_time = float(next(time for m, time in machine_ops if int(m) == machine)) / speed
        start_time = max(job_times[job - 1], machine_times[machine - 1])
        end_time = start_time + proc_time

        job_times[job - 1] = end_time
        machine_times[machine - 1] = end_time

        result.append([job, op, machine, speed, proc_time, start_time, end_time])

    return result

# Save results by machine and job
def get_results_by_machine(result, num_machines):
    result_by_machine = [[] for _ in range(num_machines)]
    for entry in result:
        result_by_machine[entry[2] - 1].append(entry)
    return result_by_machine

def get_results_by_job(result, num_jobs):
    result_by_job = [[] for _ in range(num_jobs)]
    for entry in result:
        result_by_job[entry[0] - 1].append(entry)
    return result_by_job

# Calculate performance metrics
def calculate_Cmax(result):
    return max(entry[6] for entry in result)

def calculate_standby_times(result_by_machine, Cmax):
    standby_times = []
    for machine in result_by_machine:
        if not machine:
            standby_times.append(Cmax)
        else:
            standby = sum(machine[i + 1][5] - machine[i][6] for i in range(len(machine) - 1))
            standby += machine[0][5] + (Cmax - machine[-1][6])
            standby_times.append(standby)
    return standby_times

def calculate_energy(result_by_machine, Cmax):
    wait_times = calculate_standby_times(result_by_machine, Cmax)
    TEC = []
    standby_energy = 1
    for i, machine in enumerate(result_by_machine):
        standby_energy_total = wait_times[i] * standby_energy
        operational_energy_total = sum(4 * entry[4] * entry[3] ** 2 for entry in machine)
        TEC.append(standby_energy_total + operational_energy_total)
    return TEC, sum(TEC)

def calculate_workload_balance(result_by_machine):
    workload = [sum(entry[4] for entry in machine) for machine in result_by_machine]
    avg_workload = sum(workload) / len(result_by_machine)
    return sqrt(sum((wl - avg_workload) ** 2 for wl in workload))

# Operators
def balance_workload(result_by_machine, schedule, job_machine_times):
    workload = [sum(entry[4] for entry in machine) for machine in result_by_machine]
    max_workload_machine = workload.index(max(workload)) + 1

    candidate_ops = [entry for entry in schedule if int(entry[2]) == max_workload_machine]
    if not candidate_ops:
        print("\n" + "-" * 100 + "\nNo changes possible\n" + "-" * 100 + "\n")
        return schedule

    op_to_move = choice(candidate_ops)
    job, op, _, speed = op_to_move
    possible_machines = [int(machine[0]) for machine in job_machine_times[job - 1][op - 1]]

    min_workload_machine = min(possible_machines, key=lambda m: workload[m - 1])
    op_to_move[2] = str(min_workload_machine)

    print(f"\n{'-' * 100}\nJob: {job}, Operation: {op}, moved from machine {max_workload_machine} to {min_workload_machine}\n{'-' * 100}\n")

    return schedule

def swap_operations(schedule, num_jobs):
    mod_schedule = deepcopy(schedule)
    op1, op2 = randint(0, len(mod_schedule) - 1), randint(0, len(mod_schedule) - 1)
    mod_schedule[op1], mod_schedule[op2] = mod_schedule[op2], mod_schedule[op1]

    while not is_valid_schedule(mod_schedule, num_jobs):
        mod_schedule = deepcopy(schedule)
        op1, op2 = randint(0, len(mod_schedule) - 1), randint(0, len(mod_schedule) - 1)
        mod_schedule[op1], mod_schedule[op2] = mod_schedule[op2], mod_schedule[op1]

    print(f"\nOperation {schedule[op1][1]} of job {schedule[op1][0]} swapped with operation {schedule[op2][1]} of job {schedule[op2][0]}\n")
    return mod_schedule

def is_valid_schedule(schedule, num_jobs):
    expected_ops = [1] * num_jobs
    for job, op, *_ in schedule:
        if op != expected_ops[job - 1]:
            return False
        expected_ops[job - 1] += 1
    return True

def change_speed(schedule, job_num, op_num):
    mod_schedule = deepcopy(schedule)
    speeds = get_possible_speeds()

    for i, entry in enumerate(mod_schedule):
        if entry[0] == job_num and entry[1] == op_num:
            current_speed = entry[3]
            new_speed = choice([s for s in speeds if s != current_speed])
            entry[3] = new_speed
            print(f"Changed speed of job {job_num}, operation {op_num}, from {current_speed} to {new_speed}")
            break
    return mod_schedule

# Display results
def display_results_by_job(results_by_job, num_jobs):
    print("\nSchedule results by job [job, operation, machine, speed, processing time, start, end]:")
    for i in range(num_jobs):
        print(f"Job {i + 1}: {results_by_job[i]}")

def display_results_by_machine(results_by_machine, num_machines):
    print("\nSchedule results by machine [job, operation, machine, speed, processing time, start, end]:")
    for i in range(num_machines):
        print(f"Machine {i + 1}: {results_by_machine[i]}")

def display_TEC(TEC_by_machine, TEC_total):
    print(f"TEC by machine:\n{TEC_by_machine}\nTotal TEC: {TEC_total}")

# Initialize parameters
def initialize_parameters():
    population_size = 40
    num_memeplexes = 5
    num_iterations = 5
    memeplex_size = ceil(population_size / num_memeplexes)
    return population_size, num_memeplexes, num_iterations, memeplex_size

# Generate initial population
def generate_initial_population(population_size, job_machine_times):
    population = []
    for _ in range(population_size):
        schedule = generate_random_schedule(num_jobs, deepcopy(job_machine_times))
        result = create_schedule(schedule, num_jobs, num_machines, job_machine_times)
        result_by_machine = get_results_by_machine(result, num_machines)
        Cmax = calculate_Cmax(result)
        TEC, _ = calculate_energy(result_by_machine, Cmax)
        WB = calculate_workload_balance(result_by_machine)
        population.append([result, TEC, WB])
    return population

# Generate non-dominated set
def generate_omega(population):
    omega = []
    for i, sol_i in enumerate(population):
        TEC_i, WB_i = sol_i[1], sol_i[2]
        if all(TEC_i < sol_j[1] or WB_i < sol_j[2] for j, sol_j in enumerate(population) if i != j):
            omega.append(sol_i)
    return omega

# Merge population and omega
def merge_population_and_omega(population, omega):
    return population + omega

# Construct memeplexes
def construct_memeplexes(num_memeplexes, P_barre, memeplex_size):
    memeplexes = [[] for _ in range(num_memeplexes)]
    while P_barre:
        for u in range(num_memeplexes):
            if len(P_barre) > 1:
                x1, x2 = choice(P_barre), choice(P_barre)
                while x1 == x2:
                    x2 = choice(P_barre)
                selected = x1 if (x1[1] < x2[1] and x1[2] < x2[2]) else x2
                memeplexes[u].append(selected)
                P_barre.remove(selected)
            elif P_barre:
                memeplexes[u].append(P_barre.pop())
    return memeplexes

# Global search operators
def swap_operations_globally(x1, xbest):
    delta = 0.5
    new_list = []
    x, xb = deepcopy(x1), deepcopy(xbest)
    for i in range(len(x[0])):
        if random() < delta:
            new_list.append(xb[0].pop(0))
        else:
            new_list.append(x[0].pop(0))
    return new_list

def swap_machines_globally(x1, xbest):
    g1, g2 = randint(0, len(x1[0]) - 2), randint(g1 + 1, len(x1[0]) - 1)
    return [xb if g1 <= i <= g2 else x for i, (x, xb) in enumerate(zip(x1[0], xbest[0]))]

def swap_speeds_globally(x1, xbest):
    g1, g2 = randint(0, len(x1[0]) - 2), randint(g1 + 1, len(x1[0]) - 1)
    return [xb if g1 <= i <= g2 else x for i, (x, xb) in enumerate(zip(x1[0], xbest[0]))]

# Local search operators
def insert_operation(xbest):
    x = deepcopy(xbest)
    simple_x = [op[0] for op in x[0]]
    j, k = randint(0, len(simple_x) - 1), randint(0, len(simple_x) - 1)
    simple_x.insert(k, simple_x.pop(j))

    new_x = [[job, op] for job, op in enumerate(simple_x, 1)]
    for i in range(num_jobs + 1):
        count = 1
        for job in new_x:
            if job[0] == i:
                job[1] = count
                count += 1

    for job in new_x:
        for op in x[0]:
            if job[0] == op[0] and job[1] == op[1]:
                job.extend(op[2:])
    return new_x

def change_machine_locally(xbest):
    x = deepcopy(xbest)
    ops = [randint(0, len(x[0]) - 1) for _ in range(randint(1, len(x[0])))]
    new_x = [[job, op, (right_machine(job, op, machine, speed) if i in ops else machine), speed] for i, (job, op, machine, speed) in enumerate(x[0])]
    return new_x

def change_speed_locally(xbest):
    x = deepcopy(xbest)
    ops = [randint(0, len(x[0]) - 1) for _ in range(randint(1, len(x[0])))]
    new_x = [[job, op, machine, (right_speed(speed) if i in ops else speed)] for i, (job, op, machine, speed) in enumerate(x[0])]
    return new_x

# Randomly select x and xb
def select_randomly(memeplex, memeplex_num):
    copy_memeplex = deepcopy(memeplex[memeplex_num])
    while True:
        xb = choice(copy_memeplex)
        copy_memeplex.remove(xb)
        if all(xb[1] < x[1] or xb[2] < x[2] for x in memeplex[memeplex_num]):
            break
    x = choice([x for x in memeplex[memeplex_num] if x != xb])
    return x, xb

# Create final schedule
def create_final_schedule(schedule, num_jobs, num_machines, job_machine_times):
    result = create_schedule(schedule, num_jobs, num_machines, job_machine_times)
    result_by_machine = get_results_by_machine(result, num_machines)
    Cmax = calculate_Cmax(result)
    TEC, _ = calculate_energy(result_by_machine, Cmax)
    WB = calculate_workload_balance(result_by_machine)
    return [result, TEC, WB]

# Search process within memeplex
def search_within_memeplex(memeplex, num_iterations, memeplex_num, beta, nu, num_machines, num_jobs, omega, max_omega_size):
    for _ in range(num_iterations):
        x1, xbest = select_randomly(memeplex, memeplex_num)
        if random() < beta:
            new_schedule = swap_operations_globally(x1, xbest)
        elif random() < nu:
            new_schedule = swap_machines_globally(x1, xbest)
        else:
            new_schedule = swap_speeds_globally(x1, xbest)

        z = create_final_schedule(new_schedule, num_jobs, num_machines, job_machine_times)
        if z[1] < xbest[1] or z[2] < xbest[2]:
            if all(z[1] < o[1] and z[2] < o[2] for o in omega):
                omega.append(z)
            if len(omega) > max_omega_size:
                crowding_distance = lambda sol: sum(abs(sol[i][j] - sol[i - 1][j]) for i in range(1, len(sol)) for j in [1, 2])
                omega = sorted(omega, key=crowding_distance, reverse=True)[:max_omega_size]
            memeplex[memeplex_num][memeplex[memeplex_num].index(xbest)] = z
    return omega

# SFLA algorithm
def sfla():
    population_size, num_memeplexes, num_iterations, memeplex_size = initialize_parameters()
    population = generate_initial_population(population_size, job_machine_times)
    omega = generate_omega(population)
    P_barre = merge_population_and_omega(population, omega)

    beta, nu = 0.5, 0.8
    max_omega_size = 10
    num_generations = 1

    for _ in range(num_generations):
        if _ > 0:
            P_barre = [x for sublist in memeplex for x in sublist] + omega
        memeplex = construct_memeplexes(num_memeplexes, P_barre, memeplex_size)
        for memeplex_num in range(num_memeplexes):
            omega = search_within_memeplex(memeplex, num_iterations, memeplex_num, beta, nu, num_machines, num_jobs, omega, max_omega_size)

    return omega

# Final display of results
def display_final_results():
    omega = sfla()
    for i, solution in enumerate(omega):
        print(f"Solution {i}: TEC = {solution[1]}, WB = {solution[2]}")
    Cmax = calculate_Cmax(omega[0][0])
    display_results_by_machine(get_results_by_machine(omega[0][0], num_machines), num_machines)
    TEC_by_machine, total_TEC = calculate_energy(get_results_by_machine(omega[0][0], num_machines), Cmax)
    print(f"TEC by machine: {TEC_by_machine}\nTotal TEC: {total_TEC}")

# Execute the final display
display_final_results()