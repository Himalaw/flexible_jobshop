# coding=utf-8
from math import sqrt, ceil
from random import randint, random, choice
from copy import deepcopy

# Read table2D from a file
def read_table2D(filePath):
    with open(filePath, "r") as file:
        lines = file.read().split("\n")
    table2D = [line.split() for line in lines if line.strip() != ""]
    return table2D

# Process and extract data from table2D
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
            op_machines = [
                (job_data[idx + 1 + 2 * m], job_data[idx + 2 + 2 * m])
                for m in range(machine_count)
            ]
            job_ops.append(op_machines)
            idx += 1 + 2 * machine_count
        job_machine_times.append(job_ops)
    
    return table2D, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job, job_machine_times

# Available speeds
def available_speeds():
    return [0.25, 0.5, 1]

# Create an empty 2D list for job-machine times
def initialize_empty_2D_list(job_machine_times):
    return [[] for _ in range(len(job_machine_times))]

# Copy a 2D list
def copy_2D_list(data):
    return [list(inner) for inner in data]

# Copy a list of operations
def copy_operations(op_list):
    return list(op_list)

# Generate a random schedule
def generate_random_schedule(nb_jobs, job_machine_times):
    speeds = available_speeds()
    schedule = []
    empty_job_machine_times = initialize_empty_2D_list(job_machine_times)
    op_counts = [0] * nb_jobs

    while job_machine_times != empty_job_machine_times:
        available_jobs = [j for j in range(nb_jobs) if job_machine_times[j]]
        job = choice(available_jobs)
        op_counts[job] += 1
        machines = job_machine_times[job][0]
        machine = choice([m[0] for m in machines])
        speed = choice(speeds)
        schedule.append([job + 1, op_counts[job], machine, speed])
        del job_machine_times[job][0]
    
    return schedule

# Schedule operations and calculate times
def schedule_operations(schedule, nb_jobs, nb_machines, job_machine_times):
    job_times = [0] * nb_jobs
    machine_times = [0] * nb_machines
    results = []

    for job, op, machine, speed in schedule:
        job_idx = job - 1
        op_idx = op - 1
        machines = job_machine_times[job_idx][op_idx]
        proc_time = float(next(m[1] for m in machines if m[0] == machine))
        proc_time /= speed

        start_time = max(job_times[job_idx], machine_times[int(machine) - 1])
        end_time = start_time + proc_time
        job_times[job_idx] = end_time
        machine_times[int(machine) - 1] = end_time

        results.append([job, op, machine, speed, proc_time, start_time, end_time])
    
    return results

# Save results by machine
def save_results_by_machine(results, nb_machines):
    results_by_machine = [[] for _ in range(nb_machines)]
    for result in results:
        machine_idx = int(result[2]) - 1
        results_by_machine[machine_idx].append(result)
    return results_by_machine

# Save results by job
def save_results_by_job(results, nb_jobs):
    results_by_job = [[] for _ in range(nb_jobs)]
    for result in results:
        job_idx = result[0] - 1
        results_by_job[job_idx].append(result)
    return results_by_job

# Calculate Cmax
def calculate_cmax(results):
    return max(result[6] for result in results)

# Calculate standby time for each machine
def calculate_standby_time(results_by_machine, cmax):
    standby_times = []
    for machine_results in results_by_machine:
        if not machine_results:
            standby_times.append(cmax)
        else:
            total_standby = machine_results[0][5]
            for i in range(len(machine_results) - 1):
                total_standby += machine_results[i + 1][5] - machine_results[i][6]
            total_standby += cmax - machine_results[-1][6]
            standby_times.append(total_standby)
    return standby_times

# Calculate energy consumption (TEC)
def calculate_energy(results_by_machine, cmax):
    standby_times = calculate_standby_time(results_by_machine, cmax)
    se = 1
    tec_standby = [wait * se for wait in standby_times]
    tec_on = [
        sum(4 * result[4] * result[3] ** 2 for result in machine_results)
        for machine_results in results_by_machine
    ]
    total_tec = [standby + on for standby, on in zip(tec_standby, tec_on)]
    return total_tec, sum(total_tec)

# Calculate workload balance (WB)
def calculate_workload_balance(results_by_machine):
    workloads = [sum(result[4] for result in machine_results) for machine_results in results_by_machine]
    average_workload = sum(workloads) / len(results_by_machine)
    return sqrt(sum((workload - average_workload) ** 2 for workload in workloads))

# Display results by job
def display_results_by_job(results_by_job):
    print("\nResults by Job [job, operation, machine, speed, proc, start, end]:")
    for job_idx, job_results in enumerate(results_by_job, start=1):
        print(f"Job {job_idx}: {job_results}")

# Display results by machine
def display_results_by_machine(results_by_machine):
    print("\nResults by Machine [job, operation, machine, speed, proc, start, end]:")
    for machine_idx, machine_results in enumerate(results_by_machine, start=1):
        print(f"Machine {machine_idx + 1}: {machine_results}")

# Display TEC
def display_tec(tec_by_machine, total_tec):
    print("TEC by Machine:", tec_by_machine)
    print("Total TEC:", total_tec)

# Perform balancing operation
def perform_balancing(results_by_machine, saved_schedule, job_machine_times):
    workloads = [sum(result[4] for result in machine_results) for machine_results in results_by_machine]
    heaviest_machine = workloads.index(max(workloads)) + 1
    candidate_operations = [op for op in saved_schedule if op[2] == str(heaviest_machine)]

    if not candidate_operations:
        print("\nNo changes made.")
        return saved_schedule

    num_deps = len(candidate_operations)
    machine_choices = []

    for i in range(num_deps):
        job, op = candidate_operations[i][:2]
        machines = job_machine_times[job - 1][op - 1]
        machine_choices.append([m[0] for m in machines])

    min_workload = min(workloads)
    best_machine = None
    best_op_idx = None

    for i, choices in enumerate(machine_choices):
        for machine in choices:
            if workloads[int(machine) - 1] < min_workload:
                min_workload = workloads[int(machine) - 1]
                best_machine = machine
                best_op_idx = i
    
    if best_machine:
        op_idx = saved_schedule.index(candidate_operations[best_op_idx])
        saved_schedule[op_idx][2] = str(best_machine)
        print(f"\nJob {candidate_operations[best_op_idx][0]}, Operation {candidate_operations[best_op_idx][1]} moved from Machine {heaviest_machine} to {best_machine}.")
    else:
        print("\nNo changes made.")
    
    return saved_schedule

# Change operation order randomly
def change_operation_order(saved_schedule, job_machine_times, nb_jobs):
    modified_schedule = deepcopy(saved_schedule)
    num_ops = len(saved_schedule)
    op1, op2 = randint(0, num_ops - 1), randint(0, num_ops - 1)
    modified_schedule[op1], modified_schedule[op2] = modified_schedule[op2], modified_schedule[op1]

    while not check_validity(modified_schedule, nb_jobs):
        modified_schedule = deepcopy(saved_schedule)
        op1, op2 = randint(0, num_ops - 1), randint(0, num_ops - 1)
        modified_schedule[op1], modified_schedule[op2] = modified_schedule[op2], modified_schedule[op1]
    
    if op1 != op2:
        print(f"\nOperation {saved_schedule[op1][1]} of Job {saved_schedule[op1][0]} swapped with Operation {saved_schedule[op2][1]} of Job {saved_schedule[op2][0]}.")
    else:
        print("\nNo changes made.")
    
    return modified_schedule

# Check validity of the operation sequence
def check_validity(schedule, nb_jobs):
    expected_ops = [1] * nb_jobs

    for job, op, *_ in schedule:
        job_idx = job - 1
        if op != expected_ops[job_idx]:
            return False
        expected_ops[job_idx] += 1
    
    return True

# Modify the speed of a job operation
def modify_speed(schedule, job, op):
    modified_schedule = deepcopy(schedule)
    speeds = available_speeds()

    for i, (j, o, *rest) in enumerate(modified_schedule):
        if j == job and o == op:
            current_speed = rest[2]
            new_speed = choice([s for s in speeds if s != current_speed])
            modified_schedule[i][3] = new_speed
            print(f"Speed of Job {job}, Operation {op} changed from {current_speed} to {new_speed}.")
            break
    
    return modified_schedule

# Run the simulation once
def run_simulation():
    # Load data
    file_path = "Dataprojet/testlotmachine.fjs"
    table2D = read_table2D(file_path)
    table2D, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job, job_machine_times = process_data(table2D)

    # Generate initial random schedule
    initial_schedule = generate_random_schedule(nb_jobs, deepcopy(job_machine_times))
    saved_schedule = copy_operations(initial_schedule)
    results = schedule_operations(initial_schedule, nb_jobs, nb_machines, job_machine_times)

    # Calculate initial criteria
    results_by_machine = save_results_by_machine(results, nb_machines)
    results_by_job = save_results_by_job(results, nb_jobs)
    cmax = calculate_cmax(results)
    tec_by_machine, total_tec = calculate_energy(results_by_machine, cmax)
    wb = calculate_workload_balance(results_by_machine)

    # Display initial results
    display_results_by_job(results_by_job)
    display_results_by_machine(results_by_machine)
    print("Cmax =", cmax)
    display_tec(tec_by_machine, total_tec)
    print("WB:", wb)

    # Perform balancing and display new results
    balanced_schedule = perform_balancing(results_by_machine, saved_schedule, job_machine_times)
    balanced_results = schedule_operations(balanced_schedule, nb_jobs, nb_machines, job_machine_times)
    balanced_results_by_machine = save_results_by_machine(balanced_results, nb_machines)
    balanced_results_by_job = save_results_by_job(balanced_results, nb_jobs)
    balanced_cmax = calculate_cmax(balanced_results)
    balanced_tec_by_machine, balanced_total_tec = calculate_energy(balanced_results_by_machine, balanced_cmax)
    balanced_wb = calculate_workload_balance(balanced_results_by_machine)

    display_results_by_machine(balanced_results_by_machine)
    display_tec(balanced_tec_by_machine, balanced_total_tec)
    print("WB:", balanced_wb)
    print("\nImprovement in TEC:", total_tec - balanced_total_tec)
    print("Improvement in WB:", wb - balanced_wb)

    # Change operation order and display new results
    modified_schedule = change_operation_order(saved_schedule, job_machine_times, nb_jobs)
    modified_results = schedule_operations(modified_schedule, nb_jobs, nb_machines, job_machine_times)
    modified_results_by_machine = save_results_by_machine(modified_results, nb_machines)
    modified_results_by_job = save_results_by_job(modified_results, nb_jobs)
    modified_cmax = calculate_cmax(modified_results)
    modified_tec_by_machine, modified_total_tec = calculate_energy(modified_results_by_machine, modified_cmax)
    modified_wb = calculate_workload_balance(modified_results_by_machine)

    display_results_by_machine(modified_results_by_machine)
    display_tec(modified_tec_by_machine, modified_total_tec)
    print("WB:", modified_wb)
    print("\nImprovement in TEC:", total_tec - modified_total_tec)
    print("Improvement in WB:", wb - modified_wb)

    # Modify speed and display new results
    speed_modified_schedule = modify_speed(saved_schedule, 2, 1)
    speed_modified_results = schedule_operations(speed_modified_schedule, nb_jobs, nb_machines, job_machine_times)
    speed_modified_results_by_machine = save_results_by_machine(speed_modified_results, nb_machines)
    speed_modified_results_by_job = save_results_by_job(speed_modified_results, nb_jobs)
    speed_modified_cmax = calculate_cmax(speed_modified_results)
    speed_modified_tec_by_machine, speed_modified_total_tec = calculate_energy(speed_modified_results_by_machine, speed_modified_cmax)
    speed_modified_wb = calculate_workload_balance(speed_modified_results_by_machine)

    display_results_by_machine(speed_modified_results_by_machine)
    display_tec(speed_modified_tec_by_machine, speed_modified_total_tec)
    print("WB:", speed_modified_wb)
    print("\nImprovement in TEC:", total_tec - speed_modified_total_tec)
    print("Improvement in WB:", wb - speed_modified_wb)

# Execute the simulation
run_simulation()

'''SFLA'''

# Initialize parameters for SFLA
def initialize_parameters():
    population_size = 40
    num_memeplexes = 5
    num_iterations_per_memeplex = 5
    memeplex_size = ceil(population_size / num_memeplexes)
    return population_size, num_memeplexes, num_iterations_per_memeplex, memeplex_size

# Load data
file_path = "Dataprojet/Mk01.fjs"
table2D = read_table2D(file_path)
table2D, nb_jobs, nb_machines, avg_ops_per_machine, ops_per_job, job_machine_times = process_data(table2D)

# Generate initial population
def initialize_population(population_size, job_machine_times):
    population = []
    for _ in range(population_size):
        random_schedule = generate_random_schedule(nb_jobs, deepcopy(job_machine_times))
        results = schedule_operations(random_schedule, nb_jobs, nb_machines, job_machine_times)
        results_by_machine = save_results_by_machine(results, nb_machines)
        cmax = calculate_cmax(results)
        tec_by_machine, total_tec = calculate_energy(results_by_machine, cmax)
        wb = calculate_workload_balance(results_by_machine)
        population.append([results, total_tec, wb])
    return population

# Create non-dominated set omega
def create_omega(population):
    omega = []
    for i in range(len(population)):
        current = deepcopy(population[i])
        current_tec = population[i][1]
        current_wb = population[i][2]
        non_dominated = all(current_tec < other[1] or current_wb < other[2] for j, other in enumerate(population) if i != j)
        if non_dominated:
            omega.append(current)
    return omega

# Combine population and omega
def combine_population_and_omega(population, omega):
    combined = deepcopy(population) + deepcopy(omega)
    return combined

# Construct memeplexes
def construct_memeplexes(num_memeplexes, combined_population, memeplex_size):
    memeplexes = [[] for _ in range(num_memeplexes)]
    idx = 0

    while combined_population and idx < num_memeplexes:
        for _ in range(memeplex_size):
            if len(combined_population) > 1:
                candidate1, candidate2 = choice(combined_population), choice(combined_population)
                while candidate1 == candidate2:
                    candidate2 = choice(combined_population)

                if (candidate1[1] < candidate2[1] and candidate1[2] < candidate2[2]):
                    memeplexes[idx].append(candidate1)
                    combined_population.remove(candidate1)
                elif (candidate1[1] > candidate2[1] and candidate1[2] > candidate2[2]):
                    memeplexes[idx].append(candidate2)
                    combined_population.remove(candidate2)
                else:
                    selected = choice([candidate1, candidate2])
                    memeplexes[idx].append(selected)
                    combined_population.remove(selected)
            elif len(combined_population) == 1:
                memeplexes[idx].append(combined_population[0])
                combined_population.clear()
        idx += 1
    
    return memeplexes

'''Global Search'''

# Swap two operations
def swap_operations(x1, xbest):
    delta = 0.5
    x = deepcopy(x1)
    xb = deepcopy(xbest)
    new_list = []

    if xb != x:
        for _ in range(len(x[0])):
            alpha = random()
            if alpha < delta:
                temp = xb[0][0]
                new_list.append([temp[0], temp[1], temp[2], temp[3]])
                xb[0].pop(0)
                x[0] = [op for op in x[0] if op[0] != temp[0] or op[1] != temp[1]]
            else:
                temp = x[0][0]
                new_list.append([temp[0], temp[1], temp[2], temp[3]])
                x[0].pop(0)
                xb[0] = [op for op in xb[0] if op[0] != temp[0] or op[1] != temp[1]]
    else:
        new_list = [[op[0], op[1], op[2], op[3]] for op in xb[0]]

    return new_list

# Swap machine
def swap_machines(x1, xbest):
    x = deepcopy(x1)
    xb = deepcopy(xbest)
    num_ops = len(x[0])
    g1, g2 = randint(0, num_ops - 2), randint(g1 + 1, num_ops - 1)

    new_machine_list = []

    for i in range(num_ops):
        if g1 <= i <= g2:
            job, op = xb[0][i][:2]
            machine = next(op[2] for op in x[0] if op[0] == job and op[1] == op)
            new_machine_list.append([job, op, machine, xb[0][i][3]])
        else:
            new_machine_list.append(xb[0][i])

    return new_machine_list

# Swap speed
def swap_speeds(x1, xbest):
    x = deepcopy(x1)
    xb = deepcopy(xbest)
    num_ops = len(x[0])
    g1, g2 = randint(0, num_ops - 2), randint(g1 + 1, num_ops - 1)

    new_speed_list = []

    for i in range(num_ops):
        if g1 <= i <= g2:
            job, op = xb[0][i][:2]
            speed = next(op[3] for op in x[0] if op[0] == job and op[1] == op)
            new_speed_list.append([job, op, xb[0][i][2], speed])
        else:
            new_speed_list.append(xb[0][i])

    return new_speed_list

'''Neighborhood Search'''

# Insert operation at a new position
def insert_operation_at_new_position(xbest):
    x = deepcopy(xbest)
    flat_schedule = [op[0] for op in x[0]]
    j, k = randint(0, len(x[0]) - 1), randint(0, len(x[0]) - 1)

    while k == j:
        k = randint(0, len(x[0]) - 1)

    teta_j = flat_schedule.pop(j)
    flat_schedule.insert(k, teta_j)

    new_schedule = []

    for job in flat_schedule:
        new_schedule.append([job, 0])

    for job_idx in range(nb_jobs + 1):
        counter = 1
        for op in new_schedule:
            if op[0] == job_idx:
                op[1] = counter
                counter += 1

    for i, op in enumerate(new_schedule):
        job, op_num = op[:2]
        machine, speed = next((o[2], o[3]) for o in x[0] if o[0] == job and o[1] == op_num)
        new_schedule[i] = [job, op_num, machine, speed]

    return new_schedule

# Get a random machine for the operation
def get_random_machine(job, op, current_machine):
    if isinstance(job_machine_times[job - 1][op - 1][0], str):
        return current_machine

    machines = job_machine_times[job - 1][op - 1]
    random_machine = choice(machines)[0]

    if len(machines) > 1:
        while random_machine == current_machine:
            random_machine = choice(machines)[0]
    else:
        return current_machine

    return random_machine

# Change machines randomly for some operations
def change_machines_randomly(xbest):
    x = deepcopy(xbest)
    num_ops = randint(1, len(x[0]))
    ops_to_change = [randint(0, len(x[0]) - 1) for _ in range(num_ops)]

    new_schedule = deepcopy(x[0])

    for op_idx in ops_to_change:
        job, op, machine, speed = new_schedule[op_idx]
        new_machine = get_random_machine(job, op, machine)
        new_schedule[op_idx] = [job, op, new_machine, speed]

    return new_schedule

# Get a random speed different from the current speed
def get_random_speed(current_speed):
    speeds = available_speeds()
    new_speed = choice([s for s in speeds if s != current_speed])
    return new_speed

# Change speeds randomly for some operations
def change_speeds_randomly(xbest):
    x = deepcopy(xbest)
    num_ops = randint(1, len(x[0]))
    ops_to_change = [randint(0, len(x[0]) - 1) for _ in range(num_ops)]

    new_schedule = deepcopy(x[0])

    for op_idx in ops_to_change:
        job, op, machine, speed = new_schedule[op_idx]
        new_speed = get_random_speed(speed)
        new_schedule[op_idx] = [job, op, machine, new_speed]

    return new_schedule

# Display memeplexes
def display_memeplexes(memeplexes):
    for i, memeplex in enumerate(memeplexes, start=1):
        print(f"Memeplex {i}:")
        for sol in memeplex:
            print([sol[1], sol[2]])

# Randomly select x and xbest from a memeplex
def randomly_select_solutions(memeplex, numplex):
    candidates = deepcopy(memeplex[numplex])
    xbest = None

    for _ in range(len(candidates)):
        candidate = choice(candidates)
        candidates.remove(candidate)
        non_dominated = all(candidate[1] < sol[1] or candidate[2] < sol[2] for sol in memeplex[numplex])
        if non_dominated:
            xbest = candidate
            break

    x = choice([sol for sol in memeplex[numplex] if sol != xbest])

    return x, xbest

# Generate final results from a list of operations
def generate_final_results(schedule, nb_jobs, nb_machines, job_machine_times):
    results = schedule_operations(schedule, nb_jobs, nb_machines, job_machine_times)
    results_by_machine = save_results_by_machine(results, nb_machines)
    results_by_job = save_results_by_job(results, nb_jobs)
    cmax = calculate_cmax(results)
    tec_by_machine, total_tec = calculate_energy(results_by_machine, cmax)
    wb = calculate_workload_balance(results_by_machine)
    return [results, total_tec, wb]

# Search process within a memeplex
def search_process(memeplex, num_iterations, numplex, beta, nu, nb_machines, nb_jobs, omega, H):
    w = 1
    while w <= num_iterations:
        x1, xbest = randomly_select_solutions(memeplex, numplex)
        alpha = random()

        if alpha < beta:
            new_schedule = swap_operations(x1, xbest)
        elif beta <= alpha <= nu:
            new_schedule = swap_machines(x1, xbest)
        else:
            new_schedule = swap_speeds(x1, xbest)

        new_solution = generate_final_results(new_schedule, nb_jobs, nb_machines, job_machine_times)

        k = 1
        while new_solution[1] > xbest[1] and new_solution[2] > xbest[2] and k < 4:
            if k == 1:
                new_schedule = insert_operation_at_new_position(xbest)
            elif k == 2:
                new_schedule = change_machines_randomly(xbest)
            elif k == 3:
                new_schedule = change_speeds_randomly(xbest)
            k += 1
            new_solution = generate_final_results(new_schedule, nb_jobs, nb_machines, job_machine_times)

        if new_solution[1] < xbest[1] or new_solution[2] < xbest[2]:
            omega.append(new_solution)
            omega = [sol for sol in omega if not (sol[1] >= new_solution[1] and sol[2] >= new_solution[2])]
            if len(omega) > H:
                crowding_distances = [(1000, 0)] + [
                    (
                        abs(omega[i + 2][1] - omega[i][1]) / (abs(omega[0][1] - omega[-1][1]) + 0.00001)
                        + abs(omega[i + 2][2] - omega[i][2]) / (abs(omega[0][2] - omega[-1][2]) + 0.00001),
                        i + 1
                    )
                    for i in range(len(omega) - 2)
                ] + [(1000, len(omega) - 1)]

                crowding_distances.sort(reverse=True, key=lambda x: x[0])
                omega = [omega[i] for _, i in crowding_distances[:H]]

            memeplex[numplex][memeplex[numplex].index(xbest)] = new_solution

        w += 1

    return omega

# Perform SFLA
def sfla():
    population_size, num_memeplexes, num_iterations_per_memeplex, memeplex_size = initialize_parameters()
    initial_population = initialize_population(population_size, job_machine_times)
    omega = create_omega(initial_population)
    combined_population = combine_population_and_omega(initial_population, omega)
    memeplexes = construct_memeplexes(num_memeplexes, combined_population, memeplex_size)

    beta = 0.5
    nu = 0.8
    num_iterations = 150
    H = 10

    for _ in range(num_iterations):
        combined_population = memeplexes.copy()
        combined_population += omega
        memeplexes = construct_memeplexes(num_memeplexes, combined_population, memeplex_size)
        for numplex in range(num_memeplexes):
            omega = search_process(memeplexes, num_iterations_per_memeplex, numplex, beta, nu, nb_machines, nb_jobs, omega, H)

    return omega

# Display final results
def display_final_results():
    omega = sfla()
    for i, sol in enumerate(omega):
        print(f"Solutions {i}: TEC = {sol[1]}, WB = {sol[2]}")

# Execute SFLA and display results
display_final_results()