# coding=utf-8
from math import sqrt, ceil
from random import randint, choice, random
from PSO import *  # Ensure you have this module

# Function to read a 2D table from a file
def read_2D_table(file_path):
    with open(file_path, "r") as file:
        lines = file.read().split("\n")
    return [line.split() for line in lines if line.strip()]

# Process data from the table and extract relevant information
def extract_data(table2D):
    num_jobs = int(table2D[0][0])
    num_machines = int(table2D[0][1])
    avg_ops_per_machine = float(table2D[0][2])
    
    ops_per_job = [int(table2D[line + 1][0]) for line in range(num_jobs)]
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
    return [0.25, 0.5, 1.0]

# Check if a machine is available
def is_machine_available(machine_index, machine_avail):
    return int(machine_avail[machine_index - 1]) == 1

# Initialize all machines as available
def init_machines(num_machines):
    return [1] * num_machines + [0]

# Initialize all jobs as available
def init_jobs(num_jobs):
    return [1] * num_jobs

# Check if a job is available
def is_job_available(job_index, job_avail):
    return int(job_avail[job_index - 1]) == 1

# Choose a machine from available options
def choose_machine(operation_options, machine_avail):
    speed_list = get_possible_speeds()
    speed = choice(speed_list)
    possible_machines = []

    if operation_options:
        if isinstance(operation_options[0], list):
            for option in operation_options:
                if is_machine_available(int(option[0]), machine_avail):
                    possible_machines.append(option)
            chosen_machine = choice(possible_machines) if possible_machines else []
        else:
            chosen_machine = operation_options if is_machine_available(int(operation_options[0]), machine_avail) else []
    else:
        chosen_machine = []

    if chosen_machine:
        chosen_machine[1] = float(chosen_machine[1]) / speed

    return chosen_machine, speed

# Get the first operations for each job
def get_initial_operations(couples_2D, num_jobs):
    initial_operations = [couples_2D[i].pop(0) for i in range(num_jobs)]
    return initial_operations

# Choose a random job from available jobs
def choose_random_job(job_list, job_avail):
    available_indices = [i for i, available in enumerate(job_avail) if available]
    chosen_index = choice(available_indices) if available_indices else 10000
    chosen_job = job_list[chosen_index] if chosen_index < 10000 else []
    return chosen_job, chosen_index

# Choose an operation for a job with an available machine
def choose_operation_for_job(job_list, machine_avail, job_avail):
    operation_chosen = []
    speed = 0
    chosen_index = 10000

    if any(job_avail) and any(machine_avail):
        while not operation_chosen and job_list:
            chosen_job, chosen_index = choose_random_job(job_list, job_avail)
            operation_chosen, speed = choose_machine(chosen_job, machine_avail)
            if not operation_chosen:
                job_list.pop(chosen_index)

    return operation_chosen, chosen_index, speed

# Schedule jobs and update lists
def schedule_jobs(job_list, machine_avail, couples_2D, job_avail):
    operation_chosen, chosen_index, speed = choose_operation_for_job(job_list, machine_avail, job_avail)

    if operation_chosen:
        machine_avail[int(operation_chosen[0]) - 1] = 0
        job_avail[chosen_index] = 0
        if couples_2D[chosen_index]:
            job_list[chosen_index] = couples_2D[chosen_index].pop(0)
        else:
            job_list[chosen_index] = [str(len(machine_avail)), '0']

    return job_list, operation_chosen, chosen_index, couples_2D, machine_avail, speed

# Initialize time lists for jobs and machines
def init_time_lists(num_jobs, num_machines):
    job_times = [1000000] * num_jobs
    machine_release_times = [1000000] * num_machines
    return job_times, machine_release_times

# Update job and machine release times
def update_times(job_times, machine_release_times, chosen_index, operation_chosen, current_time):
    if operation_chosen:
        job_times[chosen_index] = float(operation_chosen[1]) + current_time
        machine_release_times[int(operation_chosen[0]) - 1] = float(operation_chosen[1]) + current_time
    return job_times, machine_release_times

# Initialize result lists for machines and jobs
def init_results(num_machines, num_jobs):
    return [[] for _ in range(num_machines)], [[] for _ in range(num_jobs)]

# Update scheduling results
def update_results(results, operation_chosen, chosen_index, current_time, speed):
    machine_index = int(operation_chosen[0]) - 1
    results[machine_index].append([chosen_index + 1, operation_chosen[1], current_time, speed])
    return results

def update_job_results(job_results, operation_chosen, chosen_index, current_time, speed):
    job_results[chosen_index].append([operation_chosen[0], operation_chosen[1], current_time, speed])
    return job_results

# Get the minimum time between two lists
def get_minimum_time(job_times, machine_release_times):
    return min(min(job_times), min(machine_release_times))

# Schedule all possible jobs at the current time
def schedule_all_jobs(job_list, machine_avail, couples_2D, job_times, machine_release_times, job_avail, current_time, results, job_results, operation_order):
    job_list, operation_chosen, chosen_index, couples_2D, machine_avail, speed = schedule_jobs(job_list, machine_avail, couples_2D, job_avail)
    job_times, machine_release_times = update_times(job_times, machine_release_times, chosen_index, operation_chosen, current_time)
    if operation_chosen:
        operation_order.append(chosen_index + 1)
        results = update_results(results, operation_chosen, chosen_index, current_time, speed)
        job_results = update_job_results(job_results, operation_chosen, chosen_index, current_time, speed)

    while operation_chosen:
        job_list, operation_chosen, chosen_index, couples_2D, machine_avail, speed = schedule_jobs(job_list, machine_avail, couples_2D, job_avail)
        job_times, machine_release_times = update_times(job_times, machine_release_times, chosen_index, operation_chosen, current_time)
        if operation_chosen:
            operation_order.append(chosen_index + 1)
            results = update_results(results, operation_chosen, chosen_index, current_time, speed)
            job_results = update_job_results(job_results, operation_chosen, chosen_index, current_time, speed)

    return job_list, couples_2D, machine_avail, job_times, machine_release_times, results, job_results, operation_order

# Release jobs and machines when their times are up
def release_jobs_and_machines(job_times, machine_avail, job_avail, machine_release_times):
    job_index = job_times.index(min(job_times))
    machine_index = machine_release_times.index(min(machine_release_times))
    machine_avail[machine_index] = 1
    job_avail[job_index] = 1
    job_times[job_index] = 1000000
    machine_release_times[machine_index] = 1000000
    return job_times, machine_avail, job_avail, machine_release_times

# Generate a random schedule
def generate_random_schedule(file_path):
    table2D, num_jobs, num_machines, avg_ops_per_machine, ops_per_job, couples_2D = extract_data(read_2D_table(file_path))
    machine_avail = init_machines(num_machines)
    job_avail = init_jobs(num_jobs)
    operation_order = []
    job_list = get_initial_operations(couples_2D, num_jobs)
    job_times, machine_release_times = init_time_lists(num_jobs, num_machines)
    results, job_results = init_results(num_machines, num_jobs)
    current_time = 0

    job_list, couples_2D, machine_avail, job_times, machine_release_times, results, job_results, operation_order = schedule_all_jobs(
        job_list, machine_avail, couples_2D, job_times, machine_release_times, job_avail, current_time, results, job_results, operation_order
    )

    while job_list != [[str(num_machines + 1), '0']] * num_jobs:
        current_time = get_minimum_time(job_times, machine_release_times)
        job_times, machine_avail, job_avail, machine_release_times = release_jobs_and_machines(job_times, machine_avail, job_avail, machine_release_times)
        job_list, couples_2D, machine_avail, job_times, machine_release_times, results, job_results, operation_order = schedule_all_jobs(
            job_list, machine_avail, couples_2D, job_times, machine_release_times, job_avail, current_time, results, job_results, operation_order
        )

    return results, job_results, num_machines, num_jobs, operation_order

# Display scheduling results for machines
def display_machine_schedule(results, num_machines):
    print("\nMachine Scheduling Results [job, proc_time, start_time, speed]:")
    for i in range(num_machines):
        print(f"Machine {i + 1}: {results[i]}")

# Display scheduling results for jobs
def display_job_schedule(job_results, num_jobs):
    print("\nJob Scheduling Results [machine, proc_time, start_time, speed]:")
    for i in range(num_jobs):
        print(f"Job {i + 1}: {job_results[i]}")

# Calculate Cmax
def calculate_Cmax(results):
    Cmax = 0
    last_machine = -1

    for i, machine_results in enumerate(results):
        if machine_results:
            end_time = machine_results[-1][2] + machine_results[-1][1]
            if end_time > Cmax:
                Cmax = end_time
                last_machine = i + 1

    return Cmax, last_machine

# Calculate machine standby times
def calculate_standby_times(results):
    Cmax, last_machine = calculate_Cmax(results)
    total_wait_times = [0] * len(results)

    for i, machine_results in enumerate(results):
        if machine_results:
            total_wait_times[i] = machine_results[0][2]
            for j in range(len(machine_results) - 1):
                total_wait_times[i] += machine_results[j + 1][2] - (machine_results[j][2] + machine_results[j][1])
            total_wait_times[i] += Cmax - (machine_results[-1][2] + machine_results[-1][1])
        else:
            total_wait_times[i] = Cmax

    return total_wait_times

# Calculate and display energy consumption
def calculate_energy(results):
    standby_times = calculate_standby_times(results)
    SE = [1] * len(results)
    TEC_on = [0] * len(results)
    TEC_standby = [standby * se for standby, se in zip(standby_times, SE)]
    TEC = []

    for i, machine_results in enumerate(results):
        for job in machine_results:
            TEC_on[i] += 4 * job[3] ** 2 * job[1]
        TEC.append(TEC_standby[i] + TEC_on[i])

    total_TEC = sum(TEC)
    print(f"Energy Consumption per Machine: {TEC}")
    print(f"Total Energy Consumption (TEC): {total_TEC}")

    return TEC, total_TEC

# Calculate and display workload balance (WB)
def calculate_workload_balance(results):
    workloads = [sum(job[1] for job in machine_results) for machine_results in results]
    average_workload = sum(workloads) / len(results)
    WB = sqrt(sum((workload - average_workload) ** 2 for workload in workloads))

    print(f"Workload Balance (WB): {WB}")
    return WB

# Function to copy job results
def copy_job_results(job_results):
    return [list(results) for results in job_results]

# Create a list of scheduling information in order
def create_schedule_list(job_results, operation_order):
    copied_job_results = copy_job_results(job_results)
    schedule_list = []

    for job in operation_order:
        schedule_list.append([job, int(copied_job_results[job - 1][0][0]), copied_job_results[job - 1][0][1], copied_job_results[job - 1][0][3]])
        copied_job_results[job - 1].pop(0)

    return schedule_list

# Schedule jobs based on a given order
def schedule_jobs_in_order(schedule_list, machine_avail, job_avail):
    if schedule_list:
        machine_wanted = schedule_list[0][1]
        job_wanted = schedule_list[0][0]
        if is_machine_available(machine_wanted, machine_avail) and is_job_available(job_wanted, job_avail):
            job_avail[job_wanted - 1] = 0
            machine_avail[machine_wanted - 1] = 0
            operation_chosen = [schedule_list[0][1], schedule_list[0][2]]
            speed = schedule_list[0][3]
            schedule_list.pop(0)
        else:
            operation_chosen = []
            speed = 0
    else:
        operation_chosen = []
        speed = 0

    return schedule_list, operation_chosen, job_avail, machine_avail, speed

# Schedule all possible jobs based on a given order
def schedule_all_possible_jobs(schedule_list, machine_avail, job_times, machine_release_times, job_avail, current_time, results, job_results):
    schedule_list, operation_chosen, job_avail, machine_avail, speed = schedule_jobs_in_order(schedule_list, machine_avail, job_avail)
    job_times, machine_release_times = update_times(job_times, machine_release_times, chosen_index, operation_chosen, current_time)
    if operation_chosen:
        results = update_results(results, operation_chosen, chosen_index, current_time, speed)
        job_results = update_job_results(job_results, operation_chosen, chosen_index, current_time, speed)

    while operation_chosen:
        schedule_list, operation_chosen, job_avail, machine_avail, speed = schedule_jobs_in_order(schedule_list, machine_avail, job_avail)
        job_times, machine_release_times = update_times(job_times, machine_release_times, chosen_index, operation_chosen, current_time)
        if operation_chosen:
            results = update_results(results, operation_chosen, chosen_index, current_time, speed)
            job_results = update_job_results(job_results, operation_chosen, chosen_index, current_time, speed)

    return schedule_list, job_times, machine_release_times, results, job_results

# Generate a schedule based on a given operation order
def generate_schedule_from_order(operation_order, job_results, results, num_jobs, num_machines):
    machine_avail = init_machines(num_machines)
    job_avail = init_jobs(num_jobs)
    schedule_list = create_schedule_list(job_results, operation_order)
    job_times, machine_release_times = init_time_lists(num_jobs, num_machines)
    current_time = 0

    schedule_list, job_times, machine_release_times, results, job_results = schedule_all_possible_jobs(
        schedule_list, machine_avail, job_times, machine_release_times, job_avail, current_time, results, job_results
    )

    while schedule_list:
        current_time = get_minimum_time(job_times, machine_release_times)
        job_times, machine_avail, job_avail, machine_release_times = release_jobs_and_machines(job_times, machine_avail, job_avail, machine_release_times)
        schedule_list, job_times, machine_release_times, results, job_results = schedule_all_possible_jobs(
            schedule_list, machine_avail, job_times, machine_release_times, job_avail, current_time, results, job_results
        )

    return results, job_results, operation_order

# Display the final results after scheduling
def display_final_results(operation_order, results, job_results, num_jobs, num_machines):
    print("///////////////////////////////////////////////////////")
    print("New Schedule")
    print(operation_order)
    display_machine_schedule(results, num_machines)
    display_job_schedule(job_results, num_jobs)
    TEC, total_TEC = calculate_energy(results)
    WB = calculate_workload_balance(results)
    return total_TEC, WB

# Copy a 1D list
def copy_1D_list(lst):
    return lst[:]

# Calculate and display improvement in WB and TEC after optimization
def compare_results(initial_TEC, initial_WB, new_TEC, new_WB):
    improvement_WB = initial_WB - new_WB
    improvement_TEC = initial_TEC - new_TEC
    print(f"Improvement in TEC: {improvement_TEC}")
    print(f"Improvement in WB: {improvement_WB}")
    return improvement_WB, improvement_TEC

# Operators for scheduling optimization
def choose_new_order(operation_order):
    new_order = []
    print(f"Enter new operation order (old order: {operation_order}):")
    order_input = input()
    for i in range(len(operation_order)):
        new_order.append(int(order_input[i]))
    return new_order

def order_jobs_ascending(operation_order):
    new_order = copy_1D_list(operation_order)
    new_order.sort()
    return new_order

def choose_start_job(operation_order, num_jobs):
    new_order = copy_1D_list(operation_order)
    print(f"Enter the job number to start with (1 to {num_jobs}):")
    start_job = int(input())
    start_pos = new_order.index(start_job)
    new_order[0], new_order[start_pos] = new_order[start_pos], new_order[0]
    return new_order

def choose_end_job(operation_order, num_jobs):
    new_order = copy_1D_list(operation_order)
    print(f"Enter the job number to end with (1 to {num_jobs}):")
    end_job = int(input())
    end_pos = new_order.index(end_job)
    new_order[-1], new_order[end_pos] = new_order[end_pos], new_order[-1]
    return new_order

# Main program
file_path = "Dataprojet/test.fjs"

# Generate random schedule
results, job_results, num_machines, num_jobs, operation_order = generate_random_schedule(file_path)

# Choose an operator to optimize the schedule
operation_order_v2 = order_jobs_ascending(operation_order)

# Generate optimized schedule
results_v2, job_results_v2, operation_order_v2 = generate_schedule_from_order(operation_order_v2, job_results, results, num_jobs, num_machines)

# Display initial schedule
print("Initial Schedule")
print(operation_order)
display_machine_schedule(results, num_machines)
display_job_schedule(job_results, num_jobs)

# Display optimized schedule
new_TEC, new_WB = display_final_results(operation_order_v2, results_v2, job_results_v2, num_jobs, num_machines)

# Compare initial and optimized results
initial_TEC, initial_WB = calculate_energy(results)[1], calculate_workload_balance(results)
compare_results(initial_TEC, initial_WB, new_TEC, new_WB)
