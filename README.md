# Flexible Job Shop Scheduling with Particle Swarm Optimization

![pso](https://github.com/Himalaw/flexible_jobshop/assets/174485780/997b8b26-ff16-4541-a399-dc15a8be768e)

## Abstract:

<p align="justify"> This repository presents a solution to the flexible job shop scheduling problem (FJSP) using a multi-objective particle swarm optimization algorithm (MOPSO). The primary objectives are to minimize the total energy consumption and workload balance of machines. This approach is compared to existing methods like the shuffled frog leaping algorithm (SFLA), highlighting its effectiveness in addressing the complexities of FJSP in a multi-objective context.</p>

## Repository Assets:

- [Shuffled Frog Leaping Implementation]()
- [Multi-objective Particle Swarm Optimisation]()
- [Benchmark Datasets](DATA)

## Business Context:

<p align="justify"> In modern manufacturing, companies face the challenge of producing customized products at lower costs and faster delivery times while also adhering to eco-friendly practices. Efficient scheduling is critical to meet these demands. The flexible job shop scheduling problem is particularly relevant as it allows for more dynamic and efficient use of resources, which is essential for optimizing production processes in unpredictable market conditions.</p>

## Problem Description:

<p align="justify"> The flexible job shop scheduling problem involves scheduling n jobs across m machines, where each job comprises several operations that can be processed on various compatible machines at different speeds. The goal is to optimize two main criteria: total energy consumption (TEC) and workload balance (WB). This problem is complex due to its NP-hard nature and the conflicting objectives of minimizing energy consumption while balancing the workload across machines. The proposed MOPSO algorithm aims to find a set of non-dominated solutions that provide an optimal trade-off between these criteria.</p>
