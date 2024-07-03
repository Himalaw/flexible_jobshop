# coding=utf-8

# read_doc permet de stocker les données ligne par ligne d`un fichier vers un tableau,
# `filePath` étant son chemin
def read_doc(filePath):
    with open(filePath, "r") as doc:
        lines = doc.read().split("\n")
    data = [line.split() for line in lines if line.strip() != ""]
    return data

def parse_data(filePath):
    data = read_doc(filePath)
    nb_jobs = int(data[0][0])
    data = data[1:]  # Supprime la première ligne pour ne garder que les infos des jobs

    nb_tasks_per_job = [int(data[i][0]) for i in range(nb_jobs)]  # Récupère le nombre d'opérations de chaque job

    jobs_info = []
    for job_idx in range(nb_jobs):
        job_data = data[job_idx]
        job = []
        index = 1
        for _ in range(nb_tasks_per_job[job_idx]):
            num_machines = int(job_data[index])
            index += 1
            task = [
                [job_data[index + 2 * j], job_data[index + 2 * j + 1]]
                for j in range(num_machines)
            ]
            index += 2 * num_machines
            job.append(task)
        jobs_info.append(job)
    
    return jobs_info

# Exemple d'utilisation
file_path = "Data/01a.fjs"
jobs_data = parse_data(file_path)
print(jobs_data)
