# module for creating slurm jobs...
import os
import datetime

class SlurmJob:
    def __init__(self, time="1:00:00", cpus_per_task=4, mem_per_cpu=32, ntasks=1):
        self.time = time
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu = mem_per_cpu
        self.ntasks = ntasks

    def _format_time_now(self):
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%dT%H:%M:%S")

    def _makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def make_slurm_template(self, jobname="job", fname=None):

        if fname is None: cwd = os.getcwd()
        else: cwd = "/".join(fname.split("/"))[:-1]

        self._makedir(f"{cwd}/LOGS")
        outfname = f"{cwd}/LOGS/{jobname}_{self._format_time_now()}.log"
        if fname is None: fname = f"{cwd}/{jobname}_slurm.sh"

        slurm_context = f"""#!/bin/bash
#
#SBATCH --job-name={jobname}
#SBATCH --output={outfname}
#
#SBATCH --ntasks={self.ntasks}
#SBATCH --time={self.time}
#SBATCH --mem-per-cpu={self.mem_per_cpu}G
#SBATCH --cpus-per-task={self.cpus_per_task}

# input your real command below...
        """

        with open(fname, "w") as fp:
            fp.write(slurm_context)
        os.system(f"chmod +x {fname}")

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    
    parser = ArgumentParser(description='make slurm submission template', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--time", type=str, help="time to ask", default="1:00:00")
    parser.add_argument("--ncpu", type=int, help="number of cpus per task", default=4)
    parser.add_argument("-m", "--mem", type=int, help="memory per cpu", default=32)
    parser.add_argument("--ntask", type=int, help="number of tasks", default=1)
    parser.add_argument("-j", "--job", type=str, help="job names", default="job")

    args = parser.parse_args()

    slurmjob = SlurmJob(time=args.time, cpu_per_task=args.ncpu, mem_per_cpu=args.mem, ntasks=args.ntask)
    slurmjob.make_slurm_template(jobname=args.job)