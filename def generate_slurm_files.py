def generate_slurm_files(num_runs, data_split):
    for i in range(num_runs):
        filename = f"slurm/run_{data_split}_{i}.slurm"
        with open(filename, "w") as f:
            template = f"""#! /bin/sh
#SBATCH --job-name=jailbreak_InSPEcT_{i}
#SBATCH --output=logs/jailbreak_InSPEcT_{i}.out
#SBATCH --error=logs/jailbreak_InSPEcT_{i}.err
#SBATCH --time=1-00:00:00      # 1 day
#SBATCH --gpus=1               # Request 4 GPU
#SBATCH --nodes=1
#SBATCH --account=gpu-research
#SBATCH --partition=killable

# /home/gamir/DER-Roei/dhgottesman/hallucasting/passage_eval/QA/aliases

python /home/morg/students/yahavt/InSPEcT/scripts/create_patching_outputs.py -m "google/gemma-2-2b-it" -d "{data_split}" -i {i}
"""
            f.write(template)

generate_slurm_files(100, "test")
