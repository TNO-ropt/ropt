#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name={{job_name}}
#SBATCH --output={{output}}
#SBATCH --chdir={{working_directory}}
#SBATCH --ntasks={{cores}}
# Make sure not all memory is claimed:
#SBATCH --mem-per-cpu=512M
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}

{{command}}
