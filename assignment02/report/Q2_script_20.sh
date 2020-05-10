#!/bin/bash
#$ -P rse-com6012
#$ -q rse-com6012.q
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 20 #number of cores
#$ -l rmem=10G #number of memery
#$ -o Q2_output_20 #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M xxiao17@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit Q2_code.py
