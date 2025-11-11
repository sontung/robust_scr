filenames = [
    "/home/n11373598/hpc-home/work/glace_experiment/results/res-dept-1F-2025-07-11_12-31-55.txt",
    "/home/n11373598/hpc-home/work/glace_experiment/results/res-dept-B1-2025-07-11_19-09-14.txt",
    "/home/n11373598/hpc-home/work/glace_experiment/results/res-dept-4F-2025-07-11_05-01-41.txt",
]

output_filename = (
    "/home/n11373598/hpc-home/work/glace_experiment/results/merged_results.txt"
)

with open(output_filename, "w") as outfile:
    for fname in filenames:
        with open(fname, "r") as infile:
            for line in infile:
                outfile.write(line)
