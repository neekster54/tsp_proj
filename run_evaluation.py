import argparse
import csv
import platform

from tsp.instance import load_dataset
from evaluate import evaluate, TSP_ALGORITHMS 

OUTPUT_COLUMNS = [
    "instance", "n", "reps", "algorithm", "optimum", "length", "length_two_opt", "two_opt_moves", 
    "algo_time", "two_opt_time", "total_time", "ratio", "ratio_two_opt", "error", "error_two_opt", "status",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args()

    print(f"python {platform.python_version()} on {platform.platform()}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for inst in load_dataset():
            for algorithm in TSP_ALGORITHMS:
                try:
                    row, _, _ = evaluate(inst, algorithm, reps=args.reps)
                    row["status"] = "ok"
                except Exception as exc:
                    row = {"instance": inst.name, "n": inst.n, "algorithm": algorithm,
                           "status": f"failed: {exc!r}"}
                writer.writerow(row)
                f.flush()
                print(f"{inst.name:>10} {algorithm:>17} {row['status']:>4} ")


if __name__ == "__main__":
    main()