import argparse
import csv
import platform

from tsp.instance import load_dataset
from evaluate import evaluate, MAX_CITIES, TSP_ALGORITHMS

OUTPUT_COLUMNS = [
    "instance", "n", "reps", "algorithm", "optimum", "length", "length_two_opt", "two_opt_moves", 
    "algo_time", "two_opt_time", "total_time", "ratio", "ratio_two_opt", "error", "error_two_opt",
    "length_three_opt", "three_opt_moves", "three_opt_time", "total_time_three_opt", "ratio_three_opt", "error_three_opt",
    "status",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", default="results/results.csv")
    parser.add_argument("--three-opt-max-n", type=int, default=150,
                        help="skip 3-opt on instances with more cities than this, since it is cubic per pass")
    args = parser.parse_args()

    print(f"python {platform.python_version()} on {platform.platform()}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for inst in load_dataset():
            for algorithm in TSP_ALGORITHMS:
                if inst.n > MAX_CITIES.get(algorithm, inst.n):
                    row = {"instance": inst.name, "n": inst.n, "algorithm": algorithm,
                           "status": f"skipped: n > {MAX_CITIES[algorithm]}"}
                else:
                    try:
                        row, _, _, _ = evaluate(inst, algorithm, reps=args.reps,
                                            three_opt=inst.n <= args.three_opt_max_n)
                        row["status"] = "ok"
                    except Exception as exc:
                        row = {"instance": inst.name, "n": inst.n, "algorithm": algorithm,
                               "status": f"failed: {exc!r}"}
                writer.writerow(row)
                f.flush()
                print(f"{inst.name:>10} {algorithm:>17} {row['status']:>4} ")


if __name__ == "__main__":
    main()