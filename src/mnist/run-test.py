import time
import csv
from marabou_encoding import marabouEncoding
import numpy as np
import random
import argparse

# Information for noting the results in csv format
veri_info = [
    "Specification",
    "Network file",
    "Epsilon",
    "Delta",
    "Time taken",
    "Upper bound",
    "Lower bound",
    "SAT",
]
veri_dict = {i: None for i in veri_info}


def write_csv_header(csvfile):
    with open(csvfile, mode="w") as file:
        writer = csv.DictWriter(file, fieldnames=veri_info)
        writer.writeheader()


def write_line_csv(csvfile):
    with open(csvfile, mode="a") as file:
        writer = csv.DictWriter(file, fieldnames=veri_info)
        writer.writerow(veri_dict)


Timeout = 3600


def main():

    parser = argparse.ArgumentParser(description="verification")
    parser.add_argument(
        "--network",
        type=int,
        default=1,
        help="input batch size for training (default: 100)",
    )
    args = parser.parse_args()

    mean_digit = []
    with open(
        "../../TrainingData/mnist/eight_mean_pixel.txt", "r"
    ) as f:  # mean digit for eight now
        mean_digit = f.readlines()
        mean_digit = [float(i.strip("\n")) for i in mean_digit]

    # lower_bound = 0.2
    # upper_bound = 0.2
    # prop1 = ["checking-digit", 8]
    # prop2 = ["checking-confidence", 0.01, 10]
    # prop3 = ["checking-equivalence", 0.1]
    # prop4 = ["checking-fairness", 0.01, 8]
    # prop5 = ["checking-robustness", 0.01]
    # prop6 = ["checking-digit-safe-radius", lower_bound, upper_bound, mean_digit, "digit"]
    # prop7 = ["checking-autoencoder-safe-radius", lower_bound, upper_bound, mean_digit, "autoenc", 0.05]

    prop1 = ["checking-digit"]
    prop2 = ["checking-confidence"]
    prop3 = ["checking-equivalence"]
    prop4 = ["checking-digit-confidence"]
    prop5 = ["check-confidence-random-sampling"]
    prop6 = ["checking-confidence-counterexample"]
    prop = prop6
    mara = marabouEncoding()
    # network_file = "../../networks/mnist/mnist_digit_fc.onnx"  # mnist_digit_fc.onnx / mnist_confidence_fc / mnist_eq_fc / mnist_safe_region_fc / mnist_autoenc_safe_region_fc / mnist_adv_fc / mnist_fair_fc

    print("------Checking properties of DNN, specification: " + prop[0] + "-------")
    print("------Checking properties of DNN Using Marabou-------")

    # write csv header
    # csvfile = "../../validate/mnist/" + prop[0] + ".csv"

    # digit experiment
    if prop[0] == "checking-digit":
        for digit in range(10):
            # if digit == 2:
            #     continue

            csvfile = (
                "../../validate/mnist/mnist-"
                + prop[0]
                + "-exp"
                + str(args.network)
                + "-sign"
                + str(digit)
                + ".csv"
            )
            write_csv_header(csvfile)

            network_file = (
                "../../networks/mnist/mnist_digit"
                + str(digit)
                + "_exp"
                + str(args.network)
                + ".onnx"
            )
            prop.append(digit)

            veri_dict.update(
                {
                    "Specification": prop[0],
                    "Network file": network_file,
                }
            )

            # Verification procedure
            log_time = time.time()
            result = mara.checkProperties(
                prop=prop,
                networkFile=network_file,
            )

            sat = True if result == "sat" else False

            mara_time = time.time() - log_time
            print("mara time: " + str(mara_time))
            veri_dict.update(
                {
                    "Time taken": mara_time,
                    "SAT": sat,
                }
            )

            write_line_csv(csvfile)

    # confidence experiment
    # setting epsilon and delta
    elif prop[0] == "checking-confidence":

        csvfile = (
            "../../validate/mnist/mnist-"
            + prop[0]
            + "-exp"
            + str(args.network)
            + ".csv"
        )
        write_csv_header(csvfile)

        for epsilon in np.linspace(0.05, 0.14, 10):
            for delta in range(1, 21, 1):

                network_file = (
                    "../../networks/mnist/mnist_confidence_exp"
                    + str(args.network)
                    + ".onnx"
                )
                prop.append(epsilon)
                prop.append(delta)
                veri_dict.update(
                    {
                        "Specification": prop[0],
                        "Network file": network_file,
                        "Epsilon": epsilon,
                        "Delta": delta,
                    }
                )

                # Verification procedure
                log_time = time.time()
                result = mara.checkProperties(
                    prop=prop,
                    networkFile=network_file,
                )

                sat = True if result == "sat" else False

                mara_time = time.time() - log_time
                print("mara time: " + str(mara_time))
                veri_dict.update(
                    {
                        "Time taken": mara_time,
                        "SAT": sat,
                    }
                )

                write_line_csv(csvfile)

    # equivalence experiment
    # setting epsilon
    elif prop[0] == "checking-equivalence":

        csvfile = (
            "../../validate/mnist/mnist-"
            + prop[0]
            + "-exp"
            + str(args.network)
            + ".csv"
        )
        write_csv_header(csvfile)

        for epsilon in np.linspace(0.06, 0.14, 10):

            network_file = (
                "../../networks/mnist/mnist_eq_exp" + str(args.network) + ".onnx"
            )
            prop.append(epsilon)
            veri_dict.update(
                {
                    "Specification": prop[0],
                    "Network file": network_file,
                    "Epsilon": epsilon,
                }
            )

            # Verification procedure
            log_time = time.time()
            result = mara.checkProperties(
                prop=prop,
                networkFile=network_file,
            )

            print(result)
            sat = True if result == "sat" else False

            mara_time = time.time() - log_time
            print("mara time: " + str(mara_time))
            veri_dict.update(
                {
                    "Time taken": mara_time,
                    "SAT": sat,
                }
            )

            write_line_csv(csvfile)

    elif prop[0] == "checking-confidence-counterexample":

        for delta in range(1):

            network_file = (
                "../../networks/mnist/mnist_confidence_exp"
                + str(args.network)
                + ".onnx"
            )
            prop.append(delta)
            veri_dict.update(
                {
                    "Specification": prop[0],
                    "Network file": network_file,
                    "Delta": delta,
                }
            )

            # Verification procedure
            log_time = time.time()
            counterExample = mara.checkProperties(
                prop=prop,
                networkFile=network_file,
            )
            mara_time = time.time() - log_time
            print("mara time: " + str(mara_time))
            veri_dict.update(
                {
                    "Time taken": mara_time,
                }
            )

    # worst-case confidence experiment
    elif prop[0] == "checking-digit-confidence":
        epsilon = 0.02
        acc = 0.01
        for digit in range(9, 10):
            upper_bound = 10
            lower_bound = 0
            network_file = (
                "../../networks/mnist/mnist_" + str(digit) + "_10_digit_fc.onnx"
            )
            veri_dict.update(
                {
                    "Specification": prop[0],
                    "Network file": network_file,
                }
            )
            prop.append(digit)
            time_for_one_digit = time.time()

            while upper_bound - lower_bound > acc:
                mid = (upper_bound + lower_bound) / 2
                prop.append(mid)

                # Verification procedure
                log_time = time.time()
                result = mara.checkProperties(
                    prop=prop,
                    networkFile=network_file,
                )
                mara_time = time.time() - log_time
                print("mara time: " + str(mara_time))

                sat = True if result == "sat" else False

                veri_dict.update(
                    {
                        "Time taken": mara_time,
                        "Delta": mid,
                        "Upper bound": upper_bound,
                        "Lower bound": lower_bound,
                        "SAT": sat,
                    }
                )

                if result == "sat":
                    upper_bound = mid + epsilon
                else:
                    lower_bound = mid - epsilon
                write_line_csv(csvfile)

                prop.pop()
                if time.time() - time_for_one_digit > Timeout:
                    break
            prop.pop()
            break

    # random sampling
    elif prop[0] == "check-confidence-random-sampling":
        upper_bound = 10
        lower_bound = 0
        for digit in range(9, 10):
            time_for_one_digit = time.time()
            digit_upper = upper_bound
            digit_lower = lower_bound

            network_file = (
                "../../networks/mnist/mnist_" + str(digit) + "_10_digit_fc.onnx"
            )
            veri_dict.update(
                {
                    "Specification": prop[0],
                    "Network file": network_file,
                }
            )
            prop.append(digit)

            while True:
                curr_point = random.uniform(lower_bound, upper_bound)
                prop.append(curr_point)

                log_time = time.time()
                result = mara.checkProperties(
                    prop=prop,
                    networkFile=network_file,
                )
                mara_time = time.time() - log_time

                if (
                    result == "sat"
                    and curr_point < digit_upper
                    and curr_point > digit_lower
                ):
                    digit_upper = curr_point
                elif (
                    result == "unsat"
                    and curr_point > digit_lower
                    and curr_point < digit_upper
                ):
                    digit_lower = curr_point

                sat = True if result == "sat" else False

                veri_dict.update(
                    {
                        "Time taken": mara_time,
                        "Delta": curr_point,
                        "Upper bound": digit_upper,
                        "Lower bound": digit_lower,
                        "SAT": sat,
                    }
                )
                write_line_csv(csvfile)

                prop.pop()
                if time.time() - time_for_one_digit > Timeout:
                    break
            prop.pop()
            break


if __name__ == "__main__":
    main()
