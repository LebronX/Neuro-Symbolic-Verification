import numpy as np
from marabou_encoding import marabouEncoding
import time
import csv
import argparse

# Information for noting the results in csv format
veri_info = [
    "Specification",
    "Network file",
    "Epsilon",
    "Delta",
    "Time taken",
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


def main():
    # prop1 = ["checking-sign", 14]
    # prop2 = ["checking-confidence", 0.05, 50]
    # prop3 = ["checking-equivalence", 0.1]
    # prop4 = ["checking-fairness", 0.05]
    # prop = prop2
    # output_name = ["output_nuv", "output_prop"]
    # network_file = "../../networks/gtsrb/gtsrb_confidence_fc.onnx"  # gtsrb_sign_fc /gtsrb_eq_fc / gtsrb_confidence_fc
    # print("------Checking properties of DNN, specification: " + prop[0] + "-------")

    parser = argparse.ArgumentParser(description="verification")
    parser.add_argument(
        "--network",
        type=int,
        default=1,
        help="input batch size for training (default: 100)",
    )
    args = parser.parse_args()

    prop1 = ["checking-sign"]
    prop2 = ["checking-confidence"]
    prop3 = ["checking-equivalence"]
    prop = prop1
    mara = marabouEncoding()

    if prop[0] == "checking-sign":
        # sign experiment
        for sign in range(0, 10):

            # writing csv file header
            csvfile = (
                "../../validate/gtsrb/gtsrb-"
                + prop[0]
                + "-exp"
                + str(args.network)
                + "-sign"
                + str(sign)
                + ".csv"
            )
            print(csvfile)
            write_csv_header(csvfile)

            network_file = (
                "../../networks/gtsrb/gtsrb_sign_fc_sign_"
                + str(sign)
                + "_exp"
                + str(args.network)
                + ".onnx"
            )
            prop.append(sign)
            print("------Checking properties of DNN Using Marabou-------")
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
            prop.pop()

    elif prop[0] == "checking-confidence":
        # writing csv file header
        csvfile = (
            "../../validate/gtsrb/gtsrb-"
            + prop[0]
            + "-exp"
            + str(args.network)
            + ".csv"
        )
        write_csv_header(csvfile)

        # confidence experiment
        # setting epsilon and delta
        for epsilon in np.linspace(0.05, 0.14, 10):
            for delta in range(1, 21, 1):

                network_file = (
                    "../../networks/gtsrb/gtsrb_confidence_fc_exp"
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

                prop.pop()
                prop.pop()

    elif prop[0] == "checking-equivalence":

        # writing csv file header
        csvfile = (
            "../../validate/gtsrb/gtsrb-"
            + prop[0]
            + "-exp"
            + str(args.network)
            + ".csv"
        )
        write_csv_header(csvfile)

        # equivalence experiment
        # setting epsilon
        for epsilon in np.linspace(0.05, 0.14, 10):
            network_file = (
                "../../networks/gtsrb/gtsrb_eq_fc_exp" + str(args.network) + ".onnx"
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


if __name__ == "__main__":
    main()
