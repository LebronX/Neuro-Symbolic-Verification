from maraboupy import Marabou
from maraboupy import MarabouCore
import torch
from torchvision import datasets, transforms
import time


class marabouEncoding:
    def __init__(self):
        self.var = {}

    def checkProperties(self, prop, networkFile):
        # Reading DNN using our own version of reading onnx file
        network_verified = Marabou.read_onnx_deepproperty(networkFile)
        # network_verified = Marabou.read_onnx(networkFile)

        if prop[0] == "checking-digit":
            print("-----------checking target digit----------")
            return self.checkDigit(network_verified, prop[1])
        elif prop[0] == "checking-confidence":
            print("-----------checking confidence ----------")
            return self.checkConfidence(network_verified, prop[1], prop[2])
        elif prop[0] == "checking-equivalence":
            print("-----------checking equivalence of two network----------")
            return self.checkEq(network_verified, prop[1])
        elif (
            prop[0] == "checking-digit-confidence"
            or prop[0] == "check-confidence-random-sampling"
        ):
            print("-----------checking target digit confidence----------")
            return self.checkDigitConfidence(network_verified, prop[1], prop[2])
        elif prop[0] == "checking-fairness":
            print("-----------checking fairness----------")
            self.checkFair(network_verified, prop[1], prop[2])
        elif prop[0] == "checking-digit-safe-radius":
            print("-----------checking digit radius----------")
            self.checkSafeRadius(network_verified, prop[1], prop[2], prop[3], prop[4])
        elif prop[0] == "checking-autoencoder-safe-radius":
            print("-----------checking autoencoder radius----------")
            self.checkSafeRadius(
                network_verified, prop[1], prop[2], prop[3], prop[4], prop[5]
            )
        elif prop[0] == "checking-confidence-counterexample":
            print("-----------checking confidence counterexample ----------")
            self.checkConfidenceCounterexample(network_verified, prop[1])

    def checkDigit(self, network_verified, number):
        # This DNN outputs two elements, one is the prediction(dim = 10), the other one is true/false
        inputVars_verified = network_verified.inputVars[0]  # 784
        outputVars_verified = network_verified.outputVars  # 2*(10and2)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        # Encoding specification network, when specification network consider this digit is indeed the one wanted
        eq_prop = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq_prop.addAddend(1, outputVars_verified[1][1])
        eq_prop.addAddend(-1, outputVars_verified[1][0])
        eq_prop.setScalar(1)  # confidence level? IDK, cause no softmax/relu

        disjunction = [[eq_prop]]
        network_verified.addDisjunctionConstraint(disjunction)

        # Encoding NUV, if any other digit have higher confidence
        disjunction = []
        for i in range(len(outputVars_verified[0])):
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][i])
            eq_verified.addAddend(-1, outputVars_verified[0][number])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
        network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

        if vals:
            return "sat"
        else:
            return "unsat"

    def checkConfidence(self, network_verified, epsilon, delta):
        inputVars_verified = network_verified.inputVars[0]  # 784 / 14*14
        outputVars_verified = network_verified.outputVars  # 2*(10and784)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        # Encoding specification network, l-inf norm
        for i in range(len(outputVars_verified[1])):

            eq_property_1 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq_property_1.addAddend(1, outputVars_verified[1][i])
            eq_property_1.addAddend(-1, inputVars_verified[i])
            eq_property_1.setScalar(epsilon)
            network_verified.addDisjunctionConstraint([[eq_property_1]])

            eq_property_2 = MarabouCore.Equation(MarabouCore.Equation.LE)
            eq_property_2.addAddend(1, inputVars_verified[i])
            eq_property_2.addAddend(-1, outputVars_verified[1][i])
            eq_property_2.setScalar(epsilon)
            network_verified.addDisjunctionConstraint([[eq_property_2]])

        # Encoding NUV, assume the input image should be classified as a specific digit, e.g., eight
        for j in range(len(outputVars_verified[0])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][8])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        # Encoding NUV, mean difference between max_val and all the others
        disjunction = []
        eq_verified = MarabouCore.Equation(MarabouCore.Equation.LE)
        for j in range(len(outputVars_verified[0])):
            eq_verified.addAddend(1, outputVars_verified[0][8])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
        eq_verified.setScalar(delta * 9)
        disjunction.append([eq_verified])
        network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

        if vals:
            return "sat"
        else:
            return "unsat"

    def checkConfidenceCounterexample(self, network_verified, delta):
        # For generating random noise

        inputVars_verified = network_verified.inputVars[0]  # 784 / 14*14
        outputVars_verified = network_verified.outputVars  # 2 * (10 and 196)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        # Encoding specification network, l-inf norm
        # for i in range(len(outputVars_verified[1])):

        #     eq_property_1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        #     eq_property_1.addAddend(1, outputVars_verified[1][i])
        #     eq_property_1.addAddend(-1, inputVars_verified[i])
        #     eq_property_1.setScalar(epsilon)
        #     network_verified.addDisjunctionConstraint([[eq_property_1]])

        #     eq_property_2 = MarabouCore.Equation(MarabouCore.Equation.LE)
        #     eq_property_2.addAddend(1, inputVars_verified[i])
        #     eq_property_2.addAddend(-1, outputVars_verified[1][i])
        #     eq_property_2.setScalar(epsilon)
        #     network_verified.addDisjunctionConstraint([[eq_property_2]])

        # Encoding NUV, assume the input image should be classified as a specific digit, e.g., eight
        for j in range(len(outputVars_verified[0])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][8])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        # Encoding NUV, mean difference between max_val and all the others
        disjunction = []
        eq_verified = MarabouCore.Equation(MarabouCore.Equation.LE)
        for j in range(len(outputVars_verified[0])):
            eq_verified.addAddend(1, outputVars_verified[0][8])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
        eq_verified.setScalar(delta * 9)
        disjunction.append([eq_verified])
        network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

    def checkDigitConfidence(self, network_verified, digit, delta):
        inputVars_verified = network_verified.inputVars[0]  # 784 / 14*14
        outputVars_verified = network_verified.outputVars  # 2*(10and2)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        # Encoding specification network, when specification network consider this digit is indeed the one wanted
        eq_prop = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq_prop.addAddend(1, outputVars_verified[1][1])
        eq_prop.addAddend(-1, outputVars_verified[1][0])
        eq_prop.setScalar(1)

        # Encoding NUV, assume the input image should be classified as a specific digit, e.g., eight
        for j in range(len(outputVars_verified[0])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][digit])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        # Encoding NUV, mean difference between max_val and all the others
        disjunction = []
        eq_verified = MarabouCore.Equation(MarabouCore.Equation.LE)
        for j in range(len(outputVars_verified[0])):
            eq_verified.addAddend(1, outputVars_verified[0][digit])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
        eq_verified.setScalar(delta * 9)
        disjunction.append([eq_verified])
        network_verified.addDisjunctionConstraint(disjunction)

        option = Marabou.createOptions(timeoutInSeconds=600)
        vals, stats = network_verified.solve(options=option)

        return "unsat" if not vals else "sat"

    def checkEq(self, network_verified, epsilon):
        inputVars_verified = network_verified.inputVars[0]  # 784
        outputVars_verified = network_verified.outputVars  # 2*(10and10)

        print(outputVars_verified)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        disjunction = []
        for i in range(len(outputVars_verified[1])):

            eq_property_1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_property_1.addAddend(1, outputVars_verified[1][i])
            eq_property_1.addAddend(-1, outputVars_verified[0][i])
            eq_property_1.setScalar(epsilon)
            disjunction.append([eq_property_1])

            eq_property_2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_property_2.addAddend(1, outputVars_verified[0][i])
            eq_property_2.addAddend(-1, outputVars_verified[1][i])
            eq_property_2.setScalar(epsilon)
            disjunction.append([eq_property_2])

        network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

        if vals:
            return "sat"
        else:
            return "unsat"

    def checkFair(self, network_verified, epsilon, delta):
        inputVars_verified = network_verified.inputVars  # 2*(784and784)
        outputVars_verified = network_verified.outputVars  # 2*(10and10)

        for i in range(len(inputVars_verified)):
            for j in range(len(inputVars_verified[0])):
                network_verified.setLowerBound(inputVars_verified[i][j], -0.4242)
                network_verified.setUpperBound(inputVars_verified[i][j], 2.8)

        # Non-sensitive feature be the same
        for i in range(1, len(inputVars_verified[0])):
            disjunction = []
            eq_property_1 = MarabouCore.Equation(MarabouCore.Equation.EQ)
            eq_property_1.addAddend(1, inputVars_verified[0][i])
            eq_property_1.addAddend(-1, inputVars_verified[1][i])
            eq_property_1.setScalar(0)
            disjunction.append([eq_property_1])
            network_verified.addDisjunctionConstraint(disjunction)

        # Sensitive feature different, index 0 for now
        disjunction = []
        eq_property_2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq_property_2.addAddend(1, inputVars_verified[0][0])
        eq_property_2.addAddend(-1, inputVars_verified[1][0])
        eq_property_2.setScalar(epsilon)
        disjunction.append([eq_property_2])
        network_verified.addDisjunctionConstraint(disjunction)

        # Encoding NUV, assume the input image should be classified as digit eight, e.g., eight
        for i in range(len(outputVars_verified[0])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][8])
            eq_verified.addAddend(-1, outputVars_verified[0][i])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        # Encoding the output of property network, the counter example is classified as seven currently
        for i in range(len(outputVars_verified[1])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[1][7])
            eq_verified.addAddend(-1, outputVars_verified[1][i])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

    def checkSafeRadius(
        self,
        network_verified,
        lower_bound,
        upper_bound,
        mean_digit,
        prop_net_type,
        autoenc_diff=0,
    ):
        inputVars_verified = network_verified.inputVars[0]  # 784
        outputVars_verified = network_verified.outputVars  # 2 or 784

        # Encoding input region
        for i in range(len(inputVars_verified)):
            final_lower_bound = (
                0 if mean_digit[i] - lower_bound < 0 else mean_digit[i] - lower_bound
            )
            final_upper_bound = (
                1 if mean_digit[i] + upper_bound > 1 else mean_digit[i] + upper_bound
            )
            network_verified.setLowerBound(inputVars_verified[i], final_lower_bound)
            network_verified.setUpperBound(inputVars_verified[i], final_upper_bound)

        # Encoding output
        if prop_net_type == "digit":

            eq_prop = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_prop.addAddend(1, outputVars_verified[0])
            eq_prop.addAddend(-1, outputVars_verified[1])
            eq_prop.setScalar(0)
            disjunction = [[eq_prop]]
            network_verified.addDisjunctionConstraint(disjunction)

        elif prop_net_type == "autoenc":
            disjunction = []
            # l-inf, any pixel violate distance
            for i in range(len(outputVars_verified)):

                eq_prop1 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq_prop1.addAddend(1, outputVars_verified[i])
                eq_prop1.addAddend(-1, inputVars_verified[i])
                eq_prop1.setScalar(autoenc_diff)
                eq_prop2 = MarabouCore.Equation(MarabouCore.Equation.GE)
                eq_prop2.addAddend(1, inputVars_verified[i])
                eq_prop2.addAddend(-1, outputVars_verified[i])
                eq_prop2.setScalar(autoenc_diff)

                disjunction.append([eq_prop1])
                disjunction.append([eq_prop2])

            network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

    def checkRobustness(self, network_verified, epsilon):
        inputVars = network_verified.inputVars[0]
        outputVars = network_verified.outputVars
        print(outputVars)

        data = [
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4115,
                -0.2842,
                -0.0551,
                -0.0551,
                -0.2969,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.3478,
                0.4922,
                1.5741,
                2.2996,
                2.2996,
                1.2050,
                -0.2078,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                0.2886,
                2.2105,
                2.3378,
                1.5741,
                1.6759,
                2.4396,
                0.7341,
                -0.3860,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.3988,
                1.1795,
                2.1469,
                0.4031,
                -0.2969,
                -0.0169,
                2.0451,
                1.9560,
                -0.0296,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.2969,
                1.5359,
                1.3705,
                -0.3351,
                -0.3478,
                0.6195,
                2.4015,
                2.6306,
                0.7722,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.3606,
                1.2941,
                2.0323,
                0.5559,
                1.0777,
                2.2996,
                2.7578,
                2.5160,
                0.5177,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                0.2377,
                1.7014,
                2.1978,
                2.2869,
                1.7269,
                2.0451,
                1.8414,
                -0.1569,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.3988,
                -0.1824,
                0.2631,
                0.1486,
                -0.1951,
                1.1923,
                1.0904,
                -0.3860,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.2460,
                1.5868,
                0.8359,
                -0.4115,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.1569,
                1.9687,
                1.3705,
                -0.3351,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.2333,
                1.5232,
                1.2305,
                -0.3351,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
            [
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.4242,
                -0.3988,
                -0.1442,
                -0.1696,
                -0.4115,
                -0.4242,
                -0.4242,
                -0.4242,
            ],
        ]

        res = [i for arr in data for i in arr]

        for i in range(len(inputVars)):
            network_verified.setLowerBound(inputVars[i], res[i] - 0.1)
            network_verified.setUpperBound(inputVars[i], res[i] + 0.1)

        eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq1.addAddend(1, outputVars[7])
        eq1.addAddend(-1, outputVars[9])
        eq1.setScalar(0)

        disjunction = [[eq1]]
        network_verified.addDisjunctionConstraint(disjunction)

        option = Marabou.createOptions(verbosity=1)
        vals, stats = network_verified.solve(options=option)

    def compute_adv_example(
        self, networkFile, data, original_label, target_adv, epsilon, resize
    ):
        network_trained = Marabou.read_onnx(networkFile)
        inputVars = network_trained.inputVars[0]
        outputVars = network_trained.outputVars

        data = data.tolist()

        for i in range(len(inputVars)):
            network_trained.setLowerBound(inputVars[i], data[i] - epsilon)
            network_trained.setUpperBound(inputVars[i], data[i] + epsilon)

        eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq1.addAddend(1, outputVars[target_adv])
        eq1.addAddend(-1, outputVars[original_label])
        eq1.setScalar(0)

        disjunction = [[eq1]]
        network_trained.addDisjunctionConstraint(disjunction)

        option = Marabou.createOptions(verbosity=0)
        vals, stats = network_trained.solve(options=option, verbose=False)

        if not vals:
            return torch.FloatTensor([])
        else:
            res_list = []
            i = 0
            for key, value in vals.items():
                res_list.append(value)
                i += 1
                if i >= resize:
                    break
            return torch.FloatTensor(res_list)

    def compute_most_adv_example(
        self, networkFile, data, original_label, target_adv, epsilon, resize
    ):
        timeout = time.time() + 0.5  # 1 minutes from now
        first_compute = True  # flag
        res_val = {}
        data = data.tolist()  #

        # bound for binary search in order to find the most adv example
        bound = 0
        last_bound = 0
        upper_bound = 1

        if target_adv == original_label:
            return torch.FloatTensor([])
        while True:
            network_trained = Marabou.read_onnx(networkFile)
            inputVars = network_trained.inputVars[0]
            outputVars = network_trained.outputVars

            for i in range(len(inputVars)):
                network_trained.setLowerBound(inputVars[i], data[i] - epsilon)
                network_trained.setUpperBound(inputVars[i], data[i] + epsilon)

            eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq1.addAddend(1, outputVars[target_adv])
            eq1.addAddend(-1, outputVars[original_label])
            eq1.setScalar(bound)

            disjunction = [[eq1]]
            network_trained.addDisjunctionConstraint(disjunction)

            option = Marabou.createOptions(verbosity=0)
            vals, stats = network_trained.solve(options=option, verbose=False)

            # record the last "most adv" example and binary search the bound
            if vals:
                res_val = vals
                last_bound = bound
                bound = (bound + upper_bound) / 2
            elif not vals:
                tmp_bound = last_bound
                last_bound = bound
                bound = (tmp_bound + bound) / 2

            if not vals and first_compute:  # If no adv example in the first computation
                return torch.FloatTensor([])
            elif time.time() > timeout:  # timeout, return the last adv example
                res_list = []
                i = 0
                for key, value in res_val.items():
                    res_list.append(value)
                    i += 1
                    if i >= resize:
                        break
                return torch.FloatTensor(res_list)

            first_compute = False

    def adv_acc(
        self,
        networkFile_noadvtraining,
        networkFile_advtraining,
        networkFile_mosttraining,
    ):

        epsilon = 0.01
        network_noadv = Marabou.read_onnx(networkFile_noadvtraining)
        network_adv = Marabou.read_onnx(networkFile_advtraining)
        network_mostadv = Marabou.read_onnx(networkFile_mosttraining)
        inputVars_noadv = network_noadv.inputVars[0]
        outputVars_noadv = network_noadv.outputVars
        inputVars_adv = network_adv.inputVars[0]
        outputVars_adv = network_adv.outputVars
        inputVars_mostadv = network_mostadv.inputVars[0]
        outputVars_mostadv = network_mostadv.outputVars

        test_kwargs = {"batch_size": 1}
        transform = transforms.Compose(
            [
                transforms.Resize(14),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)

        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        for data, target in test_loader:

            k = 0
            for i in range(14):
                for j in range(14):
                    lower_bound = data[0][0][i][j] - epsilon
                    upper_bound = data[0][0][i][j] + epsilon
                    network_noadv.setLowerBound(inputVars_noadv[k], lower_bound)
                    network_noadv.setUpperBound(inputVars_noadv[k], upper_bound)
                    network_adv.setLowerBound(inputVars_adv[k], lower_bound)
                    network_adv.setUpperBound(inputVars_adv[k], upper_bound)
                    network_mostadv.setLowerBound(inputVars_mostadv[k], lower_bound)
                    network_mostadv.setUpperBound(inputVars_mostadv[k], upper_bound)
                    k += 1

            # no adv training network
            begin_noadv_solving_time = time.time()
            eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq1.addAddend(1, outputVars_noadv[7])
            eq1.addAddend(-1, outputVars_noadv[target.item()])
            eq1.setScalar(0)

            disjunction = [[eq1]]
            network_noadv.addDisjunctionConstraint(disjunction)

            option = Marabou.createOptions(verbosity=0)
            vals_noadv, stats_noadv = network_noadv.solve(options=option, verbose=True)
            noadv_solving_time = begin_noadv_solving_time - time.time()

            # adv training network
            begin_adv_solving_time = time.time()
            eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq1.addAddend(1, outputVars_adv[7])
            eq1.addAddend(-1, outputVars_adv[target.item()])
            eq1.setScalar(0)

            disjunction = [[eq1]]
            network_adv.addDisjunctionConstraint(disjunction)

            option = Marabou.createOptions(verbosity=0)
            vals_adv, stats_adv = network_adv.solve(options=option, verbose=True)
            adv_solving_time = begin_adv_solving_time - time.time()

            # most adv training network
            begin_mostadv_solving_time = time.time()
            eq1 = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq1.addAddend(1, outputVars_mostadv[7])
            eq1.addAddend(-1, outputVars_mostadv[target.item()])
            eq1.setScalar(0)

            disjunction = [[eq1]]
            network_mostadv.addDisjunctionConstraint(disjunction)

            option = Marabou.createOptions(verbosity=0)
            vals_mostadv, stats_mostadv = network_mostadv.solve(
                options=option, verbose=True
            )
            mostadv_solving_time = begin_mostadv_solving_time - time.time()

            print("noadv solving time" + str(noadv_solving_time))
            print("adv solving time" + str(adv_solving_time))
            print("mostadv solving time" + str(mostadv_solving_time))

            # if vals_adv:
            # print("adv in both!")
            # break
