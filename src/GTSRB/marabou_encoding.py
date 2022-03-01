from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np


class marabouEncoding:
    def __init__(self):
        self.var = {}

    def checkProperties(self, prop, networkFile):
        # Reading DNN using our own version of reading onnx file
        network_verified = Marabou.read_onnx_deepproperty(networkFile)

        if prop[0] == "checking-sign":
            print("-----------checking targeted digit----------")
            return self.checkSign(network_verified, prop[1])
        elif prop[0] == "checking-confidence":
            print("-----------checking confidence ----------")
            return self.checkConfidence(network_verified, prop[1], prop[2])
        elif prop[0] == "checking-equivalence":
            print("-----------checking equivalence of two network----------")
            return self.checkEq(network_verified, prop[1])
        elif prop[0] == "checking-fairness":
            print("-----------checking fairness----------")
            self.checkFair(network_verified, prop[1])

    def checkSign(self, network_verified, number):
        # This DNN outputs two elements, one is the prediction(dim = 10), the other one is true/false
        inputVars_verified = network_verified.inputVars[0]  # resize*resize
        outputVars_verified = network_verified.outputVars  # 2*(43and2)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        # Encoding property network, when property network consider this digit is indeed the one wanted
        eq_prop = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq_prop.addAddend(1, outputVars_verified[1][1])
        eq_prop.addAddend(-1, outputVars_verified[1][0])
        eq_prop.setScalar(0)  # confidence level? IDK, cause no softmax/relu

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
        inputVars_verified = network_verified.inputVars[0]  # resize * resize
        outputVars_verified = network_verified.outputVars  # 2*(43 and resize*resize)

        # Encoding input region
        for i in range(len(inputVars_verified)):
            network_verified.setLowerBound(inputVars_verified[i], -0.4242)
            network_verified.setUpperBound(inputVars_verified[i], 2.8)

        # Encoding property network, l-inf norm
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

        # Encoding NUV, assume the input image should be classified as a specific sign, e.g., eight
        for j in range(len(outputVars_verified[0])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][14])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        # Encoding network under verified, mean difference between max_val and all the others
        disjunction = []
        eq_verified = MarabouCore.Equation(MarabouCore.Equation.LE)
        for j in range(len(outputVars_verified[0])):
            eq_verified.addAddend(1, outputVars_verified[0][14])
            eq_verified.addAddend(-1, outputVars_verified[0][j])
        eq_verified.setScalar(delta * 42)
        disjunction.append([eq_verified])
        network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

        if vals:
            return "sat"
        else:
            return "unsat"

    def checkEq(self, network_verified, epsilon):
        inputVars_verified = network_verified.inputVars[0]  # resize * resize
        outputVars_verified = network_verified.outputVars  # 2*(43and43)

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

    def checkFair(self, network_verified, epsilon):
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

        # Encoding NUV, assume the input image should be classified as stop sign (idx: 14)
        for i in range(len(outputVars_verified[0])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[0][14])
            eq_verified.addAddend(-1, outputVars_verified[0][i])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        # Encoding the output of property network, the counter example is classified as other sign
        for i in range(len(outputVars_verified[1])):
            disjunction = []
            eq_verified = MarabouCore.Equation(MarabouCore.Equation.GE)
            eq_verified.addAddend(1, outputVars_verified[1][7])
            eq_verified.addAddend(-1, outputVars_verified[1][i])
            eq_verified.setScalar(0)
            disjunction.append([eq_verified])
            network_verified.addDisjunctionConstraint(disjunction)

        vals, stats = network_verified.solve()

    def compute_adv_example(self, networkFile, data, label, target_adv):
        network_verified = Marabou.read_onnx(networkFile)
        print(data)
        print(label)
        print(target_adv)
