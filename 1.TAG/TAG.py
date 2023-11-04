from class_list import *
from sentence_transformers import SentenceTransformer
import spacy
import json

MODEL = SentenceTransformer('../../data/distiluse-base-multilingual-cased')
test_model_en = spacy.load('en_core_web_sm')
frequency_threshold = 1
test_architecture_list = list()  # Test Architectures list.
tau_cbf = 0.01
tau_coh = 0.001


# Read the data and store it test case by test case.
def generate_test_step(path):
    test_cases_list = list()  # Store all test cases in order.
    test_case = TestCase()
    tc_number = 0  # The global index of test case.
    ts_local_index = 0  # Test step's local index in the test case.

    with open(path, encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:  # Determines whether a test case is finished.
                if len(test_case.ts_list) > 0:  # If a test case is over, starts the following.
                    test_case.ts_former = len(test_case.ts_list)
                    test_case.coupling_original = get_coupling_original(test_case)
                    test_case.coupling.append(get_coupling(test_case))
                    test_case.coh_original = coh_initial_functions(test_case)
                    test_case.coh.append(coh_initial_functions(test_case))
                    test_cases_list.append(test_case)  # The set of test cases, storing TCs in order.
                    test_case = TestCase()
                    tc_number = tc_number + 1  # The index of the next test case.
                    ts_local_index = 0
                continue
            json_obj = json.loads(line)  # Read the test steps by line.

            # Stored in a new test step, including the following:
            test_step = TestStep(json_obj['index_global'], json_obj['step'],
                                 json_obj['parameter'], json_obj['returns'], tc_number, ts_local_index)
            ts_local_index = ts_local_index + 1  # The next test step's index.
            action_object = extract_action_object_en(test_step.step)
            test_step.action = action_object[0]
            test_step.object.append(action_object[1])

            if len(test_step.object[0]) == 0 and len(test_step.action) != 0:
                # Many ROOTs in the dictionary model are actually obj's, but are placed in the action.
                test_step.object[0] = test_step.action

            if len(test_step.action) == 0 or len(test_step.object[0]) == 0:
                print(test_step.action, test_step.object, test_step.index_global)

            # Delete the unnecessary definite complement.
            test_step.step_clean = test_step.action + str(test_step.object[0])

            # Store embedding.
            test_step.embedding = MODEL.encode([test_step.step_clean])

            # Store dataflow.
            for object1 in test_step.object:
                if object1 in test_case.dataflow.keys():
                    # dataflow {object:position}
                    test_case.dataflow[object1].append(test_step.index_local)
                else:
                    test_case.dataflow.setdefault(object1, [test_step.index_local])
                test_case.add_ts(test_step)
                test_case.index_global = tc_number

    all_ts_clean = save_test_cases(test_cases_list)
    return test_cases_list, all_ts_clean


# CBF = NCD + NDD
# NCD: # of control dependencies among functions.
# NDD: # of data dependencies between test functions manifested through shared access to external data objects.
def get_coupling(test_case):
    # NDD
    DD = 0
    for object1 in test_case.dataflow.keys():
        if len(test_case.dataflow[object1]) > 1:
            DD = DD + len(test_case.dataflow[object1])

    # NCD
    if len(test_case.coupling) < 1:
        coupling = test_case.coupling_original / (len(test_case.ts_list) + 1)  # .coupling_original is the initial CBF.
    elif len(test_case.coupling) > 0:
        CD = test_case.ts_former + len(test_case.coupling)  # Generate a test function for each round.
        dependency = CD + DD
        tf_number = test_case.ts_former + len(test_case.coupling) + 1
        coupling = dependency / tf_number

    return coupling


# Calculate the coupling of the initial state in a test case.
def get_coupling_original(test_case):
    CD = 0
    DD = 0

    # Calculate CD
    CD = len(test_case.ts_list)

    # CalculateDD
    for object1 in test_case.dataflow.keys():
        if len(test_case.dataflow[object1]) > 1:
            DD = DD + len(test_case.dataflow[object1])

    coupling = CD + DD

    return coupling


# Calculate the sum coh of the initial test functions.
def coh_initial_functions(test_case):
    coh_sum = 0
    for ts in test_case.ts_list:
        obj_number = len(ts.object)
        if len(ts.parameters) > 1:
            obj_number = obj_number + (len(ts.parameters) - 1)
        act = 1  # One initial test function realizes one test action.
        conn = obj_number
        coh = conn / (act * (obj_number + 1))
        coh_sum += coh
    return coh_sum


# Calculate the Coh of each test function.
def get_coh_every_tp(test_architecture_list):
    coh_every_tp = []  # Record the Coh for each test function in order.
    for tp in test_architecture_list:
        # v(ta, o) counts whether a test action 'ta' visits a data object
        # 'o' in the test function with 1 indicating yes, otherwise 0.
        v = 0

        # ndo is the number of data objects visited by the test actions
        ndo = len(tp.object_list)

        for obj in tp.object_list.keys():
            v += tp.object_list[obj]

        # nta is the number of test actions in a test function
        nta = len(tp.ts_list)

        coh = v / (nta * (ndo + 1))
        coh_every_tp.append(coh)

    return coh_every_tp


# Extract the test action and the visited object.
def extract_action_object_en(testStep):
    doc = test_model_en(testStep)
    for token in doc:
        if token.dep_ == 'ROOT':
            action_word = str(token)
            for token1 in doc:
                if token1.dep_ == 'dobj':
                    object_word = str(token1)
    action_object = [action_word, object_word]
    return action_object


# Save all test steps' step_clean.
def save_test_cases(test_cases_list):
    all_ts_clean = ''
    # Calculation of number of occurrences
    for test_case in test_cases_list:
        for i in range(len(test_case.ts_list)):
            test_case.ts_list_str = test_case.ts_list_str + test_case.ts_list[i].step_clean
            all_ts_clean = all_ts_clean + test_case.ts_list[i].step_clean
    return all_ts_clean


# Find common sequences.
def generate_common_sequences(test_cases_list, all_ts_clean):
    for test_case in test_cases_list:
        x = -1
        cf_index = []
        for i in range(len(test_case.ts_list)):  # Calculate the number of occurrences of each tp
            if i <= x:
                continue
            count = all_ts_clean.count(test_case.ts_list[i].step_clean)
            if count > frequency_threshold:  # Recorded as a common sequence.
                cf_index.append(i)
                cf_temp = test_case.ts_list[i].step_clean
                x, cf_index, follow_min_occ = return_cf(cf_temp, test_case, all_ts_clean, i, cf_index)
                test_case.commonSequence.append(cf_index)
                test_case.occ_time.append(min(count, follow_min_occ))
                cf_index = []  # Reset
            else:
                if len(cf_index) > 0:
                    test_case.commonSequence.append(cf_index)
                    test_case.occ_time.append(count)
                    cf_index = []  # Reset
                else:
                    test_case.commonSequence.append([i])
                    test_case.occ_time.append(count)
    return test_cases_list


# Returns the longest common sequences.
def return_cf(cf_temp, test_case, all_ts_clean, i, cf_index):
    follow_min_count = 99999
    if i + 1 >= len(test_case.ts_list):
        return i, cf_index, follow_min_count
    cf_temp = cf_temp + test_case.ts_list[i + 1].step_clean
    if all_ts_clean.count(cf_temp) > frequency_threshold:
        follow_min_count = all_ts_clean.count(cf_temp)
        i = i + 1
        cf_index.append(i)
        i, cf_index, follow_count = return_cf(cf_temp, test_case, all_ts_clean, i, cf_index)
        follow_min_count = min(follow_min_count, follow_count)
        return i, cf_index, follow_min_count
    else:
        return i, cf_index, follow_min_count


# Test Architecture Generation.
def test_architecture_generation(test_cases_list):
    for test_case in test_cases_list:
        generated_functions = []
        test_pattern_planning_controlFlow_and_dataflow(test_case, generated_functions)


# Select the strategy.
# Execute test_pattern_planning_controlFlow() is the ATAGcf strategy;
# Execute test_pattern_planning_dataflow() is the ATAGdf strategy;
# Execute test_pattern_planning_controlFlow() and test_pattern_planning_dataflow() is ATAGcf+df.
def test_pattern_planning_controlFlow_and_dataflow(test_case, generated_functions):
    test_pattern_planning_controlFlow(test_case)
    test_pattern_planning_dataflow(test_case, generated_functions)


# Generate test functions based on ATAGcf.
def test_pattern_planning_controlFlow(test_case):
    # Determine if there are common sequences.
    control_flow = [sub_list for sub_list in test_case.commonSequence if len(sub_list) > 1]
    if len(control_flow) == 0:
        return test_architecture_list

    # Aggregate all the TEST STEPS in the common sequence.
    generated_functions = []
    for cf in control_flow:
        test_pattern = TestFunction()
        test_pattern.generated_based_on.append('cf')
        for j in range(cf[0], cf[-1] + 1):
            test_pattern.index.append(test_case.ts_list[j].index_global)
            test_pattern.ts_list.append(test_case.ts_list[j])
            for obj in test_case.ts_list[j].object:
                test_pattern.object_list[obj] = 0

                # Count the number of visits of obj.
                for z in range(len(test_pattern.ts_list)):
                    if obj in test_pattern.ts_list[z].object:
                        test_pattern.object_list[obj] = test_pattern.object_list[obj] + 1

            # Count the number of occurrences of the test function.
            test_pattern.number_occ_in_tc = test_case.occ_time[test_case.commonSequence.index(cf)]
        for ts in test_pattern.ts_list:
            test_pattern.step_clean += ts.step_clean
        test_architecture_list.append(test_pattern)
        generated_functions.append(test_pattern)

    for cf1 in control_flow:
        for k in range(cf1[0] + 1, cf1[-1] + 1):
            test_case.ts_list[cf1[0]].follow_ts.append(test_case.ts_list[k])
            test_case.ts_list[cf1[0]].step = test_case.ts_list[cf1[0]].step + "," + test_case.ts_list[k].step
            test_case.ts_list[cf1[0]].parameters += test_case.ts_list[k].parameters
            test_case.ts_list[cf1[0]].returns += test_case.ts_list[k].returns
            test_case.ts_list[cf1[0]].step_clean = test_case.ts_list[cf1[0]].step_clean + "," + test_case.ts_list[
                k].step_clean

            for obj in test_case.ts_list[k].object:
                if obj not in test_case.ts_list[cf1[0]].object:
                    test_case.ts_list[cf1[0]].object.append(obj)

    for cf2 in reversed(control_flow):
        del test_case.ts_list[cf2[0] + 1:cf2[-1] + 1]
        del cf2[1:]

    test_case.dataflow = {}
    for ts in test_case.ts_list:
        ts.index_local = test_case.ts_list.index(ts)
        for obj1 in ts.object:
            if obj1 in test_case.dataflow.keys():
                test_case.dataflow[obj1].append(ts.index_local)
            else:
                test_case.dataflow[obj1] = [ts.index_local]

    coupling_1 = get_coupling(test_case)
    coh_every_tp = get_coh_every_tp(generated_functions)
    coh_1 = sum(coh_every_tp) + test_case.coh_original

    test_case.coupling.append(coupling_1)
    test_case.coh.append(coh_1)

    if abs(test_case.coupling[-1] - test_case.coupling[-2]) > tau_cbf and abs(
            test_case.coh[-1] - test_case.coh[-2]) > tau_coh:
        test_pattern_planning_controlFlow(test_case)
    else:
        return


# Generate test functions based on ATAGdf.
def test_pattern_planning_dataflow(test_case, generated_functions):
    min_dataflow, first_index, second_index = return_dataflow(test_case.dataflow, test_case.ts_list)
    # Predefined min_dataflow=99 for no dataflow exists;
    # or the last dataflow covers the entire test cases, terminate the generation.
    if min_dataflow == 99 or min_dataflow == len(
            test_case.ts_list) - 1:
        return test_architecture_list

    # Aggregation
    test_pattern = TestFunction()
    test_pattern.generated_based_on.append('df')
    for x in range(first_index, second_index + 1):
        test_pattern.index.append(return_index(test_case.ts_list[x]))
        test_pattern.ts_list.append(test_case.ts_list[x])
        for obj3 in test_case.ts_list[x].object:
            test_pattern.object_list[obj3] = 0
            # Count the number of visits of obj.
            for z in range(len(test_pattern.ts_list)):
                if obj3 in test_pattern.ts_list[z].object:
                    test_pattern.object_list[obj3] = test_pattern.object_list[obj3] + 1

    # New test function covers the test actions from first_index to second_index.
    for i in range(first_index + 1, second_index + 1):
        test_case.ts_list[first_index].follow_ts.append(test_case.ts_list[i])
        test_case.ts_list[first_index].step = test_case.ts_list[first_index].step + "," + test_case.ts_list[i].step
        test_case.ts_list[first_index].parameters.append(test_case.ts_list[i].parameters)
        test_case.ts_list[first_index].returns.append(test_case.ts_list[i].returns)

        for obj in test_case.ts_list[i].object:
            if obj not in test_case.ts_list[first_index].object:
                test_case.ts_list[first_index].object.append(obj)

    del test_case.ts_list[first_index + 1:second_index + 1]

    # Updated dataflow in the test case.
    test_case.dataflow = {}
    for ts in test_case.ts_list:
        ts.index_local = test_case.ts_list.index(ts)  # Updated index.
        for obj1 in ts.object:
            if obj1 in test_case.dataflow.keys():
                test_case.dataflow[obj1].append(ts.index_local)
            else:
                test_case.dataflow[obj1] = [ts.index_local]

    for ts in test_pattern.ts_list:
        test_pattern.step_clean += ts.step_clean
    test_architecture_list.append(test_pattern)
    generated_functions.append(test_pattern)

    # Calculate CBF and Coh.
    coupling_1 = get_coupling(test_case)
    coh_every_tp = get_coh_every_tp(generated_functions)
    coh_1 = sum(coh_every_tp) + test_case.coh_original

    test_case.coupling.append(coupling_1)
    test_case.coh.append(coh_1)

    if any(isinstance(value, list) and len(value) > 1 for value in test_case.dataflow.values()):
        if abs(test_case.coupling[-1] - test_case.coupling[-2]) > tau_cbf and abs(
                test_case.coh[-1] - test_case.coh[-2]) > tau_coh:
            test_pattern_planning_dataflow(test_case, generated_functions)
        else:
            return
    else:
        return


# Return data flows.
def return_dataflow(dataflow, ts_list):
    min_dataflow = 99
    first_index = 0
    second_index = 0
    for object, index in dataflow.items():
        if len(index) > 1:
            first, second, distance = getTwoClosestElements(index)
            if distance < min_dataflow:
                min_dataflow, first_index, second_index = distance, first, second

    return min_dataflow, first_index, second_index


def getTwoClosestElements(index):
    dif = 99
    first, second, distance = 0, 0, 99
    for i, v in enumerate(index[:-1]):
        distance = abs(v - index[i + 1])
        if distance < dif:
            first, second, dif = v, index[i + 1], distance
    return first, second, dif


# Return all indexes.
def return_index(ts):
    return_index_list = [ts.index_global]
    if len(ts.follow_ts) != 0:
        for i in range(len(ts.follow_ts)):
            return_index_list.append(return_index(ts.follow_ts[i]))

    return return_index_list

# Output the result.
def output_result(test_architecture_list):
    for tf in test_architecture_list:
        print(tf.index)
