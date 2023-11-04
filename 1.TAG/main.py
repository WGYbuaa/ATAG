from TAG import *

if __name__ == '__main__':
    print("******************* Start Data Reading *********************")
    # Read test case data from the json file and store it test case by test case.
    test_cases_list, all_ts_clean = generate_test_step(path='data.json')

    print("******************* Find Common Sequence. *********************")
    test_cases_list = generate_common_sequences(test_cases_list, all_ts_clean)

    print("******************* Start Test Architecture Generation. *********************")
    test_architecture_generation(test_cases_list)

    print("******************* Output Test Architectures. *********************")
    output_result(test_architecture_list)

    print("******************* Finish. *********************")
