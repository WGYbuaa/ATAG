from json import JSONEncoder



class NodeEncoder(JSONEncoder):
    def default(self, o):
        try:
            return o.__dict__
        except:
            return "convert to json error"


# Each test case is stored separately.
class TestCase:
    def __init__(self):
        self.index_global = 0  # The index of this test case in all data.
        self.ts_list = []  # All test steps included in this test case.
        self.ts_list_str = ''  # Connect all step_clean in the test case, used for common sequence determination.
        self.ts_former = 0  # The total number of initial testing steps.
        self.coupling = []  # Record Ave CBF.
        self.coupling_original = 0  # Calculate the initial CBF.
        self.coh = []  # Record sum Coh.
        self.coh_original = 0  # initial test functions sum Coh.

        self.dataflow = {}  # The dataflow and its corresponding position in this tc.
        self.commonSequence = []  # record common sequence in the test case.
        self.occ_time = []  # Record the number of occurrences of each common sequence.

    def add_ts(self, test_step):
        self.ts_list.append(test_step)

    def __repr__(self):
        s = str(self.index_global)
        return s

    def __str__(self):
        return self.__repr__()


# Each test step is stored separately.
class TestStep:
    def __init__(self, index, instruction, parameters, returns, tc_number, ts_local_index):
        self.index_global = index  # The index of test step in the entire file.
        self.index_local = ts_local_index  # The index of the test step in test case.
        self.step = instruction  # Test step's instruction.
        self.parameters = parameters  # Parameter list
        self.returns = [returns]  # Return.
        self.action = ""  # test step's test action.
        self.object = list()  # data object
        self.follow_ts = []  # When generating a function, record all test steps.
        self.embedding = []  # test step's embedding
        self.step_clean = ""  # Remove special symbols, numbers, etc.

    # When generating a test function, the first test step follows the remaining test steps.
    def follow_ts(self, test_step):
        self.follow_ts.append(test_step)

    def __repr__(self):
        s = str(self.index_global) + '-' + self.step
        return s

    def __str__(self):
        return self.__repr__()


# Record generated test function.
class TestFunction:
    def __init__(self):
        self.ts_list = []  # The test steps realized in this test function.
        self.index = []  # Covered init test functions.
        self.object_list = {}  # Visited objects and the number of visits.
        self.number_occ_in_tc = 1  # The number of occurrences.
        self.step_clean = ''  # Record step_clean
        self.generated_based_on = []  # Record the strategy based on which the test function was generated.

    def add_ts(self, test_step):
        self.ts_list.append(test_step)

    def __repr__(self):
        s = str(self.index)
        return s

    def __str__(self):
        return self.__repr__()
