import numpy as np
import helper
import pandas as pd
import os

np.random.seed(1784)


num_iters = 100
time_horizon = 1000





# Defining parameters for the set of experiments we are running:
M_list = np.linspace(10, 100, 20)
h = 1.0
b_list = [1.]
# delta_list = np.linspace(0.1, 0.5, 20)

delta_list = np.linspace(0, 0.5, 20)

# delta_list = np.linspace(0, 0.1, 6)


distr_list = [helper.multi_params]


algo_list = {'static' : helper.static_multi_allocation}
# algo_list = {}

algo_list.update({f'bang_bang_{delta}': 
                  (lambda delta: (lambda t, state, budget_t, demand_t, M, distr_params: 
                                  helper.bang_bang_multi_allocation(t, state, budget_t, demand_t, M, distr_params, delta)))
                  (delta) for delta in delta_list})



for distr_params in distr_list:
    data = []
    print(f"Evaluating for: {distr_params['budget_name'], distr_params['demand_name']}")
    for M in M_list:
        print(f'Evaluating for M: {M}')
        for b in b_list:
            print(f'Evaluating for b: {b}')
            # Set up the experiments and run them, append the data
            for alg_name, algorithm in algo_list.items():
                print(f'Algorithm: {alg_name}')
                new_data = helper.evaluate_multi_algorithm(b, h, M, alg_name, algorithm, distr_params, time_horizon, num_iters)
                data.extend(new_data)


    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    file_name = f'./data/data_{distr_params["budget_name"]}_{distr_params["demand_name"]}.csv'
    print(f'Appending data to: {file_name}')
    
    # Check if file exists to determine whether to write header
    file_exists = os.path.isfile(file_name)

    # Append the DataFrame to the existing CSV file (without writing the header if the file exists)
    df.to_csv(file_name, mode='a', header=not file_exists, index=False)

    print(df.head(5))