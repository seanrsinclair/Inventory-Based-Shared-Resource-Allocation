import numpy as np
import re


DEBUG = False


# Defining parameters for the distribution we are running on
normal_params = {
    'budget_name': 'normal',
    'mu_b': 5,
    'sigma_b': 1,
    'demand_name': 'normal',
    'mu_n': 5,
    'sigma_n': 1
}

exp_poisson_params = {
    'budget_name': 'exponential',
    'mu_b': 5,
    'sigma_b': 1,
    'demand_name': 'poisson',
    'mu_n': 5,
    'sigma_n': 1
}

time_varying_normal_params = {
    'budget_name' : 'cyclic_normal',
    'mu_b': 5,
    'sigma_b': 1,
    'demand_name': 'cyclic_normal',
    'mu_n': 5,
    'sigma_n': 1

}

multi_params = {
    'num_types' : 3,
    'num_resources' : 5,
    'budget_name' : 'normal_multi',
    'mu_b' : [5,5,5,5,5],
    'sigma_b' : 1,
    'demand_name' : 'normal_multi',
    'mu_n' : [1.25, 1.5, 2.25],
    'sigma_n' : 1,
    'weights': np.array([
    [3.9, 3.0, 2.8, 2.7, 1.9],  # omnivore
    [3.9, 3.0, 0.1, 2.7, 0.1],  # vegetarian
    [3.9, 3.0, 2.8, 2.7, 0.1],  # prepared-only
], dtype=float)

}

# Adding dictionary to dataframe
def add_to_data(data, algo_name, metric, cost, **kwargs):
    """Add result dictionary to the data list."""
    data.append({
        'algorithm': algo_name,
        'metric': metric,
        'value': cost,
        **kwargs
    })


# Gets delta number from string of algorithm name
def extract_number(input_string, M):
    match = re.search(r'bang_bang_([-+]?[0-9]*\.?[0-9]+)', input_string)
    if match:
        return float(match.group(1))
    else:
        if input_string == 'static':
            return 0
        else:
            return np.inf


# Samples the distributions according to the type
def sample_budget(distr_params, t):
    if distr_params['budget_name'] == 'normal':
        return max(0, np.random.normal(distr_params['mu_b'], distr_params['sigma_b']))

    elif distr_params['budget_name'] == 'exponential':
        return max(0,np.random.exponential(distr_params['mu_n']))

    elif distr_params['budget_name'] == 'cyclic_normal':
        sequence = [distr_params['mu_b']-2, distr_params['mu_b']-1, distr_params['mu_b'], distr_params['mu_b']+1, distr_params['mu_b']+2]
        return max(0, np.random.normal(sequence[t % len(sequence)], distr_params['sigma_b']))
    
    elif distr_params['budget_name'] == 'normal_multi':
        return np.maximum(0, np.random.normal(distr_params['mu_b'], distr_params['sigma_b']))


# Samples the distributions according to the type
def sample_demand(distr_params, t):
    if distr_params['demand_name'] == 'normal':
        return max(0, np.random.normal(distr_params['mu_n'], distr_params['sigma_n']))
    
    elif distr_params['demand_name'] == 'poisson':
        return max(0,np.random.poisson(distr_params['mu_b']))

    elif distr_params['demand_name'] == 'cyclic_normal':
        sequence = [distr_params['mu_n']-2, distr_params['mu_n']-1, distr_params['mu_n'], distr_params['mu_n']+1, distr_params['mu_n']+2]
        return max(0, np.random.normal(sequence[t % len(sequence)], distr_params['sigma_n']))

    elif distr_params['demand_name'] == 'normal_multi':
        return np.maximum(0, np.random.normal(distr_params['mu_n'], distr_params['sigma_n']))



# Static allocation algorithm
def static_allocation(t, state, budget_t, demand_t, M, distr_params):
    return distr_params['mu_b'] / distr_params['mu_n']




# Static allocation algorithm
def static_multi_allocation(t, state, budget_t, demand_t, M, distr_params):
    return np.asarray([np.asarray(distr_params['mu_b']) / np.sum(np.asarray(distr_params['mu_n'])) for _ in range(len(distr_params['mu_n']))])



# Bang bang allocation algorithm
def bang_bang_allocation(t, state, budget_t, demand_t, M, distr_params, delta):
    if state > (M / 2):
        return min(M, (distr_params['mu_b'] / distr_params['mu_n']) + (delta / 2))
    elif state < (M / 2):
        return max(0, (distr_params['mu_b'] / distr_params['mu_n']) - (delta / 2))
    else:
        return distr_params['mu_b'] / distr_params['mu_n']

def bang_bang_multi_allocation(t, state, budget_t, demand_t, M, distr_params, delta):
    num_resource = len(state)
    num_types = len(distr_params['mu_n'])

    alloc = np.zeros((num_types, num_resource))  # allocation is \Theta x K
    for k in range(num_resource): # for each resource:
        if state[k] > (M / (2*num_resource)):
            alloc[:, k] = np.asarray([distr_params['mu_b'][k] / np.sum(distr_params['mu_n']) + (delta / 2) for typ in range(num_types)])
        elif state[k] < (M / (2*num_resource)):
            alloc[:, k] = np.asarray([distr_params['mu_b'][k] / np.sum(distr_params['mu_n']) - (delta / 2) for typ in range(num_types)])
        else:
            alloc[:, k] = np.asarray([distr_params['mu_b'][k] / np.sum(distr_params['mu_n']) for typ in range(num_types)])
    return alloc



# Adaptive allocation algorithm
def adaptive_allocation(t, state, budget_t, demand_t, M, distr_params):
    if demand_t == 0:
        return 0
    else:
        return (state + budget_t) / demand_t


# Adaptive allocation algorithm
def adaptive_multi_allocation(t, state, budget_t, demand_t, M, distr_params):
    if np.sum(demand_t) == 0:
        return np.asarray([[0 for typ in range(distr_params['num_resources'])] for _ in range(distr_params['num_types'])])
    else:
        return np.asarray([(state + budget_t) / np.sum(demand_t) for _ in range(distr_params['num_types'])])


# Runs a simulation to evaluate algorithms performance, saving the data

def evaluate_multi_algorithm(b, h, M, alg_name, algorithm, distr_params, time_horizon = 10000, num_iters = 100):

    data_list = []
    num_resources = len(distr_params['mu_b'])
    num_types = len(distr_params['mu_n'])

    common_params = {'algorithm' : alg_name,
                     'delta' : extract_number(alg_name, M),
                        'b': b, 
                        'h': h, 
                        'M': M,
                        }
    common_params.update(distr_params)

    for _ in range(num_iters):
        state = np.asarray([M / (2*num_resources) for _ in range(num_resources)])
        overage = 0
        underage = 0

        low_alloc = np.asarray([[M for _ in range(num_resources)] for _ in range(num_types)])
        high_alloc = np.asarray([[0 for _ in range(num_resources)] for _ in range(num_types)])

        for t in range(time_horizon): # Loops over the time horizon
            if DEBUG: print(f'Current state: {state}')
            budget_t = sample_budget(distr_params, t) # samples donations and demand
            demand_t = sample_demand(distr_params, t)
            if DEBUG: print(f'Budget arrival: {budget_t}, Demand arrival: {demand_t}')
            allocation = algorithm(t, state, budget_t, demand_t, M, distr_params) # Sets allocation

            low_alloc = np.minimum(allocation, low_alloc) # Keeps track of smallest and largest allocation for fairness
            high_alloc = np.maximum(allocation, high_alloc)




            w = np.maximum(np.sum(state) + np.sum(budget_t) - (demand_t[:, None] * allocation).sum() - M, 0) # underage and overage costs
            v = np.abs(np.minimum(np.sum(state) + np.sum(budget_t) - (demand_t[:, None] * allocation).sum(), 0))
            overage += w
            underage += v
            state = np.asarray([np.clip(state[k] + budget_t[k] - (demand_t[:, None] * allocation)[:,k].sum(),0,(M / num_resources)) for k in range(num_resources)])

        diff = high_alloc - low_alloc                       # (T, R)
        vals = (distr_params['weights'] * diff).sum(axis=1)                       # (T,)  w[type] · (high - low)
        theta_star = np.argmax(vals)                        # arg max over types
        fair = vals[theta_star]


        add_to_data(data_list, alg_name, 'overage', (1/time_horizon)*overage, **common_params)
        add_to_data(data_list, alg_name, 'underage', (1/time_horizon)*underage, **common_params)
        add_to_data(data_list, alg_name, 'efficiency', (1/time_horizon)*(b*overage+h*underage), **common_params)
        add_to_data(data_list, alg_name, 'fair', fair, **common_params)
    return data_list



def evaluate_algorithm(b, h, M, alg_name, algorithm, distr_params, time_horizon = 10000, num_iters = 100):

    data_list = []

    common_params = {'algorithm' : alg_name,
                     'delta' : extract_number(alg_name, M),
                        'b': b, 
                        'h': h, 
                        'M': M,
                        }
    common_params.update(distr_params)

    for _ in range(num_iters):
        state = M / 2
        overage = 0
        underage = 0

        low_alloc = M
        high_alloc = 0

        for t in range(time_horizon): # Loops over the time horizon
            if DEBUG: print(f'Current state: {state}')
            budget_t = sample_budget(distr_params, t) # samples donations and demand
            demand_t = sample_demand(distr_params, t)
            if DEBUG: print(f'Budget arrival: {budget_t}, Demand arrival: {demand_t}')
            allocation = algorithm(t, state, budget_t, demand_t, M, distr_params) # Sets allocation

            low_alloc = min(allocation, low_alloc) # Keeps track of smallest and largest allocation for fairness
            high_alloc = max(allocation, high_alloc)

            w = np.maximum(state + budget_t - demand_t * allocation - M, 0) # underave and overage costs
            v = np.abs(np.minimum(state + budget_t - demand_t*allocation, 0))
            overage += w
            underage += v
            state = np.clip(state + budget_t - demand_t * allocation, 0, M) # new inventory state

        fair = high_alloc - low_alloc # saves results
        add_to_data(data_list, alg_name, 'overage', (1/time_horizon)*overage, **common_params)
        add_to_data(data_list, alg_name, 'underage', (1/time_horizon)*underage, **common_params)
        add_to_data(data_list, alg_name, 'efficiency', (1/time_horizon)*(b*overage+h*underage), **common_params)
        add_to_data(data_list, alg_name, 'fair', fair, **common_params)
    return data_list

