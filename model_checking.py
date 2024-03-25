'''
{state1 : {'noact' : [['state1', 'state2', ...], [prob1, prob2, ...]], ...}}
{state1 : {action1: [['state1', 'state2', ...], [prob1, prob2, ...]], action2: ...}}
'''

import numpy as np

def is_reachable_mc(mc:dict, s:str, t:str):

    '''
    Given a Markov chain mc and two states s and t,
    return True if t is reachable from s, and False otherwise.
    '''

    if s == t:
        return True
    elif t in mc[s]['noact'][0]:
        return True
    else:
        for s_ in mc[s]['noact'][0]:
            if s_ != s:
                if is_reachable_mc(mc, s_, t):
                    return True
    return False

def is_reachable_mdp(mdp:dict, s:str, t:str):

    '''
    Given a Markv decision process mdp and two states s and t,
    return True if t is reachable from s, and False otherwise.
    '''

    neighbours = list(np.flatten([mdp[s][act][0] for act in mdp[s].keys()]))
    if s == t:
        return True
    elif t in neighbours:
        return True
    else:
        for s_ in neighbours:
            if is_reachable_mdp(mdp, s_, t):
                return True
    return False

def prob_states_mc(states:list, mc:dict):

    '''
    Given a Markov chain mc and a list of states,
    return the probability of reaching the set of states for each state in the markov chain.
    '''

    # Normalize probabilities in the markov chain
    for s in mc.keys():
        mc[s]['noact'][1] = list(mc[s]['noact'][1]/(np.sum(mc[s]['noact'][1])))

    # Create a list of states that can reach the given set of states
    S_tilde = []
    for s in mc.keys():
        if s not in states:
            for t in states:
                if is_reachable_mc(mc, s, t):
                    S_tilde.append(s)
                    break
    
    # Create the transition probabilities matrix for states in S_tilde
    A = np.zeros((len(S_tilde), len(S_tilde)))
    for s in S_tilde:
        for t in S_tilde:
            if t in mc[s]['noact'][0]:
                A[S_tilde.index(s), S_tilde.index(t)] = mc[s]['noact'][1][mc[s]['noact'][0].index(t)]
            else:
                A[S_tilde.index(s), S_tilde.index(t)] = 0

    # Create the vector b containing the probabilities of reaching the set of states within one step for each state in S_tilde
    b = np.zeros(len(S_tilde))
    for s in S_tilde:
        p=0
        for u in states:
            if u in mc[s]['noact'][0]:
                p+=mc[s]['noact'][1][mc[s]['noact'][0].index(u)]
        b[S_tilde.index(s)] = p

    # Solve the linear system of equations to find the probability of reaching the set of states for each state in S_tilde
    # x = Ax + b
    x = np.linalg.solve(np.eye(len(S_tilde))-A, b)

    # Create a dictionary to store the probabilities of reaching the set of states for each state in the markov chain
    prob_states = {}
    for s in mc.keys(): 
        if s in states:
            prob_states[s] = 1
        elif s in S_tilde:
            prob_states[s] = x[S_tilde.index(s)]
        else:
            prob_states[s] = 0
    
    return prob_states

def simulate_markov(printer, scheduler=None, start=None, end=None, iter_max=100):
    # only for markov chain
    printer.current_state = start
    printer.current_actions = list(printer.model[printer.current_state].keys())
    end_simu = printer.current_state in end
    iter = 0
    while not end_simu and iter < iter_max:
        iter += 1
        if scheduler is None:
            action = printer.current_actions[0]
        else:
            action = None
            max = 0
            for a in scheduler[printer.current_state].keys():
                if scheduler[printer.current_state][a] > max:
                    max = scheduler[printer.current_state][a]
                    action = a
        printer.current_state = np.random.choice(printer.model[printer.current_state][action][0], size=1, p=printer.model[printer.current_state][action][1]/np.sum(printer.model[printer.current_state][action][1]))[0]
        printer.current_actions = list(printer.model[printer.current_state].keys())
        end_simu = printer.current_state in end
    return printer.current_state, iter

def montecarlo(printer, delta=0.01, epsilon=0.01):
    print('\n\n------------------------------------\nMonte Carlo\n------------------------------------')
    if input("Voulez vous faire du Monte Carlo ? y/n ") != "y":
        return 0
    start = input(f"Choississez un etat de départ parmi : {list(printer.model.keys())} ")
    end = input(f"Choississez un etat de d'arriver parmi : {list(printer.model.keys())} ")
    iter_max = int(input(f"Choississez le nombre d'itération dans une simulation : "))
    N = int((np.log(2) - np.log(delta)) / ((2 * epsilon ** 2))) + 1
    succes = 0
    for i in range(N):
        result, iter = simulate_markov(printer, start=start, end=end, iter_max=iter_max)
        if result in end:
            succes += 1
    print(f"La probabilité y d'obtenir {end} en partant de {start} est destimée par yN = {succes / N} avec P(|yN - y| > {epsilon}) < {delta} en {N} itération")

def SPRT(printer, scheduler=None, epsilon=0.01, alpha=0.01, beta=0.01, iter_max=None, theta=None, start=None, end=None, verbose=True):
    if verbose:
        print('\n\n------------------------------------\nSPRT\n------------------------------------')
        if input("Voulez vous faire du SPRT ? y/n ") != "y":
            return 0
    if start is None:
        start = input(f"Choississez un etat de départ parmi : {list(printer.model.keys())} ")
    if end is None:
        end = input(f"Choississez un etat de d'arriver parmi : {list(printer.model.keys())} ")
    if theta is None:
        theta = float(input(f"Choississez la borne à tester : "))
    if iter_max is None:
        iter_max = int(input(f"Choississez le nombre d'itération dans une simulation : "))
    A = (1 - beta) / alpha
    B = beta / (1 - alpha)
    gamma1 = theta - epsilon
    gamma0 = theta + epsilon
    Rm = 1
    done = Rm >= A or Rm <= B
    iter_SPRT = 0
    while not done:
        iter_SPRT += 1
        result, iter = simulate_markov(printer, scheduler=scheduler, start=start, end=end, iter_max=iter_max)
        if result in end:
            Rm = Rm * (gamma1/gamma0)
        else:
            Rm = Rm * (1 - gamma1) / (1 - gamma0)
        done = Rm >= A or Rm <= B
    if Rm >= A:
        if verbose:
            print(f"La probabilité y d'obtenir {end} en partant de {start} est < {gamma1} en {iter_SPRT} itération")
        else:
            return 'H1'
    elif Rm <= B:
        if verbose:
            print(f"La probabilité y d'obtenir {end} en partant de {start} est > {gamma0} en {iter_SPRT} itération")
        else:
            return 'H0'

def norm1(L1, L2):
    d = []
    for i in range(len(L1)):
        d.append(np.abs(L1[i] - L2[i]))
    return np.max(d)

def norm2(L1, L2):
    d = 0
    for i in range(len(L1)):
        d += (L1[i] - L2[i]) ** 2
    return d ** 0.5

def value_iteration(printer, gamma=0.5, epsilon=0.01, norm=norm2):
    V0 = [printer.reward[s] for s in printer.model.keys()]
    V = [0 for s in printer.model.keys()]
    iter = 0
    while norm(V0, V) > epsilon:
        iter += 1
        for i, s in enumerate(list(printer.model.keys())):
            s_action = []
            for action in printer.model[s].keys():
                s_action.append(0)
                for j, arrive_state in enumerate(printer.model[s][action][0]):
                    s_action[-1] += (printer.model[s][action][1][j] / np.sum(printer.model[s][action][1])) * V0[list(printer.model.keys()).index(arrive_state)] * gamma
            V[i] = printer.reward[s] + np.max(s_action)
        V2 = V0.copy()
        V0 = V.copy()
        V = V2.copy()
    for s in printer.model.keys():
        s_action = []
        for action in printer.model[s].keys():
            s_action.append(0)
            for j, arrive_state in enumerate(printer.model[s][action][0]):
                s_action[-1] += (printer.model[s][action][1][j] / np.sum(printer.model[s][action][1])) * V0[list(printer.model.keys()).index(arrive_state)] * gamma
        printer.theta[s] = list(printer.model[s].keys())[np.argmax(s_action)]
    return iter

def get_path(printer, scheduler, start, end, size=20):
    a = None
    max = 0
    for action in printer.actions:
        if scheduler[start][action] > max:
            a = action
            max = scheduler[start][action]
    path = [[start, a]]
    while len(path) < size and path[-1][0] != end:
        s, a = path[-1]
        s = np.random.choice(printer.model[s][a][0], size=1, p=printer.model[s][a][1]/np.sum(printer.model[s][a][1]))[0]
        max = 0
        a = None
        for action in printer.actions:
            if scheduler[s][action] > max:
                a = action
                max = scheduler[s][action]
        path.append([s, a])
    return path

def satisfied(path, end):
    return path[-1][0] == end

def init_scheduler(printer):
    scheduler = {}
    for s in printer.model.keys():
        scheduler[s] = {}
        for action in printer.actions:
            scheduler[s][action] = (1 / len(printer.actions)) * (action in list(printer.model[s].keys()))
    return scheduler

def scheduler_evaluation(printer, scheduler, N, start, end):
    # initialisation
    R_plus, R_moins, Q = {}, {}, {}
    for s in printer.model.keys():
        R_plus[s], R_moins[s], Q[s] = {}, {}, {}
        for a in printer.actions:
            R_plus[s][a], R_moins[s][a], Q[s][a] = 0, 0, scheduler[s][a]
    paths = []
    for i in range(N):
        path = get_path(printer, scheduler, start, end)
        for j in range(len(path)):
            s, a = path[j]
            if path[j] not in paths:
                paths.append(path[j])
            if satisfied(path, end):
                R_plus[s][a] += 1
            else:
                R_moins[s][a] += 1
    for p in paths:
        s, a = p
        Q[s][a] = R_plus[s][a] / (R_plus[s][a] + R_moins[s][a])
    return Q


def scheduler_improvement(printer, h, epsilon, Q, scheduler):
    for s in Q.keys():
        Q_values = [Q[s][key] for key in Q[s].keys()]
        a = list(Q[s].keys())[np.argmax(Q_values)]
        for action in printer.actions:
            p_sa = (action == a) * (1 - epsilon) + epsilon * (Q[s][a]) / (np.sum(Q_values))
            scheduler[s][a] = h * scheduler[s][a] + (1 - h) * p_sa

def scheduler_optimisation(printer, h, epsilon, N, L, start, end, scheduler=None):
    if scheduler is None:
        scheduler = init_scheduler(printer)
    for i in range(L):
        Q = scheduler_evaluation(printer, scheduler, N, start, end)
        scheduler_improvement(printer, h, epsilon, Q, scheduler)
    return scheduler

def determinise(printer, scheduler):
    for s in printer.model.keys():
        max = 0
        a = None
        for action in printer.actions:
            if scheduler[s][action] > max:
                max = scheduler[s][action]
                a = action
        for action in printer.actions:
            scheduler[s][action] = 1 * (action == a)

def statistical_model_checking(printer, h, epsilon, N, L, p, etha, start, end, theta, ineq):
    T = int(np.log(etha) / np.log(1 - p)) + 1
    for i in range(T):
        print(f"{i}/{T}")
        scheduler = scheduler_optimisation(printer, h, epsilon, N, L, start, end)
        determinise(printer, scheduler)
        if SPRT(printer, epsilon=0.001, alpha=0.01, beta=0.01, iter_max=15, theta=theta, start=start, end=end, verbose=False) != ineq:
            return False
    return True


def is_equal(d1, d2):
    # return true if two dicts are equal
    for key in d1.keys():
        if d1[key] != d2[key]:
            return False
    return True

def create_inverted_graph(printer):
    for state in printer.model.keys():
        printer.inverted_graph[state] = {}
        for action in printer.actions:
            printer.inverted_graph[state][action] = [[],[]]
    for state in printer.model.keys():
        for action in printer.model[state].keys():
            for i, arrival_state in enumerate(printer.model[state][action][0]):
                if state not in printer.inverted_graph[arrival_state][action][0]:
                    printer.inverted_graph[arrival_state][action][0].append(state)
                    printer.inverted_graph[arrival_state][action][1].append(printer.model[state][action][1][i]/np.sum(printer.model[state][action][1]))
    for state in printer.inverted_graph.keys():
        for action in printer.actions:
            if printer.inverted_graph[state][action] == [[],[]]:
                del printer.inverted_graph[state][action]

def recursive_dfs(printer, node, visited=None):
    if visited is None:
        visited = []
    if node not in visited:
        visited.append(node)
    for action in printer.inverted_graph[node].keys():
        unvisited = [n for n in printer.inverted_graph[node][action][0] if n not in visited]
        for new_node in unvisited:
            recursive_dfs(printer, new_node, visited)
    return visited

def get_S1(printer, goal):
    S1 = [goal]
    # on crée le graph inversé
    create_inverted_graph(printer)
    S = recursive_dfs(printer, goal)
    return S1, S
    
def model_checking(printer):
    goal = []
    choosen = False
    while not choosen:
        goal.append(input(f"Choissisez un ou des etats de départ parmis les états disponibles comme objectif {list(printer.model.keys())} "))
        choosen = input("Voulez-vous ajouter un etat supplémentaire ? (y/n) ") == "y"
    S = []
    S1 = []
    for state in goal:
        L = get_S1(printer, goal)
        S1 += L[0]
        S += L[1]
    S0 = list(set(list(printer.model.keys())) - set(S))

if __name__ == "__main__":
    # Example
    mc = {'S0': {'noact': [['S1', 'S2'], [5, 5]]}, 'S1': {'noact': [['S1', 'S2'], [3, 7]]}, 'S2': {'noact': [['S0'], [1]]}, 'S3': {'noact': [['S3'], [1]]}}
    states = ['S0']

    print(prob_states_mc(states, mc))