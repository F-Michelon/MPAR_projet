'''
{state1 : {state2: prob, state3: prob, ...}, ...} (mc)
{state1 : {action1: {state2: prob, state3: prob, ...}, action2: {state2: prob, state3: prob, ...}, ...}, ...} (mdp)
'''

import numpy as np

def is_reachable_mc(mc:dict, s:str, t:str):

    '''
    Given a Markov chain mc and two states s and t,
    return True if t is reachable from s, and False otherwise.
    '''

    if s == t:
        return True
    elif t in mc[s]:
        return True
    else:
        for s_ in mc[s]:
            if is_reachable_mc(mc, s_, t):
                return True
    return False

def is_reachable_mdp(mdp:dict, s:str, t:str):

    '''
    Given a Markv decision process mdp and two states s and t,
    return True if t is reachable from s, and False otherwise.
    '''

    neighbours = [v.keys() for v in mdp[s].values()]
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
            A[S_tilde.index(s), S_tilde.index(t)] = mc[s][t]

    # Create the vector b containing the probabilities of reaching the set of states within one step for each state in S_tilde
    b = np.zeros(len(S_tilde))
    for s in S_tilde:
        p=0
        for u in states:
            p+=mc[s][u]
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

def simulate_markov(printer, start, end, iter_max=100):
    # only for markov chain
    printer.current_state = start
    printer.current_actions = list(printer.model[printer.current_state].keys())
    end_simu = printer.current_state in end
    iter = 0
    while not end_simu and iter < iter_max:
        iter += 1
        action = printer.current_actions[0]
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
        result, iter = printer.simulate_markov(start, end, iter_max)
        if result in end:
            succes += 1
    print(f"La probabilité y d'obtenir {end} en partant de {start} est destimée par yN = {succes / N} avec P(|yN - y| > {epsilon}) < {delta} en {N} itération")

def SPRT(printer, epsilon=0.01, alpha=0.01, beta=0.01):
    print('\n\n------------------------------------\nSPRT\n------------------------------------')
    if input("Voulez vous faire du SPRT ? y/n ") != "y":
        return 0
    start = input(f"Choississez un etat de départ parmi : {list(printer.model.keys())} ")
    end = input(f"Choississez un etat de d'arriver parmi : {list(printer.model.keys())} ")
    theta = float(input(f"Choississez la borne à tester : "))
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
        result, iter = printer.simulate_markov(start, end, iter_max)
        if result in end:
            Rm = Rm * (gamma1/gamma0)
        else:
            Rm = Rm * (1 - gamma1) / (1 - gamma0)
        done = Rm >= A or Rm <= B
    if Rm >= A:
        print(f"La probabilité y d'obtenir {end} en partant de {start} est < {gamma1} en {iter_SPRT} itération")
    elif Rm <= B:
        print(f"La probabilité y d'obtenir {end} en partant de {start} est > {gamma0} en {iter_SPRT} itération")

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

def value_iteration(printer, gamma=1, epsilon=1):
    V0 = [printer.states[s]['reward'] for s in printer.model.keys()]
    V = [0 for s in printer.model.keys()]
    while norm1(V0, V) > epsilon:
        for i, s in enumerate(list(printer.model.keys())):
            s_action = []
            for action in printer.model[s].keys():
                s_action.append(printer.states[s]['reward'] + np.sum([printer.model[s][action][1][printer.model[s][action][0].index(list(printer.model.keys())[i])] * V0[i] / np.sum(printer.model[s][action][1]) for i in range(len(V0)) if list(printer.model.keys())[i] in printer.model[s][action][0]]))
            V[i] = np.max(s_action)
        V2 = V0.copy()
        V0 = V.copy()
        V = V2.copy()
    for s in printer.model.keys():
        s_action = []
        for action in printer.model[s].keys():
            s_action.append(printer.states[s]['reward'] + np.sum([printer.model[s][action][1][printer.model[s][action][0].index(list(printer.model.keys())[i])] * V0[i] / np.sum(printer.model[s][action][1]) for i in range(len(V0)) if list(printer.model.keys())[i] in printer.model[s][action][0]]))
        printer.theta[s] = list(printer.model[s].keys())[np.argmax(s_action)]

if __name__ == "__main__":
    # Example
    mc = {'a':{'a':0.3, 'b':0.7}, 'b':{'a':0.5, 'b':0.5}}
    states = ['a']

    print(prob_states_mc(states, mc))