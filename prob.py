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

if __name__ == "__main__":

    # Example
    mc = {'a':{'a':0.3, 'b':0.7}, 'b':{'a':0.5, 'b':0.5}}
    states = ['a']

    print(prob_states_mc(states, mc))