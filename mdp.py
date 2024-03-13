from antlr4 import *
import sys
sys.path.append('gram')
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import numpy as np
import argparse
import networkx as nx
from matplotlib.widgets import RadioButtons
import matplotlib.pyplot as plt
import my_networkx as my_nx
import time
from fractions import Fraction
import warnings

warnings.filterwarnings("ignore")

class gramPrintListener(gramListener):

    def __init__(self):
        self.current_state = None
        self.current_actions = None
        self.model = {}
        self.inverted_graph = {}
        self.actions = []
        self.iter = 0
        self.to_white = False
        self.button = None
        self.epsilon = 0.05
        self.delta = 0.01
        self.N = (np.log(2) - np.log(self.delta))/(2*(self.epsilon**2))
        self.trace = ['']
        
    def enterDefstates(self, ctx):
        for x in ctx.ID():
            self.model[str(x)] = {}  
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        for x in ctx.ID():
            self.actions = [str(x) for x in ctx.ID()]
        self.actions.append('noact')
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        drop_duplicate_ids = []
        drop_duplicate_weights = []
        for i, state in enumerate(ids):
            if state not in drop_duplicate_ids:
                drop_duplicate_ids.append(state)
                drop_duplicate_weights.append(weights[i])
        self.model[dep][act] = [drop_duplicate_ids, drop_duplicate_weights] 
        if act not in self.actions:
            self.actions.append(act)
        print("Transition from " + dep + " with action "+ act + " and targets " + str(drop_duplicate_ids) + " with weights " + str(drop_duplicate_weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.model[dep] = {}
        self.model[dep]["noact"] = [ids, weights]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

    def test(self):
        print("Nous allons procéder à des tests pour voir si le modèle est correct, dans le cas contraire voulez-vous le corriger ?")
        answer = input("type yes ")
        if answer == 'yes' or answer == 'y':
            correct = True
        # Renvoie un message d'erreur si le modèle n'est pas bon
        for state in self.model.keys():
            # test si chaque etat a au moins une action (noact compte)
            if len(self.model[state].keys()) == 0:
                print(f"{state} n'a pas d'action, voulez-vous qu'il boucle sur lui même ?")
                if not correct:
                    answer = input("type yes ")
                if correct or answer == 'yes' or answer == 'y':
                    self.model[state]['noact'] = [[state],[1]]
                else:
                    return False
            # test si noact est présent en plus d'autres actions
            if 'noact' in self.model[state].keys() and len(self.model[state].keys()) > 1:
                print(f"{state} possede des actions et une transition simple, voulez-vous la supprimer ?")
                if not correct:
                    answer = input("type yes ")
                if correct or answer == 'yes' or answer == 'y':
                    del self.model[state]['noact']
                else:
                    return False
            if len(self.model[state].keys()) == 1 and list(self.model[state].keys())[0] != 'noact':
                print(f"{state} a une action qui n'est pas une transition, voulez-vous corriger ?")
                action = list(self.model[state].keys())[0]
                if not correct:
                    answer = input("type yes ")
                if correct or answer == 'yes' or answer == 'y':
                    self.model[state]['noact'] = self.model[state][action]
                    del self.model[state][action]
                else:
                    return False
            for action in self.model[state].keys():
                # test si le nombre d'etat d'arrivée est egale au nombre de poids
                if len(self.model[state][action][0]) != len(self.model[state][action][1]):
                    print(f"Le nombre d'etat d'arrivée de {state} est different du nombre de poids, voulez-vous corriger ?")
                    if not correct:
                        answer = input("type yes ")
                    if correct or answer == 'yes' or answer == 'y':
                        self.model[state][action][1] = [1 for i in range(self.model[state][action][0])]
                    else:
                        return False
                for i in range(len(self.model[state][action][0])):
                    # test si les etats d'arrivée sont biens des etats
                    arrive_state = self.model[state][action][0][i]
                    if arrive_state not in self.model.keys():
                        print(f"L'etat d'arrivée {arrive_state} de {state} n'est' pas un etat, voulez-vous l'ajouter ?")
                        if not correct:
                            answer = input("type yes ")
                        if correct or answer == 'yes' or answer == 'y':
                            self.model[arrive_state]['noact'] = [[arrive_state], [1]]
                        else:
                            return False
                    # test si les poids sont des entiers positifs
                    if type(self.model[state][action][1][i]) != int or self.model[state][action][1][i] < 0:
                        print("test si les poids sont des entiers positifs")
                        return False
        return True
    
    def simulate_markov(self, start, end, iter_max=100):
        # only for markov chain
        self.current_state = start
        self.current_actions = list(self.model[self.current_state].keys())
        end_simu = self.current_state in end
        iter = 0
        while not end_simu and iter < iter_max:
            iter += 1
            action = self.current_actions[0]
            self.current_state = np.random.choice(self.model[self.current_state][action][0], size=1, p=self.model[self.current_state][action][1]/np.sum(self.model[self.current_state][action][1]))[0]
            self.current_actions = list(self.model[self.current_state].keys())
            end_simu = self.current_state in end
        return self.current_state, iter
    
    def montecarlo(self, delta=0.01, epsilon=0.01):
        print('\n\n------------------------------------\nMONTE CARLO\n------------------------------------')
        if input("Voulez vous faire du SPRT ? y/n ") != "y":
            return 0
        start = input(f"Choississez un etat de départ parmi : {list(self.model.keys())} ")
        end = input(f"Choississez un etat de d'arriver parmi : {list(self.model.keys())} ")
        iter_max = int(input(f"Choississez le nombre d'itération dans une simulation : "))
        N = int((np.log(2) - np.log(delta)) / ((2 * epsilon ** 2))) + 1
        succes = 0
        for i in range(N):
            result, iter = self.simulate_markov(start, end, iter_max)
            if result in end:
                succes += 1
        print(f"La probabilité y d'obtenir {end} en partant de {start} est destimée par yN = {succes / N} avec P(|yN - y| > {epsilon}) < {delta} en {N} itération")
    
    def SPRT(self, epsilon=0.01, alpha=0.01, beta=0.01):
        print('\n\n------------------------------------\nSPRT\n------------------------------------')
        if input("Voulez vous faire du Monte Carlo ? y/n ") != "y":
            return 0
        start = input(f"Choississez un etat de départ parmi : {list(self.model.keys())} ")
        end = input(f"Choississez un etat de d'arriver parmi : {list(self.model.keys())} ")
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
            result, iter = self.simulate_markov(start, end, iter_max)
            if result in end:
                Rm = Rm * (gamma1/gamma0)
            else:
                Rm = Rm * (1 - gamma1) / (1 - gamma0)
            done = Rm >= A or Rm <= B
        if Rm >= A:
            print(f"La probabilité y d'obtenir {end} en partant de {start} est < {gamma1} en {iter_SPRT} itération")
        elif Rm <= B:
            print(f"La probabilité y d'obtenir {end} en partant de {start} est > {gamma0} en {iter_SPRT} itération")

    def play(self):
        self.actions.sort()
        G = nx.DiGraph(directed=True)
        
        # creating edges = possible actions
        edges_list = []
        straight_edges = []
        edges_labels = {}
        node_label = {}
        for key in self.model.keys():
            if key not in node_label.keys():
                node_label[key] = key
            for action in self.model[key].keys():
                for i, output_node in enumerate(self.model[key][action][0]):
                    if action == 'noact':
                        edges_list.append((key, output_node))
                        if output_node not in node_label.keys():
                            node_label[output_node] = output_node
                        tex_label = Fraction(int(np.sum(self.model[key][action][1])),self.model[key][action][1][i])
                        if tex_label.denominator == tex_label.numerator:
                            edges_labels[(key, output_node)] = '1'
                        else:
                            edges_labels[(key, output_node)] = f'{tex_label.denominator}/{tex_label.numerator}'
                    else:
                        node_label[key + action] = action
                        edges_list.append((key + action, output_node))
                        tex_label = Fraction(int(np.sum(self.model[key][action][1])),self.model[key][action][1][i])
                        if tex_label.denominator == tex_label.numerator:
                            edges_labels[(key + action, output_node)] = '1'
                        else:
                            edges_labels[(key + action, output_node)] = f'{tex_label.denominator}/{tex_label.numerator}'
                if action != 'noact':
                    edges_list.append((key, key + action))
                    straight_edges.append((key, key + action))
        G.add_edges_from(edges_list)
        curved_edges = list(set(G.edges()) - set(straight_edges))
        curved_edge_labels = {edge: edges_labels[edge] for edge in curved_edges}
        # Compute pos
        nodePos = nx.circular_layout(G)
        # change fake node position (actions)
        posdist = []
        for key in nodePos:
            for key2 in nodePos:
                if key in self.model.keys() and key2 in self.model.keys():
                    posdist = np.linalg.norm(np.array([nodePos[key], nodePos[key2]]))
        r = np.min(posdist) / 3 # len(self.model.keys())
        for node in G.nodes():
            if node in self.model.keys() and len(self.model[node].keys()) > 1:
                for i, action in enumerate(self.model[node].keys()):
                    nodePos[node + action] = nodePos[node] + np.array([np.cos(np.pi/2 + 2*i*np.pi/len(self.model[node].keys())),np.sin(np.pi/2 + 2*i*np.pi/len(self.model[node].keys()))]) * r

        # define params
        node_size = []
        for node in G.nodes():
            if node in self.model.keys():
                node_size.append(1000//len(self.model.keys()))
            else:
                node_size.append(0)
        color_node = []
        for node in G.nodes:
            if node == self.current_state:
                color_node.append('green')
            else:
                color_node.append('red')
        curved_color_edge = ['red' for node in curved_edges]
        for i in range(len(curved_color_edge)):
            if self.current_state in curved_edges[i][0]:
                curved_color_edge[i] = 'green'
        straight_color_edge = ['red' for node in straight_edges]
        for i in range(len(straight_color_edge)):
            if self.current_state in straight_edges[i][0]:
                straight_color_edge[i] = 'green'

        node_options = {
            'node_size': node_size,
            'node_color': color_node
        }
        straight_edge_options = {
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 5,
            'edge_color': straight_color_edge,
        }
        curved_edge_options = {
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 5,
            'edge_color': curved_color_edge,
            'connectionstyle': f'arc3, rad = {0.25}'
        }

        # drawing graph
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, ax=ax, pos=nodePos, **node_options)
        nx.draw_networkx_labels(G, ax=ax, pos=nodePos, labels=node_label)
        nx.draw_networkx_edges(G, pos=nodePos, ax=ax, edgelist=straight_edges, **straight_edge_options)
        nx.draw_networkx_edges(G, pos=nodePos, ax=ax,edgelist=curved_edges, **curved_edge_options)
        my_nx.my_draw_networkx_edge_labels(G, ax=ax, edge_labels=curved_edge_labels, pos=nodePos, rotate=False,rad = 0.25)
        plt.axis('off')

        title = f"Vous êtes dans l'état {self.current_state}"
        actions = ''
        for x in self.current_actions:
            actions += x + ' '
        if 'noact' in self.current_actions:
            title += f"\nil n'y a pas d'action possible\niter ={self.iter}"
        else:
            title += f'\nLes actions possibles sont : ' + actions + f'\niter ={self.iter}'
        plt.title(title)

        # adjust radio buttons
        rax = plt.axes([0, 0.5, 0.15, 0.30])
        color_action = ['black', 'black'] + ['white' for i in range(len(self.actions))]
        color_action[2:len(self.current_actions)] = ['black' for i in range(len(self.current_actions))]
        if 'noact' in self.current_actions:
            radio_labels = ['Hide red edges', 'Random walk'] + ['passe'] + ['' for i in range(len(self.actions) - len(self.current_actions))]
        else:
            radio_labels = ['Hide red edges', 'Random walk'] + [action for action in self.current_actions]
        radio = RadioButtons(rax, radio_labels, activecolor='white')
        radio.set_label_props({'color': color_action})
        radio.set_radio_props({'edgecolor': color_action})

        # on incrémente la trace
        self.trace.append(self.current_state)

        def color(action):
            self.button = action
            if action == 'Hide red edges':
                self.to_white = True
                self.iter -= 1
                radio.labels[0].set_text('Show all edges')
                color("Nothing")

            elif action == 'Show all edges':
                self.to_white = False
                self.iter -= 1
                radio.labels[0].set_text('Hide red edges')
                color("Nothing")

            elif action == 'Random walk':
                radio.labels[1].set_text('Stop random walk')
                while self.button != 'Stop random walk':
                    random_action = np.random.choice(self.current_actions, size=1)[0]
                    color(random_action)
                    plt.pause(0.1)

            elif action == 'Stop random walk':
                radio.labels[1].set_text('Random walk')
                fig.canvas.draw()

            elif action in self.current_actions or action == 'passe' or action == 'Nothing':
                if action == "passe" or action == 'noact':
                    new_state = np.random.choice(self.model[str(self.current_state)]["noact"][0], size=1, p=self.model[str(self.current_state)]["noact"][1]/np.sum(self.model[str(self.current_state)]["noact"][1]))[0]
                    # on incrémente la trace
                    self.trace.append('noact')
                    self.trace.append(new_state)
                elif action in self.model[str(self.current_state)] and action != "noact":
                    new_state = np.random.choice(self.model[str(self.current_state)][action][0], size=1, p=self.model[str(self.current_state)][action][1]/np.sum(self.model[str(self.current_state)][action][1]))[0]
                    # on incrémente la trace
                    self.trace.append(action)
                    self.trace.append(new_state)
                else:
                    new_state = self.current_state
                self.current_state = new_state
                self.current_actions = list(self.model[self.current_state].keys())

                # drawing graph
                if self.to_white:
                    edge_color = 'white'

                    straight_edges = []
                    for edge in G.edges():
                        if self.current_state == edge[0] and edge[1] not in self.model.keys():
                            straight_edges.append(edge)
                        elif self.current_state in edge[0]:
                            curved_edges.append(edge)
                        
                    label_node = {}
                    node_label = {}
                    for node in nodePos.keys():
                        if node in list(self.model.keys()):
                            label_node[node] = nodePos[node]
                            node_label[node] = node
                        else:
                            for action in self.current_actions:
                                if self.current_state + action == node:
                                    label_node[node] = nodePos[node]
                                    node_label[node] = action
                    curved_edge_labels = {}
                    for edge in curved_edges:
                        for action in self.current_actions:
                            if self.current_state in edge[0] or self.current_state + action in edge[1]:
                                curved_edge_labels[edge] = edges_labels[edge]
                else:
                    edge_color = 'red'

                    straight_edges = []
                    for edge in G.edges():
                        for act in self.actions:
                            if act in edge[1]:
                                straight_edges.append(edge)
                        if edge not in straight_edges:
                            curved_edges.append(edge)

                    label_node = {}
                    node_label = {}
                    for node in nodePos.keys():
                        label_node[node] = nodePos[node]
                        if node not in self.model.keys():
                            node_label[node] = node[-1]
                        else:
                            node_label[node] = node
                    curved_edge_labels = {edge: edges_labels[edge] for edge in curved_edges}

                color_node = []
                for node in node_label.keys():
                    if node == self.current_state:
                        color_node.append('green')
                    elif node not in self.model.keys():
                        color_node.append('white')
                    else:
                        color_node.append('red')
                curved_color_edge = [edge_color for node in curved_edges]
                for i in range(len(curved_color_edge)):
                    if self.current_state in curved_edges[i][0]:
                        curved_color_edge[i] = 'green'
                straight_color_edge = [edge_color for node in straight_edges]
                for i in range(len(straight_color_edge)):
                    if self.current_state in straight_edges[i][0]:
                        straight_color_edge[i] = 'green'

                node_options = {
                    'node_size': 1000//len(self.model.keys()),
                    'node_color': color_node
                }
                straight_edge_options = {
                    'width': 1,
                    'arrowstyle': '-|>',
                    'arrowsize': 5,
                    'edge_color': straight_color_edge,
                }
                curved_edge_options = {
                    'width': 1,
                    'arrowstyle': '-|>',
                    'arrowsize': 5,
                    'edge_color': curved_color_edge,
                    'connectionstyle': f'arc3, rad = {0.25}'
                }

                # drawing graph
                ax.clear()
                nx.draw_networkx_nodes(G, ax=ax, pos=nodePos, nodelist=list(node_label.keys()), **node_options)
                nx.draw_networkx_labels(G, ax=ax, pos=nodePos, labels=node_label)
                nx.draw_networkx_edges(G, pos=nodePos, ax=ax, edgelist=straight_edges, **straight_edge_options)
                nx.draw_networkx_edges(G, pos=nodePos, ax=ax,edgelist=curved_edges, **curved_edge_options)
                my_nx.my_draw_networkx_edge_labels(G, ax=ax, edge_labels=curved_edge_labels, pos=nodePos, rotate=False,rad = 0.25)
                ax.axis('off')
                self.iter += 1
                title = f"Vous êtes dans l'état {self.current_state}"
                actions = ''
                for x in self.current_actions:
                    actions += x + ' '
                if 'noact' in self.current_actions:
                    title += f"\nil n'y a pas d'action possible\niter ={self.iter}"
                else:
                    title += f'\nLes actions possibles sont : ' + actions + f'\niter ={self.iter}'
                ax.set_title(title)
                color_action = ['black', 'black'] + ['white' for i in range(len(self.actions))]
                color_action[2:len(self.current_actions)] = ['black' for i in range(len(self.current_actions))]
                for i, action in enumerate(self.current_actions):
                    if action == 'noact':
                        radio.labels[2 + i].set_text('passe')
                    else:
                        radio.labels[2 + i].set_text(action)
                radio.set_label_props({'color': color_action})
                radio.set_radio_props({'edgecolor': color_action})
                fig.canvas.draw()

        radio.on_clicked(color)
        plt.axis('off')
        plt.show()
    
    def play_terminal(self):
        print("\n--------------\n")
        finished = False
        self.trace.append(self.current_state)
        while not finished:
            finished = input("Voulez-vous arrêter? (y/n) ") == "y"
            print(f"\nVous êtes dans l'état {self.current_state} à l'itération {self.iter}")
            print("Actions possibles: " + str([x for x in self.current_actions]))
            for action in self.current_actions:
                print(f"Etats d'arrivés pour l'action {action}: {self.model[self.current_state][action][0]} Poids des transitions: {self.model[self.current_state][action][1]}")
            action = input("Choissisez une action: ")
            if action in self.model[str(self.current_state)]:
                new_state = np.random.choice(self.model[str(self.current_state)][action][0], size=1, p=self.model[str(self.current_state)][action][1]/np.sum(self.model[str(self.current_state)][action][1]))[0]
                self.iter += 1
                self.trace.append(action)
                self.trace.append(new_state)
            else:
                print("Action invalide")
                new_state = self.current_state
            self.current_state = new_state
            self.current_actions = list(self.model[self.current_state].keys())

    def print_trace(self):
        if len(self.trace) > 0:
            trace = ""
            for step in self.trace[:-1]:
                if step in self.model.keys():
                    trace += step + ' + '
                if step in self.actions:
                    trace += step + ' →' + ' '
            trace += self.trace[-1]
            return trace
        
    def create_inverted_graph(self):
        for state in self.model.keys():
            self.inverted_graph[state] = {}
            for action in self.actions:
                self.inverted_graph[state][action] = [[],[]]
        for state in self.model.keys():
            for action in self.model[state].keys():
                for i, arrival_state in enumerate(self.model[state][action][0]):
                    if state not in self.inverted_graph[arrival_state][action][0]:
                        self.inverted_graph[arrival_state][action][0].append(state)
                        self.inverted_graph[arrival_state][action][1].append(self.model[state][action][1][i]/np.sum(self.model[state][action][1]))
        for state in self.inverted_graph.keys():
            for action in self.actions:
                if self.inverted_graph[state][action] == [[],[]]:
                    del self.inverted_graph[state][action]


    def recursive_dfs(self, node, visited=None):

        if visited is None:
            visited = []

        if node not in visited:
            visited.append(node)

        for action in self.inverted_graph[node].keys():
            unvisited = [n for n in self.inverted_graph[node][action][0] if n not in visited]

            for new_node in unvisited:
                self.recursive_dfs(new_node, visited)

        return visited

    def get_S1(self, goal):
        S1 = [goal]
        # on crée le graph inversé
        self.inverted_graph()
        S = self.recursive_dfs(goal)
        return S1, S
        

    def model_checking(self):
        goal = []
        choosen = False
        while not choosen:
            goal.append(input(f"Choissisez un ou des etats de départ parmis les états disponibles comme objectif {list(self.model.keys())} "))
            choosen = input("Voulez-vous ajouter un etat supplémentaire ? (y/n) ") == "y"
        S = []
        S1 = []
        for state in goal:
            L = self.get_S1(goal)
            S1 += L[0]
            S += L[1]
        S0 = list(set(list(self.model.keys())) - set(S))


    def resume(self):
        print('\n\n------------------------------------\nRESUME\n------------------------------------')
        for state in self.model.keys():
            for action in self.model[state].keys():
                print(f"Etat {state} avec l'action {action} qui transite vers {self.model[state][action][0]} avec les poids {self.model[state][action][0]}")
        print(f"Vous avez fait {self.iter} itérations dans ce modèle")
        print("Voici la trace de votre parcourt " + self.print_trace())

def main():
    parser = argparse.ArgumentParser(description='Choose a file to read rules from.')
    parser.add_argument('filename',type=str)
    args = parser.parse_args()
    lexer = gramLexer(FileStream(args.filename))
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()
    printer = gramPrintListener()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    if printer.test():
        printer.current_state = list(printer.model.keys())[0]
        printer.current_actions = list(printer.model[printer.current_state].keys())
        answer = input("Entrer 1 pour intéragir visuellement avec le modèle, 2 pour intéragir avec le terminal ")
        if answer == '1':
            printer.play()
        if answer == '2':
            printer.play_terminal()
        printer.resume()
        print(printer.model)
        printer.create_inverted_graph()
        print(printer.inverted_graph)
        print(printer.recursive_dfs(node='S1'))
        printer.montecarlo()
        printer.SPRT()
    else:
        print("Le modèle n'est pas correct")

if __name__ == '__main__':
    main()