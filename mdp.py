from antlr4 import *
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
        self.iter = 0
        self.to_white = False
        self.button = None
        
    def enterDefstates(self, ctx):
        for x in ctx.ID():
            self.model[str(x)] = {}     
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        for x in ctx.ID():
            self.model['actions'] = [str(x) for x in ctx.ID()]
        self.model['actions'].append('noact')
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.model[dep][act] = [ids, weights]
        if act not in self.model['actions']:
            self.model['actions'].append(act)
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.model[dep]["noact"] = [ids, weights]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

    def test(self):
        print("Nous allons procéder à des tests pour voir si le modèle est correct, si non voulez-vous le corriger ?")
        answer = input("type yes ")
        if answer == 'yes' or answer == 'y':
            correct = True
        # Renvoie un message d'erreur si le modèle n'est pas bon
        for state in list(self.model.keys())[:-1]:
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
                    if self.model[state][action][0][i] not in list(self.model.keys())[:-1]:
                        print(f"Les etats d'arrivée de {state} ne sont pas des etats")
                        return False
                    # test si les poids sont des entiers positifs
                    if type(self.model[state][action][1][i]) != int or self.model[state][action][1][i] < 0:
                        print("test si les poids sont des entiers positifs")
                        return False
        return True

    def play(self):
        if self.test():
            self.model['actions'].sort()
            G = nx.DiGraph(directed=True)
            
            # creating edges = possible actions
            edges_list = []
            straight_edges = []
            edges_labels = {}
            node_label = {}
            for key in list(self.model.keys())[:-1]:
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
                    if key in list(self.model.keys())[:-1] and key2 in list(self.model.keys())[:-1]:
                        posdist = np.linalg.norm(np.array([nodePos[key], nodePos[key2]]))
            r = np.min(posdist) / 3 # len(self.model.keys())
            for node in G.nodes():
                if node in list(self.model.keys())[:-1] and len(self.model[node].keys()) > 1:
                    for i, action in enumerate(self.model[node].keys()):
                        nodePos[node + action] = nodePos[node] + np.array([np.cos(np.pi/2 + 2*i*np.pi/len(self.model[node].keys())),np.sin(np.pi/2 + 2*i*np.pi/len(self.model[node].keys()))]) * r

            # define params
            node_size = []
            for node in G.nodes():
                if node in list(self.model.keys())[:-1]:
                    node_size.append(1000//len(list(self.model.keys())[:-1]))
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
                title += f'\nLes actions possibles sont : ' + actions + '\niter ={self.iter}'
            plt.title(title)

            # adjust radio buttons
            rax = plt.axes([0, 0.5, 0.15, 0.30])
            color_action = ['black', 'black'] + ['white' for i in range(len(self.model['actions']))]
            color_action[2:len(self.current_actions)] = ['black' for i in range(len(self.current_actions))]
            if 'noact' in self.current_actions:
                radio_labels = ['Hide red edges', 'Random walk'] + ['passe'] + ['nothing' for i in range(len(self.model['actions']) - len(self.current_actions))]
            else:
                radio_labels = ['Hide red edges', 'Random walk'] + ['passe'] + ['nothing' for i in range(len(self.model['actions']) - len(self.current_actions))]
            radio = RadioButtons(rax, radio_labels, activecolor='white')
            radio.set_label_props({'color': color_action})
            radio.set_radio_props({'edgecolor': color_action})

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
                        plt.pause(1)

                elif action == 'Stop random walk':
                    radio.labels[1].set_text('Random walk')
                    fig.canvas.draw()

                elif action in self.current_actions or action == 'passe' or action == 'Nothing':
                    if action == "passe" or action == 'noact':
                        new_state = np.random.choice(self.model[str(self.current_state)]["noact"][0], size=1, p=self.model[str(self.current_state)]["noact"][1]/np.sum(self.model[str(self.current_state)]["noact"][1]))[0]
                    elif action in self.model[str(self.current_state)] and action != "noact":
                        new_state = np.random.choice(self.model[str(self.current_state)][action][0], size=1, p=self.model[str(self.current_state)][action][1]/np.sum(self.model[str(self.current_state)][action][1]))[0]
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
                            for act in self.model['actions']:
                                if act in edge[1]:
                                    straight_edges.append(edge)
                            if edge not in straight_edges:
                                curved_edges.append(edge)

                        label_node = {}
                        node_label = {}
                        for node in nodePos.keys():
                            label_node[node] = nodePos[node]
                            if node not in list(self.model.keys())[:-1]:
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
                        'node_size': 1000//len(list(self.model.keys())[:-1]),
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
                    color_action = ['black', 'black'] + ['white' for i in range(len(self.model['actions']))]
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
        else:
            print("Le modèle n'est pas correct")

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
    printer.current_state = list(printer.model.keys())[0]
    printer.current_actions = list(printer.model[printer.current_state].keys())
    printer.play()

if __name__ == '__main__':
    main()