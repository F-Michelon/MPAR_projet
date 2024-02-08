from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import numpy as np
import argparse
import networkx as nx
from matplotlib.widgets import Button, RadioButtons
import matplotlib.pyplot as plt

class gramPrintListener(gramListener):

    def __init__(self):
        pass
        
    def enterDefstates(self, ctx):
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

class gramPrintListenerbis(gramListener):

    def __init__(self):
        self.current_state = None
        self.current_actions = None
        self.model = {}
        
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
        # Renvoie un message d'erreur si le modèle n'est pas bon
        correct_model = True
        for state in list(self.model.keys())[:-1]:
            # test si chaque etat à au moins une action (noact compte)
            if len(self.model[state].keys()) == 0:
                return False
            for action in self.model[state].keys():
                # test si le nombre d'etat d'arrivée est egale au nombre de poids
                if len(self.model[state][action][0]) != len(self.model[state][action][0]):
                    return False
                for i in range(len(self.model[state][action][0])):
                    # test si les etats d'arrivée sont biens des etats
                    if self.model[state][action][0][i] not in list(self.model.keys())[:-1]:
                        return False
                    # test si les poids sont des entiers positifs
                    if type(self.model[state][action][1][i]) == int and self.model[state][action][1][i] > 0:
                        return False
        return correct_model

    def play(self):
        if self.test:
            self.model['actions'].sort()
            G = nx.DiGraph(directed=True)
            # creating edges = possible actions 
            edges_list = []
            for key in list(self.model.keys())[:-1]:
                for action in self.model[key].keys():
                    for output_node in self.model[key][action][0]:
                        edges_list.append((key, output_node))
            G.add_edges_from(edges_list)
            options = {
                'node_size': 800,
                'width': 3,
                'arrowstyle': '-|>',
                'arrowsize': 20,
            }
            # drawing graph
            color = ['red' for node in G.nodes]
            color[list(self.model.keys()).index(self.current_state)] = 'green'
            fig, ax = plt.subplots()
            nodePos = nx.circular_layout(G)
            nx.draw_networkx(G, ax=ax, arrows=True, node_color=color, pos=nodePos, **options)
            plt.axis('off')
            title = f"Vous êtes dans l'état {self.current_state}\nLes actions possibles sont : "
            for x in self.current_actions:
                title += x + ' '
            plt.title(title)

            # adjust radio buttons
            rax = plt.axes([0.05, 0.4, 0.15, 0.30])

            radio = RadioButtons(rax, self.model['actions'])

            def color(action):
                if action in self.current_actions:
                    if action == "noact":
                        new_state = np.random.choice(self.model[str(self.current_state)]["noact"][0], size=1, p=self.model[str(self.current_state)]["noact"][1]/np.sum(self.model[str(self.current_state)]["noact"][1]))[0]
                    elif action in self.model[str(self.current_state)] and action != "noact":
                        new_state = np.random.choice(self.model[str(self.current_state)][action][0], size=1, p=self.model[str(self.current_state)][action][1]/np.sum(self.model[str(self.current_state)][action][1]))[0]
                    else:
                        print("Invalid action")
                        new_state = self.current_state
                    self.current_state = new_state
                    self.current_actions = list(self.model[self.current_state].keys())

                    # drawing graph
                    color = ['red' for node in G.nodes]
                    color[list(self.model.keys()).index(self.current_state)] = 'green'
                    nx.draw_networkx(G, ax=ax, arrows=True, node_color=color, pos=nodePos, **options)
                    plt.axis('off')
                    title = f"Vous êtes dans l'état {self.current_state}\nLes actions possibles sont : "
                    for x in self.current_actions:
                        title += x + ' '
                    ax.set_title(title)
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
    printer = gramPrintListenerbis()
    walker = ParseTreeWalker()
    walker.walk(printer, tree)
    printer.current_state = list(printer.model.keys())[0]
    printer.current_actions = list(printer.model[printer.current_state].keys())
    printer.play()

if __name__ == '__main__':
    main()