from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import numpy as np
import argparse
        
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
        self.model = {}
        pass
        
    def enterDefstates(self, ctx):
        for x in ctx.ID():
            self.model[str(x)] = {}     
        print("States: %s" % str([str(x) for x in ctx.ID()]))

    def enterDefactions(self, ctx):
        print("Actions: %s" % str([str(x) for x in ctx.ID()]))

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.model[dep][act] = [ids, weights]
        print("Transition from " + dep + " with action "+ act + " and targets " + str(ids) + " with weights " + str(weights))
        
    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        self.model[dep]["noact"] = [ids, weights]
        print("Transition from " + dep + " with no action and targets " + str(ids) + " with weights " + str(weights))

    def play(self):
        print(self.model)
        finished = False
        self.current_state = list(self.model.keys())[0]
        while not finished:
            print("Current state: " + str(self.current_state))
            print("Possible actions: " + str([x for x in self.model[str(self.current_state)].keys()]))
            action = input("Choose an action: ")
            if action == "noact":
                new_state = np.random.choice(self.model[str(self.current_state)]["noact"][0], size=1, p=self.model[str(self.current_state)]["noact"][1]/np.sum(self.model[str(self.current_state)]["noact"][1]))[0]
                # print("Transition to " + str(self.model[str(self.current_state)]["noact"][0][0]) + " with weight " + str(self.model[str(self.current_state)]["noact"][1][0]))
            elif action in self.model[str(self.current_state)] and action != "noact":
                new_state = np.random.choice(self.model[str(self.current_state)][action][0], size=1, p=self.model[str(self.current_state)][action][1]/np.sum(self.model[str(self.current_state)][action][1]))[0]
                # print("Transition to " + str(self.model[str(self.current_state)][action][0][0]) + " with weight " + str(self.model[str(self.current_state)][action][1][0]))
            else:
                print("Invalid action")
                new_state = self.current_state
            self.current_state = new_state
            finished = input("Do you want to finish? (y/n) ") == "y"


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
    printer.play()

if __name__ == '__main__':
    main()