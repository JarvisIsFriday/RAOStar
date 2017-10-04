#!/usr/bin/env python
# Author: Yun Chang 
# Email: yunchang@mit.edu 

# A general tree class for constructing trees for rao search 

class Action(object):
    def __init__(self, parent, child):
        self.parent = parent 
        self.child = child
        # note each state here is a State object

    def __repr__(self):
        return "Action(%r,%r,%r)" % (self.parent, self.child)

class State(object):
    def __init__(self, label, probability): 
        self.probability= probability
        self.parent = parent

    def __equal__(self, other):
        return self.label == other.label

class belief(object):
    def __init__(self, stateslist, parent, value):
        self.states = stateslist 
        self.label = label
        self.value = value
        self.parent = parent 

    def expandNode(self, transitionFunction, observeFunction):
        prior = {}
        # trainsition function takes a state in 
        # a belief and transitions them to a set 
        # of states with probabilities 
        # first action update 
        for state in self.states:
            probstate = transitionFunction(state)
            for s in probstate:
                if s[0] not in prior.keys():
                    prior[s[0]] = s[1]
                else:
                    prior[s[0]] += s[1]
        # then observation update 
        for s in prior.keys():
            prior[s]*= observeFunction(s)
        z = sum(prior.values()) # normalization constant 
        # normalize 
        return [State(b, prior[b]) for b in prior.keys()]


class Tree(object):
    def __init__(self):
        self.nodes = set()
        # set consists of the belief objects 
        self.actions = dict()
        self.open_nodes = set()

    def __contains__(self, node):
        return node in self.nodes

    def add_node(self, node):
        # Add a node (belief) to tree
        self.nodes.add(node)
        self.actions[node] = []
    
    def add_action(self, parent, child, parent_val, child_val):
        # parent and child are of the belief object 
        self.add_node(parent)
        self.add_node(child)
        new_action = Action(parent, child)
        self.actions[parent].append(new_action)
        try: 
            self.open_states.remove(parent)
        except KeyError:
            pass
        self.open_states.add(child)

