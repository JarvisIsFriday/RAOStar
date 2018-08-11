#!/usr/bin/env python
#
#  Copyright (c) 2015 MIT. All rights reserved.
#
#   author: Pedro Santana
#   e-mail: psantana@mit.edu
#   website: people.csail.mit.edu/psantana
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#  3. Neither the name(s) of the copyright holders nor the names of its
#     contributors or of the Massachusetts Institute of Technology may be
#     used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
Enumeration Tree for systematic search

@author: Sungkweon Hong (sk5050@mit.edu).

"""


import sys
from collections import deque


class EnumTreeElement(object):
    """
    Generic graph element with a name and a unique ID.
    """

    def __init__(self, name=None, properties={}):
        self.name = name
        self.properties = properties

    __hash__ = object.__hash__

    def set_name(self, new_name):
        self.name = new_name

    def set_properties(self, new_properties):
        if isinstance(new_properties, dict):
            self.properties = new_properties
        else:
            raise TypeError(
                'enumtree element properties should be given as a dictionary.')

    def __eq__(x, y):
        return isinstance(x, EnumTreeElement) and isinstance(y, EnumTreeElement) and (x.name == y.name)

    def __ne__(self, other):
        return not self == other


class EnumTreeNode(EnumTreeElement):
    """
    Class for nodes in the enumeration tree.
    """

    def __init__(self, parent_etree_node, best_action=None, terminal=False, name=None,
                 properties={}, make_unique=False):
        super(EnumTreeNode, self).__init__(name, properties)

        # Parent node of enumeration tree
        self.parent_etree_node = parent_etree_node
        # Dictionary of differences of values from parent enumeration tree node to current enumeration tree node.
        # Key is RAOStarGraphNode's string name, value is another dictionary, which has diff values for value, execution risk and previous best action.
        self.diff = dict()

    def compute_diff(self, prev_node, new_node):
        value_diff = new_node.value - prev_node.value
        exec_risk_diff = new_node.exec_risk - prev_node.exec_risk
        prev_best_action = prev_node.best_action

        self.diff[new_node.name] = {'value_diff':value_diff, 'exec_risk_diff':exec_risk_diff, 'prev_best_action':prev_best_action}


class EnumTree(EnumTreeElement):
    """
    Class representing an enumeration tree.
    """

    def __init__(self, name=None, properties={}):
        super(EnumTree, self).__init__(name, properties)
        # Dictionary of nodes mapping their string names to themselves
        self.nodes = {}
        
    def add_node(self, node):
        """Adds a node to the hypergraph."""
        if not node in self.nodes:
            self.nodes[node.name] = node
