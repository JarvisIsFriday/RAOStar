#!/usr/bin/env python 

# convert graph obtained from RAO* to json file to be used with 
# Matt Deyo's policy visualizer http://mdeyo.com/policy-vis/ 

# Input G is of the RAOStarHyperGraph class found in raostarhypergraph.py 

import json

default_settings = {
		"nodes": {
			"display-properties": ["node_id", "value"],
			"color": "#99ccff",
			"terminal-color": "#7BE141"
		},
		"edges": {
			"display-properties": ["action", "probability"],
			"color": "#99ccff"
		},
		"hierarchical": "true",
		"nodeSpacing": 550,
		"levelSpacing": 200,
		"color": "blue"
	}

def node_info(node_object, cc): 
	nd = node_object
	nd_info = {
			"state": str(nd.state.belief),
			"acceptable-risk-ub": cc,
			"execution-risk-bound": [0,0, nd.exec_risk_bound],
			"state-risk": nd.risk,
			"value": nd.value,
			"is-terminal": str(nd.terminal)}
	return nd_info

def graph_to_json(G, cc, filename, settings=default_settings):
	# first place everything in generic dictionary 
	graph_info = {"nodes":{}, "edges":{}, "settings":settings}
	nodes = G.nodes
	node_strings = nodes.keys()
	edges = G.hyperedges
	parents = G.hyperedges.keys()
	n_ind = 0 # nodes str index 
	# add edges and nodes 
	added_nodes = {} # given node_name: node-i
	for i in range(len(parents)):
		for op in edges[parents[i]]:
			edge_str = "edge-%d"%(i)
			# add parents to node list
			if parents[i].name not in added_nodes: # add node 
				nodestr = "node-%d"%(n_ind)
				added_nodes[parents[i].name] =  nodestr
				graph_info["nodes"][nodestr] = node_info(parents[i],cc)
				n_ind += 1
			# store edgfe information 
			e_info = {
				"action": str(op.name),
				"predecessor": added_nodes[parents[i].name],
				"successors": {}}
			for c in edges[parents[i]][op]: # children 
				if c.name not in added_nodes: # add if necessary 
					nodestr = "node-%d"%(n_ind)
					added_nodes[c.name] = nodestr
					graph_info["nodes"][nodestr] = node_info(c,cc)
					n_ind += 1
				e_info["successors"][added_nodes[c.name]] = {"probability":0.1}
			graph_info["edges"][edge_str] = e_info

	with open(filename, 'w') as fjson:
			json.dump(graph_info, fjson)
	print(added_nodes)
	return graph_info

