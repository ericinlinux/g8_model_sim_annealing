import numpy as np
import networkx as nx
import pandas as pd

def calc_error(graph):
    # Error is calculated based on the polls
    emp_polls = graph.node['polls']['emp_data']
    sim_polls = pd.Series(graph.node['polls']['status'])
    diff = emp_polls - sim_polls
    return diff.pow(2).sum()


# mu = speed factor
def run_model(g, parameters=[0.5], delta_t = 1, timesteps = 204):
    graph = g.copy()
    mu = parameters[0]

    rng = np.arange(0.0, timesteps, delta_t)
    for t in rng:
        # Initialize the nodes
        if t == 0:
            for node in graph.nodes():
                func = graph.node[node]['func']
                pos = graph.node[node]['pos']
                # if func = 0, the values should be stable...
                if func != '0':
                    if pos == 'output':
                        graph.node[node]['status'] = {0:graph.node[node]['emp_data'][0]}
                        #print 'Initial poll data (t0) = ', graph.node[node]['emp_data'][0]
                    else:
                        graph.node[node]['status'] = {0:0}
            continue

        # for each node, update the values
        #if t%50 == 0:
        #    print t

        for node in graph.nodes():
            '''
                For each node (not 0 nodes...):
                    get the neighbors
                    get the function
                    get the weights for the edges
                    calculate the new status value for the node in time t
            '''
            func = graph.node[node]['func']
            try:
                previous_state = graph.node[node]['status'][t-delta_t]
            except:
                print node, t, delta_t

            if func != '0':
                # If it is identity, the operation is based on the only neighbor.
                if func == 'id':
                    try:
                        weight = graph.edge[graph.predecessors(node)[0]][node]['weight']
                        state_pred = graph.node[graph.predecessors(node)[0]]['status'][t-delta_t]

                        graph.node[node]['status'][t] = previous_state + mu*(weight*state_pred - previous_state)*delta_t
                    except:
                        #print '<> node ', node, '<-', graph.predecessors(node)[0], graph.node[graph.predecessors(node)[0]]
                        print '<time ', t, '> node:', graph.predecessors(node)[0], '-> ', node ,'(id)'
                        print t-delta_t
                elif func == 'ssum':
                    # Get all the neighbors values.
                    sum_weights = 0
                    sum_products = 0
                    try:
                        for neig in graph.predecessors(node):
                            neig_w = graph.edge[neig][node]['weight']
                            neig_s = graph.node[neig]['status'][t-delta_t]

                            sum_weights = sum_weights + neig_w
                            sum_products = sum_products + neig_w * neig_s

                            graph.node[node]['status'][t] = previous_state + mu * (sum_products/sum_weights - previous_state)*delta_t
                    except:
                        print '<time ', t, '> node:', neig, '-> ', node ,'(ssum)'
                else:
                    #graph.node[node]['status'][t] = 0
                    print 'It shouldnt be here!'

    return graph
