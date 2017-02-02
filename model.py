import numpy as np
import networkx as nx
import pandas as pd

# Error is calculated based on eso1, eso2 and ese
def calc_error(graph):
    # Error is calculated based on the polls
    emp_eso1 = graph.node['eso1']['emp_data']
    sim_eso1 = pd.Series(graph.node['eso1']['status'])
    diff1 = emp_eso1 - sim_eso1

    emp_eso2 = graph.node['eso2']['emp_data']
    sim_eso2 = pd.Series(graph.node['eso2']['status'])
    diff2 = emp_eso2 - sim_eso2

    emp_ese = graph.node['ese']['emp_data']
    sim_ese = pd.Series(graph.node['ese']['status'])
    diff3 = emp_ese - sim_ese

    return diff1.pow(2).sum() + diff2.pow(2).sum() + diff3.pow(2).sum()


# mu = speed factor
def run_model(g, parameters=[0.5], delta_t = 0.5, timesteps = 83):
    graph = g.copy()
    mu = parameters[0]

    rng = np.arange(0.0, timesteps*delta_t, delta_t)
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
                if node != 'pers':
                    previous_state = graph.node[node]['status'][t - delta_t]
            except:
                print node, t, delta_t

            if func != '0':
                # If it is identity, the operation is based on the only neighbor.
                if func == 'id':
                    try:
                        weight = graph.edge[graph.predecessors(node)[0]][node]['weight']
                        state_pred = graph.node[graph.predecessors(node)[0]]['status'][t - delta_t]
                    except:
                        print '<time ', t, '> node:', graph.predecessors(node)[0], '-> ', node, '(id)'
                        print t - delta_t

                    graph.node[node]['status'][t] = previous_state + mu * (weight * state_pred - previous_state) * delta_t


                elif func == 'ssum':
                    # Get all the neighbors values.
                    sum_weights = 0
                    sum_products = 0

                    try:
                        for neig in graph.predecessors(node):
                            if neig != 'pers':
                                neig_w = graph.edge[neig][node]['weight']
                                neig_s = graph.node[neig]['status'][t - delta_t]
                                sum_weights = sum_weights + neig_w
                                sum_products = sum_products + neig_w * neig_s
                        try:
                            traits = graph.node['pers']['traits']
                            personality = traits[['scrup', 'domini', 'persis', 'secur', 'achieve', 'power', 'coop',
                                                  'polite', 'open_cul', 'open_exp', 'universalism',
                                                  'benevolence']].sum()
                            personality2 = traits[['orderly', 'convent', 'better_org', 'open_minded', 'creative',
                                                   'curious', 'novelty_seek']].sum()
                            if node == 'pp1':
                                sum_weights = sum_weights + graph.edge['pers'][node]['weight']
                                sum_products = sum_products + personality

                            elif node == 'pp2':
                                sum_weights = sum_weights + graph.edge['pers'][node]['weight']
                                sum_products = sum_products + personality2
                        except:
                            print '(', t, ') problems with pp1 or pp2:', node

                        graph.node[node]['status'][t] = previous_state + mu * (sum_products / sum_weights - previous_state) * delta_t
                    except:
                        print '<time ', t, '> node:', neig, '-> ', node, '(ssum)'
                else:
                    print 'It shouldnt be here!'

    return graph
