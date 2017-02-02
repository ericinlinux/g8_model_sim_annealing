import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import model as m
from random import random

def generate_graph(df, weightList=None):
    graph = nx.DiGraph()
    edges_f = open('edges.txt')
    nodes_f = open('nodes.txt')
    traits_f = open('./data/personality.txt')

    # Insert nodes
    for line in nodes_f:
        node, func = line.replace(" ", "").strip().split(',')
        # Node not included
        if node not in graph.nodes():
            if node == 'eso1' or node == 'eso2' or node == 'ese':
                graph.add_node(node, attr_dict={'pos': 'output', 'func': func, 'status':{}} )
            elif func == 'id' or func == 'ssum':
                graph.add_node(node, attr_dict={'pos': 'inner', 'func': func, 'status':{}} )
            else:
                graph.add_node(node, attr_dict={'pos': 'input', 'func': func, 'status':{}} )
        else:
            print '<CONFLICT> Node already included in the list!'
            exit()

    outWeightList = []
    # Insert edges
    if weightList == None:
        for line in edges_f:
            source, target, w = line.replace(" ", "").strip().split(',')
            # Comment this line if you don't wanna start with random values
            w = float(w)*random()
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))
    else:
        for line in weightList:
            ((source, target), w) = line
            graph.add_edge(source, target, weight=float(w))
            outWeightList.append(((source, target), float(w)))

    persDict = {}
    for line in traits_f:
        trait, value = line.replace(" ", "").strip().split(',')
        persDict[trait] = float(value)

    # Initiating the information in the graph
    graph.node['pers']['traits'] = pd.Series(persDict)

    graph.node['wsc1']['status'] = df['wsc1']
    graph.node['wsc2']['status'] = df['wsc2']
    graph.node['wso1']['status'] = df['wso1']
    graph.node['wso2']['status'] = df['wso2']

    graph.node['wse']['status'] = df['wse']


    graph.node['ese']['emp_data'] = df['ese']
    graph.node['eso1']['emp_data'] = df['eso1']
    graph.node['eso2']['emp_data'] = df['eso2']

    return graph, outWeightList


'''
New parameters will be adjusted as parameters + rand[-0.05, 0.05]
'''
def neighbor(parameters):
    #print 'Received: ', parameters
    inf = -0.05
    sup = 0.05
    minn = 0.0000000001
    maxn = 1
    new_parameters = []

    for p in parameters:
        new_p = p[1] + ((sup - inf) * random() + inf)
        new_p = minn if new_p < minn else maxn if new_p > maxn else new_p
        new_parameters.append((p[0], new_p))
        '''
        if p[0][0] == 'conf_media' and p[0][1] == 'susp_media':
            minn = -1.0
            maxn = -0.000000001
            new_p = p[1] + ((sup - inf) * random() + inf)
            new_p = minn if new_p < minn else maxn if new_p > maxn else new_p
            new_parameters.append((p[0], new_p))
        elif p[0][0] == 'media_neg_sent' and p[0][1] == 'sent_pvv':
            minn = -1.0
            maxn = -0.000000001
            new_p = p[1] + ((sup - inf) * random() + inf)
            new_p = minn if new_p < minn else maxn if new_p > maxn else new_p
            new_parameters.append((p[0], new_p))
        elif p[0][0] == 'conf_pvv' and p[0][1] == 'polls':
            minn = 0.1
            maxn = 1
            new_p = p[1] + ((sup - inf) * random() + inf)
            new_p = minn if new_p < minn else maxn if new_p > maxn else new_p
            new_parameters.append((p[0], new_p))
        else:
            minn = 0.0000000001
            maxn = 1
            new_p = p[1] + ((sup - inf) * random() + inf)
            new_p = minn if new_p < minn else maxn if new_p > maxn else new_p
            new_parameters.append((p[0], new_p))
        '''
    #parameters[0] = parameters[0] + ((sup-inf)*random() + inf)
    #parameters[0] = minn if parameters[0] < minn else maxn if parameters[0] > maxn else parameters[0]
    ##parameters[1] = parameters[1] + ((sup - inf) * random() + inf)
    ##parameters[1] = minn if parameters[1] < minn else maxn if parameters[1] > maxn else parameters[1]
    #print 'Returned: ', parameters
    return new_parameters




def acceptance_probability(old_cost, new_cost, T):
    delta = new_cost-old_cost
    probability = np.exp(-delta/T)
    return probability


def prep_df():
    # Building the dataframe with all the data from the files
    df = pd.read_excel('./data/Fake data.xlsx')
    df_relevant = df[[u'wse', u'wso1', u'wso2', u'wsc1', u'wsc2', u'ese', u'eso1', u'eso2']]
    return df_relevant


def plot_results(parameters, cost_hist, parameters_hist):
    i = 0
    new_dict = {}
    for item in parameters_hist:
        for parameter in item:
            if parameter[0] not in new_dict.keys():
                new_dict[parameter[0]] = []
            else:
                new_dict[parameter[0]].append(parameter[1])
        i += 1

    df = pd.DataFrame(new_dict)
    df.columns.names = ['source', 'target']

    fig = plt.figure(figsize=((20, 30)))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_title('Cost function', size=26, style="oblique", weight='bold')
    ax1.set_xlabel('Epochs', size=24)
    ax1.set_ylabel('Mean squared error (MSE)', size=24)
    ax1.tick_params(labelsize=20)
    ax1.plot(cost_hist)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.set_title('Parameters (edges weights)', size=26, style="oblique", weight='bold')
    ax2.set_xlabel('Epochs', size=24)
    ax2.set_ylabel('Value', size=24)
    ax2.tick_params(labelsize=20)
    ax2.legend(loc='best', title="")
    df.plot(ax=ax2)

    ax3 = fig.add_subplot(3, 1, 3)

    # ax3.bar(index, dict(parameters).values())
    df2 = pd.DataFrame(parameters)
    df2.index = df2[0]
    df2['positive'] = df2[1] > 0

    # df2.plot(ax=ax3, kind='barh', color='green')
    df2[1].plot(kind='barh', color=df2.positive.map({True: 'g', False: 'r'}))

    ax3.set_title('Final parameters', size=26, fontname="Bold")
    ax3.set_xlabel('Final value', size=24)
    ax3.set_ylabel('Parameters', size=24)
    ax3.tick_params(labelsize=20)

    fig.savefig('results.png')

def main():
    # Get DF working with all the data
    df = prep_df()

    # Keeping history (vectors)
    cost_hist = list([])
    parameters_hist = list([])

    # Initiate graph (with standard edges)
    initial_graph, initial_parameters = generate_graph(df)

    # Actual cost
    g = m.run_model(initial_graph)
    old_cost = m.calc_error(g)

    # Initial parameters
    # parameters = [0.5]


    cost_hist.append(old_cost)
    parameters_hist.append(initial_parameters)

    T = 1.0
    T_min = 0.001
    # original = 0.9
    alpha = 0.9
    parameters = initial_parameters

    while T > T_min:
        print 'Temp: ', T
        #graph, parameters = generate_graph(df, initial_parameters)

        i = 1
        # original = 100
        while i <= 100:
            new_parameters = list(neighbor(parameters))
            graph, parameters = generate_graph(df, new_parameters)
            g = m.run_model(graph)
            new_cost = m.calc_error(g)

            if new_cost < old_cost:
                parameters = new_parameters[:]
                parameters_hist.append(parameters)
                old_cost = new_cost
                #print new_cost
                cost_hist.append(old_cost)
            else:
                ap = acceptance_probability(old_cost, new_cost, T)
                if ap > random():
                    parameters = new_parameters[:]
                    parameters_hist.append(parameters)
                    old_cost = new_cost
                    #print new_cost
                    cost_hist.append(old_cost)
            i += 1
        T = T*alpha

    plot_results(parameters, cost_hist, parameters_hist)

    return parameters, cost_hist, parameters_hist

if __name__ == "__main__":
    main()
