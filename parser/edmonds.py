from collections import defaultdict
from collections import namedtuple
import sys
import numpy as np


Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


def min_spanning_arborescence(arcs, sink):
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[sink] = sink
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == sink:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[tail_rep].weight > arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, sink)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, sink)
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}


def find_cycle(successor, sink):
    visited = {sink}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, sink):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == sink:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[sink]
    while stack:
        arc = stack.pop()
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail

import numpy as np

def make_ordered_list(tree, nWords):
    lst = np.zeros(nWords)
    for arc in tree:
        #arc = tree[index]
        tail = arc.tail
        head = arc.head
        lst[tail] = head
    return lst

def make_ordered_list_arcs(tree, nWords):
    lst = [0]*nWords#np.zeros(nWords)
    for index in tree:
        arc = tree[index]
        tail = arc.tail
        head = arc.head
        lst[tail] = head
    return lst

def get_tree_fast(scores, nWords):
    heads = np.zeros(nWords)
    
    for iSrc in range(nWords):
        best_score = -99999
        best_index = 0
        for iDst in range(nWords):
            if iDst != iSrc:
                if scores[iSrc][iDst].value() > best_score:
                    best_score = scores[iSrc][iDst].value()
                    best_index = int(iDst)
        heads[iSrc] = best_index
    return heads

def get_tree_score(tree, scores):
    score = 0
    for index in tree:
        arc = tree[index]
        tail = arc.tail
        head = arc.head
        s = scores[tail][head].value()
        score += s
    return score
    
def get_sort_key(item):
    return item.weight

def valid(arc, tree):
    #just one head
    for sa in tree:
        if sa.tail == arc.tail:
            return False
    stack = [arc.head]
    pos = 0
    used = [False] * len(tree)
    while pos < len(stack):
        for zz in range(len(tree)):
            if tree[zz].tail == stack[pos] and not used[zz]:
                used[zz] = True
                stack.append(tree[zz].head)
                if tree[zz].head == arc.tail:
                    return False
        pos += 1
        #print pos,len(stack)
    return True

def greedy_tree(arcs):
    arcs = sorted(arcs, key=get_sort_key, reverse=True)
    #print arcs
    final_tree = []
    for index in range(len(arcs)):
        if valid(arcs[index], final_tree):
            final_tree.append(arcs[index])
            #print arcs[index]
    return final_tree
    
def get_tree(scores, nWords):
    
    #normalize
    norm_score=[]
    for iSrc in range(0, nWords):
        row=np.zeros(nWords)
        for iDst in range(0, nWords):
            row[iDst]=scores[iSrc][iDst].value()
        e_x = np.exp(row - np.max(row))
        norm_score.append(e_x / e_x.sum())
    
            
    g = []
    for iSrc in range(1, nWords):
        for iDst in range(1, nWords):
            if iDst != iSrc:
                a = Arc(iSrc, norm_score[iSrc][iDst], iDst)
                g.append(a)
    tree = greedy_tree(g)
    best_tree = make_ordered_list(tree, nWords)
    return best_tree

def get_tree_edmonds(scores_act, nWords):
    scores=np.zeros((nWords, nWords))
    for iSrc in range (nWords):
        for iDst in range (nWords):
            if iSrc!=iDst:
                scores[iSrc][iDst]=scores_act[iSrc][iDst].value()
    #find the root
    root = 1
    root_score = 0
    for iSrc in range (1, nWords):
        if scores[iSrc][0] > root_score:
            root_score = scores[iSrc][0]
            root = iSrc
    
    g = []
    for iSrc in range (1, nWords):
        for iDst in range(1, nWords):
            if iSrc != iDst:
                arc = Arc(iSrc, -scores[iSrc][iDst], iDst)
                g.append(arc)
                
    tree = spanning_arborescence(g, root)
    
    result=make_ordered_list_arcs(tree, nWords)
    return result
    

#prices,names = _input(filename)
#g = _load(prices,prices)
#h = mst(int(root),g)
#for s in h:
#    for t in h[s]:
#        print "%d-%d" % (s,t)