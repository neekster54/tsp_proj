"""Double-tree 2-approximation."""

from tsp.graph import euler_circuit, prim_mst, shortcut


def double_tree(d, start=0):
    mst = prim_mst(d)
    doubled = mst + mst
    return shortcut(euler_circuit(len(d), doubled, start))
    