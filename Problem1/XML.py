def get_attr_number(node):
    #transform xml to byte
    tree = etree.tostring(node)
    #transfom byte to string
    tr = str(tree)
    #each of the element of score have an equal
    tot = tr.count('=')
    return tot




maxdepth = 0
def depth(elem, level):
    global maxdepth
    if len(elem) > 0: #if it is not a leaf
        level += 1
        maxdepth = max(level + 1, maxdepth)
    for i in elem:
        # go trough all child
        depth(i, level)

