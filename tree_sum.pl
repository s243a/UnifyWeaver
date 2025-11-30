% tree_sum(Tree, Sum) :- Sum is the sum of all nodes in the tree.
% Tree representation: [Value, LeftSubtree, RightSubtree]
tree_sum([], 0).
tree_sum([Value, Left, Right], Sum) :-
    tree_sum(Left, LeftSum),
    tree_sum(Right, RightSum),
    Sum is Value + LeftSum + RightSum.
