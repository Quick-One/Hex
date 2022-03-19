class UnionFind:
    """
    Notes:
        Unionfind data structure specialized for finding hex connections.

    Attributes:
        parent (dict): Each group parent
        rank (dict): Each group rank
        groups (dict): Stores the groups and chain of cells
        ignored (list): The neighborhood of board edges has to be ignored
    """

    def __init__(self) -> None:
        """
        Initialize parent and rank as empty dictionaries, we will
        lazily add items as necessary.
        """
        self.parent = {}
        self.rank = {}
        self.groups = {}

    def join(self, x, y) -> bool:
        """
        Merge the groups of x and y if they were not already,
        return False if they were already merged, true otherwise

        Args:
            x (tuple): game board cell
            y (tuple): game board cell

        """
        # Find root/parent node of x and y
        root_x = self.find(x)
        root_y = self.find(y)

        # Return false if x and y are in the same group
        if root_x == root_y:
            return False

        # Add the tree with the smaller rank to the one with the higher rank and delete the smaller tree.
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y

            self.groups[root_y].extend(self.groups[root_x])
            del self.groups[root_x]

        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x

            self.groups[root_x].extend(self.groups[root_y])
            del self.groups[root_y]

        # If the trees have same rank, add one to the other and increase it's rank.
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += 1

            self.groups[root_y].extend(self.groups[root_x])
            del self.groups[root_x]

        return True

    def find(self, x):
        """
        Get the representative element associated with the set in
        which element x resides. Uses grandparent compression to compression
        the tree on each find operation so that future find operations are faster.
        Args:
            x (tuple): game board cell
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.groups[x] = [x]

        # If root node then return node itself.
        parent_x = self.parent[x]
        if x == parent_x:
            return x

        parent_parent_x = self.parent[parent_x]
        if parent_parent_x == parent_x:
            return parent_x

        # Compress treee by bringing passed node above by making it's parent, it's parent's parent
        self.parent[x] = parent_parent_x

        return self.find(parent_parent_x)

    def connected(self, x, y) -> bool:
        """
        Check if two elements are in the same group.

        Args:
            x (tuple): game board cell
            y (tuple): game board cell
        """
        return self.find(x) == self.find(y)

    def get_groups(self) -> dict:
        """

        Returns:
            Groups
        """
        return self.groups
