import numpy as np

from env.flatland.Flatland import get_new_position


class RailGraph():
    def __init__(self):
        pass

    def reset(self, env):
        self.env = env
        #  self._build()
        #  self._recalc_weights()
        #  self.any_deadlocked = False

        self.env.distance_map.reset(self.env.agents, self.env.rail)

    def deadlock_agent(self, handle):
        return

        h, w = self.env.agents[handle].position
        for d in range(4):
            if (h, w, d) in self.nodes:
                node_i = self.nodes_dict[(h, w, d)]
                if min(np.min(self.amatrix[node_i]), np.min(self.amatrix[:, node_i])) != np.inf:
                    self.any_deadlocked = True
                self.amatrix[node_i, :] = np.inf
                self.amatrix[:, node_i] = np.inf

        for edge in self.cell_to_edge[h][w]:
            if self.amatrix[edge] != 0:
                self.any_deadlocked = True
        self.amatrix[edge] = np.inf


    def update(self):
        return
        if self.any_deadlocked:
            self._recalc_weights()
            self.any_deadlocked = False


    def dist_to_target(self, handle, h, w, d):
        return self.env.distance_map.get()[handle, h, w, d]
        i = self.target_i[self.env.agents[handle].target]
        return self.dtarget[i, h, w, d]

    def _build(self):
        self.nodes = set(agent.target for agent in self.env.agents)
        height, width = self.env.height, self.env.width
        self.valid_pos = list()

        for h in range(height):
            for w in range(width):
                pos = (h, w)
                transition_bit = bin(self.env.rail.get_full_transitions(*pos))
                total_transitions = transition_bit.count("1")
                if total_transitions > 2:
                    self.nodes.add(pos)
                if total_transitions > 0:
                    self.valid_pos.append((h, w))

        n_nodes = set()
        for h, w in self.nodes:
            for d in range(4):
                cell_transitions = self.env.rail.get_transitions(h, w, d)
                if np.any(cell_transitions):
                    n_nodes.add((h, w, d))

        self.nodes = n_nodes
    
        self.dist_to_node = -np.ones((height, width, 4, 4))
        self.next_node = [[[[None for _ in range(4)] for _ in range(4)] for _ in range(width)] for _ in range(height)]
        self.dfs_used = np.zeros((height, width, 4))
        for h in range(height):
            for w in range(width):
                for d in range(4):
                    if not self.dfs_used[h, w, d]:
                        self.dfs(h, w, d)


        self.n_nodes = len(self.nodes)
        self.nodes_dict = np.empty((height, width, 4), dtype=np.int)
        for i, (h, w, d) in enumerate(self.nodes):
            self.nodes_dict[h, w, d] = i

        self.cell_to_edge = [[list() for _ in range(width)] for _ in range(height)]

        self.amatrix = np.ones((self.n_nodes, self.n_nodes)) * np.inf
        self.amatrix[np.arange(self.n_nodes), np.arange(self.n_nodes)] = 0
        for i, (h, w, d) in enumerate(self.nodes):
            for dd in range(4):
                nnode = self.next_node[h][w][d][dd]
                if nnode is not None:
                    self.amatrix[i][self.nodes_dict[nnode]] = self.dist_to_node[h, w, d, dd]

                    cell = (h, w, d)
                    nnode_i = self.nodes_dict[nnode]
                    while cell != nnode:
                        possible_transitions = self.env.rail.get_transitions(*cell)
                        for ndir in range(4):
                            if possible_transitions[ndir] and (cell != (h, w, d) or ndir == dd):
                                nh, nw = get_new_position((cell[0], cell[1]), ndir)
                                cell = (nh, nw, ndir)
                                self.cell_to_edge[nh][nw].append((i, nnode_i))
                                break


    def _recalc_weights(self):
        self.weights = np.copy(self.amatrix)
        for k in range(self.n_nodes):
            self.weights = np.minimum(self.weights, self.weights[:, k:k+1] + self.weights[k:k+1, :])
        self._recalc_dists_to_targets()

    def _recalc_dists_to_targets(self):
        targets = list(set(agent.target for agent in self.env.agents))
        height, width = self.env.height, self.env.width
        self.target_i = np.empty((height, width), dtype=np.int)
        for i, (h, w) in enumerate(targets):
            self.target_i[h, w] = i

        self.dtarget = np.ones((len(targets), height, width, 4)) * np.inf
        for h, w in self.valid_pos:
            for d in range(4):
                if (h, w, d) in self.nodes:
                    add = 0
                    node_i = self.nodes_dict[h, w, d]
                else:
                    node_i = None
                    for ddd in range(4):
                        nnode = self.next_node[h][w][d][ddd]
                        if nnode:
                            add = self.dist_to_node[h, w, d, ddd]
                            node_i = self.nodes_dict[nnode]
                if node_i is not None:
                    for i in range(len(targets)):
                        for dd in range(4):
                            if not (targets[i][0], targets[i][1], dd) in self.nodes:
                                continue
                            tnode_i = self.nodes_dict[targets[i][0], targets[i][1], dd]
                            self.dtarget[i, h, w, d] = min(self.dtarget[i, h, w, d], self.weights[node_i][tnode_i] + add)


        
    def dfs(self, h, w, d):
        self.dfs_used[h, w, d] = 1
        possible_transitions = self.env.rail.get_transitions(h, w, d)
        for ndir in range(4):
            if possible_transitions[ndir]:
                nh, nw = get_new_position((h, w), ndir)
                if (nh, nw, ndir) in self.nodes:
                    self.dist_to_node[h, w, d, ndir] = 1
                    self.next_node[h][w][d][ndir] = (nh, nw, ndir)
                else:
                    if not self.dfs_used[nh, nw, ndir]:
                        self.dfs(nh, nw, ndir)
                    for last_dir in range(4):
                        if self.dist_to_node[nh, nw, ndir, last_dir] > -0.5:
                            self.dist_to_node[h, w, d, ndir] = self.dist_to_node[nh, nw, ndir, last_dir] + 1
                            self.next_node[h][w][d][ndir] = self.next_node[nh][nw][ndir][last_dir]


