import graph_tool.all as gt
import multiprocessing as mp
import os.path as osp
import sys
import gzip

class Visitor(gt.BFSVisitor):

    def __init__(self, pred, dist, target, source_node, max_depth=4):
        self.pred = pred
        self.dist = dist
        self.target = target
        self.max_depth = max_depth
        self.source_node = source_node

    def examine_vertex(self, u):
        global dists_dict #reachable for workers
        if u == self.target:
            distance = self.dist[u]
            if self.source_node in dists_dict.keys():
                dists_dict[self.source_node][self.target] = 0 if distance == 0 else 1/distance
            else:
                #create nested dict
                dists_dict[self.source_node] = {}
                dists_dict[self.source_node][self.target] = 0 if distance == 0 else 1/distance
            raise gt.StopSearch() #exit

        elif self.dist[u] == self.max_depth:
            if self.source_node in dists_dict.keys():
                dists_dict[self.source_node][self.target] = 0
            else:
                #create nested dict
                dists_dict[self.source_node] = {}
                dists_dict[self.source_node][self.target] = 0
            raise gt.StopSearch() #exit

    def tree_edge(self, e):
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = self.dist[e.source()] + 1

def shortest_path_length_gt(G, source_nodes):
    """
    Calculate shortest path distance to every sampled anchor node and normalize by 1/distance. No path = 0
    """
    global dists_dict
    print('workers spawned!')
    for source_node in source_nodes:
        for target_node in G.get_vertices():
            dist = G.new_vertex_property("int")
            pred = G.new_vertex_property("int64_t")
            gt.bfs_search(G, G.vertex(source_node), Visitor(pred, dist, target_node, source_node))
        if len(dists_dict.keys()) % 20 == 0:
            save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'save', f'products_dist_lisa{len(dists_dict.keys())}.pkl')
            with gzip.open(save_path, "wb") as f:
                pickled = pickle.dumps(dists_dict)
                optimized_pickle = pickletools.optimize(pickled)
                f.write(optimized_pickle)
            print(f'{len(dists_dict.keys())} embeddings saved!')

def shortest_path_gt(G, num_workers=16):
    """
    Distribute shortest path calculation jobs to async workers, merge dicts and return results
    """
    try:
        global dists_dict
        dists_dict = {}
        nodes = G.get_vertices()[2000000:]
        print('spawning workers...')
        pool = mp.Pool(processes=num_workers)
        jobs = [pool.apply_async(shortest_path_length_gt,
                args=(G, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))])) for i in range(num_workers)]
        output = [job.get() for job in jobs]
        pool.close()
        pool.join()
        print('finished!')
        save_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'save', f'products_dist_lisa_full.pkl')
        with gzip.open(save_path, "wb") as f:
            pickled = pickle.dumps(dists_dict)
            optimized_pickle = pickletools.optimize(pickled)
            f.write(optimized_pickle)
        print('all approximated shortest paths saved!')
    
    except KeyboardInterrupt:
        print('terminating workers...')
        pool.terminate()
        pool.join()
        print('workers terminated!')
        sys.exit(1)


path = osp.join(osp.dirname(osp.realpath(__file__)), 'products.graphml')
#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'products.graphml')
G = gt.load_graph(path)
print('data loaded in!')
shortest_path_gt(G)
print('shortest paths derived!')