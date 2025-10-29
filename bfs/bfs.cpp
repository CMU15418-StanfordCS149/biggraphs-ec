#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    #pragma omp parallel 
    {
        std::vector<int> local_buf;
        // 估算 vector 的大小，减少 realloc 开销
        int nthreads = omp_get_num_threads();
        int approx = (frontier->count + nthreads - 1) / nthreads;
        local_buf.reserve(std::min(approx * 2, 1024));

        // 本次函数里，所有的 new_distance 都是相同的，把计算放到外面，以减少内部循环计算开销
        int new_distance = distances[frontier->vertices[0]] + 1;

        // 遍历 frontier 的每个节点
        #pragma omp for schedule(dynamic, 64)
        for (int i=0; i<frontier->count; i++) {

            // 获取当前顶点编号
            int node = frontier->vertices[i];

            // 获取当前节点出边的起始编号和终止编号
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // 遍历当前顶点的所有出边邻居节点
            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];

                // 在进入原子比较(临界区)之前，先检查一下该节点是否已经被访问过，这样可以减少不必要的同步开销
                if(distances[outgoing] == NOT_VISITED_MARKER) {
                    if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, new_distance)) {
                        // 把节点加入到本地缓冲区，避免同步开销
                        local_buf.push_back(outgoing);
                    }
                }
            }
        }

        // 把本地缓冲区内的节点放入全局的 new_frontier 数组，一次放入一整个 buffer，减少同步开销
        // 同步开销：O(insertions) --> O(threads) 从节点插入次数将为线程数
        if (!local_buf.empty()) {
            int offset = __sync_fetch_and_add(&new_frontier->count, (int)local_buf.size());
            for (int k = 0; k < local_buf.size(); k++) {
                new_frontier->vertices[offset + k] = local_buf[k];
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
// 参数1：graph：图结构体指针
// 参数2：sol：存储结果的结构体指针
// 这个函数要计算图中所有节点到根节点的距离，存储在 sol.distances 中
void bfs_top_down(Graph graph, solution* sol) {

    // 顶点集合1
    vertex_set list1;
    // 顶点集合2
    vertex_set list2;
    // 使用 graph 初始化这两个顶点集合(其实就是分配空间并且初始化空间为空)
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // 初始化所有节点的距离为 -1， 表示“尚未访问”
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // 节点 0 作为根节点，根节点到根节点的距离为 0
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        // 清空 new_frontier，为下一轮做准备
        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // 交换 frontier 和 new_frontier 指针
        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// 对于图中的每个顶点 v：
//     如果 v 尚未被访问 并且
//        v 与边界上的顶点 u 共享一条入边：
//           将顶点 v 添加到边界；
void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances, bool *visited)
{
    // 获取当前前沿到根节点的距离
    int frontier_distance = distances[frontier->vertices[0]];
    // 提前计算，避免循环中的计算开销
    int new_distance = frontier_distance + 1; 

    #pragma omp parallel 
    {
        // 本地缓冲，减少同步开销
        std::vector<int> local_buf;
        // 估算 vector 的大小，减少 realloc 开销
        int nthreads = omp_get_num_threads();
        int approx = (frontier->count + nthreads - 1) / nthreads;
        local_buf.reserve(std::min(approx * 2, 1024));

        // 对于图中的每个顶点 v：
        #pragma omp for schedule(dynamic, 1000)
        for (int node = 0; node < g->num_nodes; node++) {
            // 如果 v 尚未被访问 并且 ... 
            if(!visited[node]) {
                // ... 并且 v 与边界上的顶点 u 共享一条入边：
                // 遍历顶点 v 的所有入边邻居节点
                int start_edge = g->incoming_starts[node];
                int end_edge = (node == g->num_nodes - 1)
                                ? g->num_edges
                                : g->incoming_starts[node + 1];
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
                    int incoming = g->incoming_edges[neighbor];
                    // 检查该入边邻居节点是否已被访问过，若被访问过，说明它在 frontier 上
                    // （只有 frontier 节点会和 unvisited 节点接触）
                    if(visited[incoming]) {
                        local_buf.push_back(node);
                        distances[node] = new_distance;
                        // 这里不能设置 visited 数组，否则会破坏 "被访问过的节点都在 frontier 上" 的假设
                        // visited[node] = true; 
                        break;
                    }
                }
            }
        }

        // 把本地缓冲区内的节点放入全局的 new_frontier 数组，一次放入一整个 buffer，减少同步开销
        // 同步开销：O(insertions) --> O(threads) 从节点插入次数将为线程数
        if (!local_buf.empty()) {
            int offset = __sync_fetch_and_add(&new_frontier->count, (int)local_buf.size());
            // 使用 memcpy 设置 new_frontier->vertices 数组，比手动for循环更快
            memcpy(&new_frontier->vertices[offset], &local_buf[0], sizeof(int) * local_buf.size());
            // 设置 visited 数组
            for (int k = 0; k < local_buf.size(); k++) {
                int node = local_buf[k];
                visited[node] = true;
            }
        }
    }
}

// 对于图中的每个顶点 v：
//     如果 v 尚未被访问 并且
//        v 与边界上的顶点 u 共享一条入边：
//           将顶点 v 添加到边界；
void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    // 顶点集合1
    vertex_set list1;
    // 顶点集合2
    vertex_set list2;
    // 使用 graph 初始化这两个顶点集合(其实就是分配空间并且初始化空间为空)
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // 记录节点是否被访问过的数组，占用内存带宽比 distances 更小，节省内存带宽
    bool* visited = (bool *) calloc(graph->num_nodes, sizeof(bool));
    // unvisted 数组无需初始化，calloc 已经将所有字节置零，表示 false

    // sol->distances 数组也无需初始化，因为在 bottom_up_step 中没有读取操作，只有写入操作

    // 节点 0 作为根节点，根节点到根节点的距离为 0
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    visited[ROOT_NODE_ID] = true;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        // 清空 new_frontier，为下一轮做准备
        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, visited);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // 交换 frontier 和 new_frontier 指针
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // 释放 visited 数组
    free(visited);
    visited = nullptr;
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
