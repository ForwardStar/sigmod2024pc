/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <omp.h>
#include <ctime>
#include <unordered_set>
#include "io.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

vector<vector<float>> nodes;
vector<vector<uint32_t>> edges;

const int M = 30;
vector<vector<bool>> visited;

bool cmp_label(const std::vector<float>& i, const std::vector<float>& j) {
  return i[0] < j[0];
}

bool cmp_time(const std::vector<float>& i, const std::vector<float>& j) {
  return i[1] < j[1];
}

float compare_with_id(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0;
  // Skip the first 2 dimensions
  // #pragma omp parallel for reduction (+:sum)
  for (size_t i = 2; i < a.size() - 1; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

void ann_search(vector<float>& q, int s, int k, vector<uint32_t>& ann, int32_t vc=-1, float ts=-1, float te=-1, int limit=1e7) {
  int tid = omp_get_thread_num();
  vector<uint32_t> visited_list;
  visited[tid][s] = true;
  visited_list.push_back(s);
  auto cmp1 = [](std::pair<uint32_t, float> i, std::pair<uint32_t, float> j) {
      return i.second > j.second;
  };
  auto cmp2 = [](std::pair<uint32_t, float> i, std::pair<uint32_t, float> j) {
      return i.second < j.second;
  };
  std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, decltype(cmp1)> Q1(cmp1);
  std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<uint32_t, float>>, decltype(cmp2)> Q2(cmp2);
  Q1.push(std::make_pair(s, compare_with_id(nodes[s], q)));
  Q2.push(std::make_pair(s, compare_with_id(nodes[s], q)));
  int cur = 0;
  while (!Q1.empty()) {
    cur++;
    int v = Q1.top().first, u = Q2.top().first;
    if (Q1.top().second > Q2.top().second || cur > limit) {
      break;
    }
    Q1.pop();
    for (auto w : edges[v]) {
      if (!visited[tid][w]) {
        visited[tid][w] = true;
        visited_list.push_back(w);
        float d1 = compare_with_id(q, nodes[w]), d2 = compare_with_id(q, nodes[u]);
        if (Q2.size() < k || d1 < d2) {
          Q1.push(std::make_pair(w, d1));
          if (vc == -1 || nodes[w][0] == vc) {
            if (ts == -1 || te == -1 || (nodes[w][1] >= ts && nodes[w][1] <= te)) {
              Q2.push(std::make_pair(w, d1));
              if (Q2.size() > k) {
                Q2.pop();
              }
            }
          }
        }
      }
    }
  }
  for (auto w : visited_list) {
    visited[tid][w] = false;
  }
  while (!Q2.empty()) {
    ann.push_back(Q2.top().first);
    Q2.pop();
  }
  std::reverse(ann.begin(), ann.end());
}

vector<uint32_t> prune(int now, vector<uint32_t>& ann) {
  vector<uint32_t> neighbours;
  for (int i = 0; i < ann.size(); i++) {
    int v = ann[i];
    bool not_dominated = true;
    for (int u : neighbours) {
      if (compare_with_id(nodes[now], nodes[u]) < compare_with_id(nodes[now], nodes[v]) && compare_with_id(nodes[u], nodes[v]) < compare_with_id(nodes[now], nodes[v])) {
        not_dominated = true;
        break;
      }
    }
    if (not_dominated) {
      neighbours.push_back(v);
    }
    if (neighbours.size() >= M) {
      break;
    }
  }
  return neighbours;
}

int get_num_threads(void) {
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    return num_threads;
}

int main(int argc, char **argv) {
  srand(time(0));
  string source_path = "dummy-data.bin";
  string query_path = "dummy-queries.bin";
  string knn_save_path = "output.bin";

  // Also accept other path for source data
  if (argc > 2) {
    source_path = string(argv[1]);
    query_path = string(argv[2]);
  }

  uint32_t num_data_dimensions = 102;

  // Read data points
  ReadBin(source_path, num_data_dimensions, nodes);
  cout<<nodes.size()<<"\n";
  // Read queries
  uint32_t num_query_dimensions = num_data_dimensions + 2;
  vector <vector<float>> queries;
  ReadBin(query_path, num_query_dimensions, queries);

  vector <vector<uint32_t>> knn_results; // for saving knn results

  uint32_t n = nodes.size();
  uint32_t d = nodes[0].size();
  uint32_t nq = queries.size();
  uint32_t block_num = 20;
  uint32_t block_size = n / block_num;
  uint32_t block_k = 10;

  cout<<"# data points:  " << n<<"\n";
  cout<<"# data point dim:  " << d<<"\n";
  cout<<"# queries:      " << nq<<"\n";
  cout<<"# blocks:    " << block_num<<"\n";
  cout<<"Block size:    " << block_size<<"\n";
  cout<<"Block K:    " << block_k<<"\n";

  /** A basic method to compute the KNN results using sampling  **/
  const int K = 100;    // To find 100-NN
  int mismatched_nums = 0;

  edges.resize(nodes.size());
  knn_results.resize(nq);
  for (int i = 0; i < get_num_threads(); i++) {
    visited.push_back(vector<bool>());
    visited[i].resize(nodes.size());
  }
  cout << "Number of threads: " << get_num_threads() << endl;

  cout << "Proprocessing..." << endl;
  for (int i = 0; i < n; i++) {
    nodes[i].push_back(i);
  }
  std::sort(nodes.begin(), nodes.end(), cmp_time);
  // for (int i = 0; i < block_num; i++) {
  //   std::sort(nodes.begin() + i * block_size, nodes.begin() + (i + 1) * block_size, cmp_label);
  // }

  cout << "Constructing the index..." << endl;
  #pragma omp parallel for
  for (int i = 0; i < block_num; i++) {
    for (int k = i * block_size; k < (i + 1) * block_size; k++) {
      vector<uint32_t> ann;
      ann_search(nodes[k], i * block_size, block_k, ann);
      edges[k] = prune(k, ann);
      for (int u : edges[k]) {
        edges[u].push_back(k);
        if (edges[u].size() > M) {
          edges[u] = prune(u, edges[u]);
        }
      }
    }
  }

  cout << "Solving queries..." << endl;
  int cur = 0;
  #pragma omp parallel for
  for(int i = 0; i < nq; i++){
    cur++;
    if (cur % 100000 == 0) {
      cout << cur << endl;
    }
    uint32_t query_type = queries[i][0];
    int32_t v = queries[i][1];
    float l = queries[i][2];
    float r = queries[i][3];
    vector<float> query_vec;

    // first push_back 2 zeros for aligning with dataset
    query_vec.push_back(0);
    query_vec.push_back(0);
    for(int j = 4; j < queries[i].size(); j++)
      query_vec.push_back(queries[i][j]);

    int li = 0, ri = block_num;
    if (query_type == 2 || query_type == 3) {
      for (int j = 0; j < block_num; j++) {
        if (nodes[j * block_size][1] <= l) {
          li = j;
        }
        if (nodes[j * block_size][1] > r) {
          ri = j;
          break;
        }
      }
    }
    
    for (int j = li; j < ri; j++) {
      ann_search(query_vec, j * block_size, 2 * K / (ri - li), knn_results[i], v, l, r, 30 * K / (ri - li));
    }
  }

  cout << "Normalizing results..." << endl;
  #pragma omp parallel for
  for (int i = 0; i < nq; i++) {
    while (knn_results[i].size() < K) {
      knn_results[i].push_back(knn_results[i][0]);
    } 
    vector<float> query_vec;

    // first push_back 2 zeros for aligning with dataset
    query_vec.push_back(0);
    query_vec.push_back(0);
    for(int j = 4; j < queries[i].size(); j++)
      query_vec.push_back(queries[i][j]);

    vector<float> dists;
    dists.resize(knn_results[i].size());
    for(uint32_t j = 0; j < knn_results[i].size(); j++) {
      if (knn_results[i][j] != n) {
        dists[j] = compare_with_id(nodes[knn_results[i][j]], query_vec);
      }
      else {
        dists[j] = 2147483647;
      }
    }
    vector<uint32_t > ids;
    ids.resize(knn_results[i].size());
    std::iota(ids.begin(), ids.end(), 0);
    // sort ids based on dists
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b){
        return dists[a] < dists[b];
    });

    vector<uint32_t> knn_sorted;
    knn_sorted.resize(K);
    for(uint32_t j = 0; j < K; j++){
      uint32_t id = knn_results[i][ids[j]];
      knn_sorted[j] = nodes[id][nodes[id].size() - 1];
    }
    knn_results[i] = knn_sorted;
  }

  std::cout << "Mismatched nums: " << mismatched_nums << std::endl;

  // save the results
  SaveKNN(knn_results, knn_save_path);
  return 0;
}