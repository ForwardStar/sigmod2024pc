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
#include "io.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

vector<vector<float>> nodes;
vector<vector<uint32_t>> edges;

const int M = 5;
vector<bool> visited;
vector<uint32_t> visited_list;

float compare_with_id(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0;
  // Skip the first 2 dimensions
  #pragma omp parallel for
  for (size_t i = 2; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

void ann_search(vector<float>& q, int s, int k, vector<uint32_t>& ann) {
  visited_list.clear();
  visited[s] = true;
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
  while (!Q1.empty()) {
    int v = Q1.top().first, u = Q2.top().first;
    if (Q1.top().second > Q2.top().second) {
      break;
    }
    Q1.pop();
    for (auto w : edges[v]) {
      if (!visited[w]) {
        visited[w] = true;
        visited_list.push_back(w);
        float d1 = compare_with_id(q, nodes[w]), d2 = compare_with_id(q, nodes[u]);
        if (Q2.size() < k || d1 < d2) {
          Q1.push(std::make_pair(w, d1));
          Q2.push(std::make_pair(w, d1));
          if (Q2.size() > k) {
            Q2.pop();
          }
        }
      }
    }
  }
  for (auto u : visited_list) {
    visited[u] = false;
  }
  while (!Q2.empty()) {
    ann.push_back(Q2.top().first);
    Q2.pop();
  }
}

vector<uint32_t> prune(int now, vector<uint32_t>& ann) {
  vector<uint32_t> neighbours;
  for (int i = ann.size() - 1; i >= 0; i--) {
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

int main(int argc, char **argv) {
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

  cout<<"# data points:  " << n<<"\n";
  cout<<"# data point dim:  " << d<<"\n";
  cout<<"# queries:      " << nq<<"\n";

  /** A basic method to compute the KNN results using sampling  **/
  const int K = 100;    // To find 100-NN
  int mismatched_nums = 0;

  edges.resize(nodes.size());
  visited.resize(nodes.size());
  cout << "Constructing the index..." << endl;
  for (int i = 0; i < std::min(400000, int(nodes.size())); i++) {
    if ((i + 1) % 100000 == 0) {
      cout << (i + 1) << endl;
    }
    vector<uint32_t> ann;
    ann_search(nodes[i], 0, K, ann);
    edges[i] = prune(i, ann);
    for (int u : edges[i]) {
      edges[u].push_back(i);
      if (edges[u].size() > M) {
        edges[u] = prune(u, edges[u]);
      }
    }
  }

  cout << "Solving queries..." << endl;
  #pragma omp parallel for
  for(int i = 0; i < nq; i++){
    if ((i + 1) % 10000 == 0) {
      std::cout << i + 1 << std::endl;
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

    vector<uint32_t> knn; // candidate knn
    ann_search(query_vec, 1, K, knn);
    knn_results.push_back(knn);
  }

  std::cout << "Mismatched nums: " << mismatched_nums << std::endl;

  // save the results
  SaveKNN(knn_results, knn_save_path);
  return 0;
}