/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include "io.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;


std::vector<int> cut;
vector <vector<float>> nodes;


bool cmp(const std::vector<float>& i, const std::vector<float>& j) {
  return i[0] < j[0];
}


bool cmp1(const std::vector<float>& i, const std::vector<float>& j) {
  return i[1] < j[1];
}


int binsearch(int v) {
  int l = 0;
  int r = cut.size() - 2;
  while (l < r) {
    int mid = l + r >> 1;
    if (int(nodes[cut[mid]][0]) < v) {
      l = mid + 1;
    }
    else {
      r = mid;
    }
  }
  return l;
}


int binsearch(int li, int ri, float v) {
  int l = li;
  int r = ri;
  if (nodes[r][1] < v) {
    return -1;
  }
  while (l < r) {
    int mid = l + r >> 1;
    if (nodes[mid][1] < v) {
      l = mid + 1;
    }
    else {
      r = mid;
    }
  }
  return l;
}


float compare_with_id(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0;
  // Skip the first 2 dimensions
  // #pragma omp parallel for
  for (size_t i = 2; i < a.size() - 1; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
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
  uint32_t sn = std::min(int(n), 15000);

  cout<<"# data points:  " << n<<"\n";
  cout<<"# data point dim:  " << d<<"\n";
  cout<<"# queries:      " << nq<<"\n";
  cout<<"Sample size:   " << sn<<"\n";

  /** A basic method to compute the KNN results using sampling  **/
  const int K = 100;    // To find 100-NN
  int mismatched_nums = 0;

  for (int i = 0; i < nodes.size(); i++) {
    nodes[i].push_back(i);
  }
  std::sort(nodes.begin(), nodes.end(), cmp);
  int last = 0;
  cut.push_back(0);
  for (int i = 1; i < nodes.size(); i++) {
    if (nodes[i][0] != nodes[i - 1][0]) {
      std::sort(nodes.begin() + last, nodes.begin() + i, cmp1);
      last = i;
      cut.push_back(last);
    }
  }
  std::sort(nodes.begin() + last, nodes.end(), cmp1);
  cut.push_back(nodes.size());
  knn_results.resize(nq);

  #pragma omp parallel for
  for(int i = 0; i < nq; i++){
    // if ((i + 1) % 10000 == 0) {
    //   std::cout << i + 1 << std::endl;
    // }
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

    // Handling 4 types of queries
    if(query_type == 0){  // only ANN
        for(uint32_t j = 0; j < sn; j++){
            knn.push_back(j);
        }
    }
    else if(query_type == 1){ // equal + ANN
        int idx = binsearch(v);
        for(uint32_t j = cut[idx]; j < cut[idx + 1]; j++){
            if(nodes[j][0] == v){
                knn.push_back(j);
            }
            else {
              break;
            }
            if (knn.size() >= sn) {
              break;
            }
        }
    }
    else if(query_type == 2){ // range + ANN
        for (int i = 0; i < cut.size() - 1; i++) {
          int li = cut[i], ri = cut[i + 1] - 1;
          int idx = binsearch(li, ri, l);
          if (idx == -1) {
            continue;
          }
          else {
            for (int j = idx; j <= ri; j++) {
              if (nodes[j][1] >= l && nodes[j][1] <= r) {
                knn.push_back(j);
              }
              else {
                break;
              }
              if (knn.size() >= sn) {
                break;
              }
            }
          }
          if (knn.size() >= sn) {
            break;
          }
        }
    }
    else if(query_type == 3){ // equal + range + ANN
      int idx = binsearch(v);
      int li = cut[idx], ri = cut[idx + 1] - 1;
      int idx1 = binsearch(li, ri, l);
      if (idx1 != -1) {
        for (int j = idx1; j <= ri; j++) {
          if (nodes[j][1] >= l && nodes[j][1] <= r) {
            knn.push_back(j);
          }
          else {
            break;
          }
          if (knn.size() >= sn) {
            break;
          }
        }
      }
    }

    // If the number of knn in the sampled data is less than K, then fill the rest with the last few nodes
    if(knn.size() < K){
      // std::cout << knn.size() << std::endl;
      uint32_t s = 1;
      mismatched_nums += K - knn.size();
      while(knn.size() < K) {
        knn.push_back(n);
        s = s + 1;
      }
    }

    // build another vec to store the distance between knn[i] and query_vec
    auto cmp2 = [](std::pair<uint32_t, float> i, std::pair<uint32_t, float> j) {
        return i.second < j.second;
    };
    std::priority_queue<std::pair<uint32_t, float>, std::vector<std::pair<int, float>>, decltype(cmp2)> dists(cmp2);
    for(uint32_t j = 0; j < knn.size(); j++) {
      if (knn[j] == n) {
        dists.push(std::make_pair(n, 2147483647));
      }
      else {
        dists.push(std::make_pair(knn[j], compare_with_id(nodes[knn[j]], query_vec)));
      }
      if (dists.size() > K) {
        dists.pop();
      }
    }

    while (!dists.empty()) {
      auto u = dists.top();
      if (u.first == n) {
        knn_results[i].push_back(n);
      }
      else {
        knn_results[i].push_back(uint32_t(nodes[u.first][nodes[u.first].size() - 1]));
      }
      dists.pop();
    }
  }

  std::cout << "Mismatched nums: " << mismatched_nums << std::endl;

  // save the results
  SaveKNN(knn_results, knn_save_path);
  return 0;
}