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


float compare_with_id(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0;
  // Skip the first 2 dimensions
  for (size_t i = 2; i < a.size(); ++i) {
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
  if (argc > 1) {
    source_path = string(argv[1]);
  }

  uint32_t num_data_dimensions = 102;
  float sample_proportion = 0.001;

  // Read data points
  vector <vector<float>> nodes;
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
  uint32_t sn = uint32_t(sample_proportion * n);

  cout<<"# data points:  " << n<<"\n";
  cout<<"# data point dim:  " << d<<"\n";
  cout<<"# queries:      " << nq<<"\n";

  /** A basic method to compute the KNN results using sampling  **/
  const int K = 100;    // To find 100-NN

  for(uint i = 0; i < nq; i++){
    uint32_t query_type = queries[i][0];
    int32_t v = queries[i][1];
    float l = queries[i][2];
    float r = queries[i][3];
    vector<float> query_vec;

    // first push_back 2 zeros for aligning with dataset
    query_vec.push_back(0);
    query_vec.push_back(0);
    for(uint j = 4; j < queries[i].size(); j++)
      query_vec.push_back(queries[i][j]);

    vector<uint32_t> knn; // candidate knn

    // Handling 4 types of queries
    if(query_type == 0){  // only ANN
        for(uint32_t j = 0; j < sn; j++){
            knn.push_back(j);
        }
    }
    else if(query_type == 1){ // equal + ANN
        for(uint32_t j = 0; j < sn; j++){
            if(nodes[j][0] == v){
                knn.push_back(j);
            }
        }
    }
    else if(query_type == 2){ // range + ANN
      for(uint32_t j = 0; j < sn; j++){
        if(nodes[j][1] >= l && nodes[j][1] <= r)
          knn.push_back(j);
      }
    }
    else if(query_type == 3){ // equal + range + ANN
      for(uint32_t j = 0; j < sn; j++){
        if(nodes[j][0] == v && nodes[j][1] >= l && nodes[j][1] <= r)
          knn.push_back(j);
      }
    }

    // If the number of knn in the sampled data is less than K, then fill the rest with the last few nodes
    if(knn.size() < K){
      uint32_t s = 1;
      while(knn.size() < K) {
        knn.push_back(n - s);
        s = s + 1;
      }
    }

    // build another vec to store the distance between knn[i] and query_vec
    vector<float> dists;
    dists.resize(knn.size());
    for(uint32_t j = 0; j < knn.size(); j++)
      dists[j] = compare_with_id(nodes[knn[j]], query_vec);

    vector<uint32_t > ids;
    ids.resize(knn.size());
    std::iota(ids.begin(), ids.end(), 0);
    // sort ids based on dists
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b){
        return dists[a] < dists[b];
    });
    vector<uint32_t> knn_sorted;
    knn_sorted.resize(K);
    for(uint32_t j = 0; j < K; j++){
      knn_sorted[j] = knn[ids[j]];
    }
    knn_results.push_back(knn_sorted);
  }

  // save the results
  SaveKNN(knn_results, knn_save_path);
  return 0;
}