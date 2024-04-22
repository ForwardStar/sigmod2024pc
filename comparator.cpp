/**
 *  Example code using sampling to find KNN.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_set>
#include "io.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char **argv) {
  string output_std_path = "output_std.bin";
  string output_path = "output.bin";

  int num_data_dimensions = 100;
  vector<vector<uint32_t>> output_std, output;
  ReadBinResults(output_std_path, num_data_dimensions, output_std);
  ReadBinResults(output_path, num_data_dimensions, output);

  int n = output_std.size();
  float avg_recall = 0;
  for (int i = 0; i < n; i++) {
    float recall = 0;
    std::unordered_set<int> occurred_ids;
    for (int j = 0; j < num_data_dimensions; j++) {
        occurred_ids.insert(output_std[i][j]);
    }
    for (int j = 0; j < num_data_dimensions; j++) {
        if (occurred_ids.find(output[i][j]) != occurred_ids.end()) {
            recall = recall + 1;
        }
    }
    avg_recall += recall / num_data_dimensions;
  }
  avg_recall /= n;
  std::cout << "Average recall: " << avg_recall << std::endl;
  return 0;
}