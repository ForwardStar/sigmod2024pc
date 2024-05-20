## SIGMOD 2024 Programming Contest
This is a solution for the [SIGMOD 2024 programming contest](http://sigmodcontest2024.eastus.cloudapp.azure.com/index.shtml). Given a dataset D and a query set Q, this contest aims to find the 100 approximated nearest neighbours (ANN) in D of the queried vectors in Q within 20 minutes. Our solution achieved a recall of 0.7177 and ranked #15.

To compile the codes, simply run:
```sh
make all
```

Our main solution is implemented in ``hnsw.cpp``, which combines the graph partitioning algorithm with the HNSW algorithm. You can execute it by:
```sh
./hnsw
```

It would read ``dummy-data.bin`` as the dataset D and ``dummy-queries.bin`` as the query set Q, and then output the ANN results to ``output.bin``.

To generate the standard results of the dummy dataset, run the ``baseline`` program and the results would be written into ``output.bin``. Rename ``output.bin`` to ``output_std.bin``.

Then you can run the ``comparator`` program to compute the recall on the dummy dataset.

Besides, there is a sampling-based algorithm implemented in ``sampling.cpp``. However, it yields a poor performance on large datasets.
