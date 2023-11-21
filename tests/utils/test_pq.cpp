// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "partition_and_pq.h"
#include "util.h"
#include <index.h>
#include <math_utils.h>

// DEPRECATED: NEED TO REPROGRAM

int main(int argc, char **argv) {
    if (argc != 6) {
        diskann::cout << argv[0]
                      << " format: train_set "
                         "num_clusters_per_chunk number_chunks "
                         "max_reps prefix_for_working_directory "
                      << std::endl;
        exit(-1);
    }

    size_t num_centers = (size_t)strtol(argv[2], NULL, 10);
    size_t num_chunks = (size_t)strtol(argv[3], NULL, 10);
    size_t max_reps = (size_t)strtol(argv[4], NULL, 10);

    generate_pq_pivots(argv[1], num_centers, num_chunks, max_reps, argv[5]);
    return 0;
}
