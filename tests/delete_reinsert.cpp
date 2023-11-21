#include "v2/index_merger.h"
#include "v2/merge_insert.h"

#include <Neighbor_Tag.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <future>
#include <index.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <timer.h>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "partition_and_pq.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// #define UNIT

// #define Merge_Size 30000000
// #define NUM_INSERT_THREADS 4
// #define NUM_SEARCH_THREADS 10
// diskann::Timer globalTimer;

// int begin_time = 0;
#ifndef UNIT
template <typename T, typename TagT>
void search_kernel(T *query, size_t query_num, size_t query_aligned_dim,
                   unsigned *gt_ids, float *gt_dists, size_t gt_dim,
                   const int recall_at, _u64 L,
                   diskann::MergeInsert<T, TagT> &index,
                   tsl::robin_set<TagT> &active_tags) {
    // print basic information
    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS "
              << std::setw(18) << "Mean Latency (ms)" << std::setw(12)
              << "90 Latency" << std::setw(12) << "95 Latency" << std::setw(12)
              << "99 Latency" << std::setw(12) << "99.9 Latency"
              << std::setw(12) << recall_string << std::setw(12)
              << "Mean disk IOs" << std::endl;
    std::cout
        << "==============================================================="
           "==============="
        << std::endl;
    // prep for search
    std::vector<uint32_t> query_result_tags;
    std::vector<float> query_result_dists;
    query_result_dists.resize(recall_at * query_num);
    query_result_tags.resize(recall_at * query_num);

    diskann::QueryStats *stats = new diskann::QueryStats[query_num];
    std::vector<double> latency_stats(query_num, 0);
    auto s = std::chrono::high_resolution_clock::now();
    for (_s64 i = 0; i < (int64_t)query_num; i++) {
        auto qs = std::chrono::high_resolution_clock::now();
        index.search_sync(query + (i * query_aligned_dim), recall_at, L,
                          (query_result_tags.data() + (i * recall_at)),
                          query_result_dists.data() + (i * recall_at),
                          stats + i);
        auto qe = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = qe - qs;
        latency_stats[i] = diff.count() * 1000;
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float)(((double)query_num) / diff.count());
    // compute mean recall, IOs
    float mean_recall = 0.0f;
    mean_recall = diskann::calculate_recall(
        (unsigned)query_num, gt_ids, gt_dists, (unsigned)gt_dim,
        query_result_tags.data(), (unsigned)recall_at, (unsigned)recall_at,
        active_tags);
    float mean_ios = (float)diskann::get_mean_stats(
        stats, query_num,
        [](const diskann::QueryStats &stats) { return stats.n_ios; });
    std::sort(latency_stats.begin(), latency_stats.end());
    std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
              << ((float)std::accumulate(latency_stats.begin(),
                                         latency_stats.end(), 0)) /
                     (float)query_num
              << std::setw(12)
              << (float)latency_stats[(_u64)(0.90 * ((double)query_num))]
              << std::setw(12)
              << (float)latency_stats[(_u64)(0.95 * ((double)query_num))]
              << std::setw(12)
              << (float)latency_stats[(_u64)(0.99 * ((double)query_num))]
              << std::setw(12)
              << (float)latency_stats[(_u64)(0.999 * ((double)query_num))]
              << std::setw(12) << mean_recall << std::setw(12) << mean_ios
              << std::endl;
    delete[] stats;
}
template <typename T, typename TagT>
void merge_kernel(diskann::MergeInsert<T, TagT> &index) {
    index.final_merge();
}

template <typename T, typename TagT>
void insertion_kernel(T *data_load, diskann::MergeInsert<T, TagT> &index,
                      const tsl::robin_set<TagT> &insert_tags,
                      size_t aligned_dim) {
    diskann::Timer timer;
    std::cout << "Begin Insert" << std::endl;
    for (_s64 i = 0; i < (_s64)insert_tags.size(); i++) {
        index.insert(data_load + aligned_dim * insert_tags[i], insert_tags[i]);
    }
    float time_secs = timer.elapsed() / 1.0e6f;
    std::cout << "Inserted " << insert_tags.size() << " points in " << time_secs
              << "s" << std::endl;
}

template <typename T, typename TagT = uint32_t>
void deletion_kernel(diskann::MergeInsert<T, TagT> &index,
                     const tsl::robin_set<TagT> del_tags) {
    diskann::Timer timer;
    std::cout << "Begin Delete" << std::endl;
    for (auto iter : del_tags) {
        index.lazy_delete(iter);
    }
    std::cout << "Deletion time : " << timer.elapsed() / 1000 << " ms"
              << std::endl;
}

template <typename TagT = uint32_t>
tsl::robin_set<TagT> random_tags(size_t num, size_t upper = 1000000) {
    if (num > upper) {
        std::cerr << "The number of changing points should be less than "
                  << upper << std::endl;
        exit(0);
    }
    // FIXME: modify the range of random_tags
    tsl::robin_set<TagT> tags_set;
    while (tags_set.size() < num) {
        tags_set.insert(rand() % (upper - 1) + 1);
    }
    return tags_set;
}

template <typename T, typename TagT = uint32_t>
void run_single_iter(T *data_load, diskann::MergeInsert<T> &merge_insert,
                     size_t num_points, size_t changing_points,
                     size_t aligned_dim) {
    // TODO:get_random_tags
    const auto chg_pt = random_tags<TagT>(changing_points, num_points);
    // TODO:delete
    deletion_kernel<T, TagT>(merge_insert, chg_pt);
    // TODO:insert
    insertion_kernel<T, TagT>(data_load, merge_insert, chg_pt, aligned_dim);
    // TODO:merge
    merge_kernel<T, TagT>(merge_insert);
}

template <typename T, typename TagT = uint32_t>
void DeleteReinsert(const std::string &data_path, const unsigned L_mem,
                    const unsigned R_mem, const float alpha_mem,
                    const unsigned L_disk, const unsigned R_disk,
                    const float alpha_disk, std::string &save_path,
                    const std::string &query_file,
                    const std::string &truthset_file,
                    const std::string &index_file, unsigned iter_times = 1,
                    size_t changing_points = 100, const int recall_at = 10,
                    _u64 Lsearch = 20, const unsigned beam_width = 5) {
    // FIXME:summary parameters
    diskann::Parameters paras;
    paras.Set<unsigned>("L_mem", L_mem);
    paras.Set<unsigned>("R_mem", R_mem);
    paras.Set<float>("alpha_mem", alpha_mem);
    paras.Set<unsigned>("L_disk", L_disk);
    paras.Set<unsigned>("R_disk", R_disk);
    paras.Set<float>("alpha_disk", alpha_disk);
    paras.Set<unsigned>("C", 160);
    paras.Set<unsigned>("beamwidth", beam_width);
    paras.Set<unsigned>("nodes_to_cache", 0);
    paras.Set<unsigned>("num_search_threads", 2);
    // FIXME:load truthset
    unsigned *gt_ids = nullptr;
    // uint32_t* gt_tags = nullptr; whether to use it
    float *gt_dists = nullptr;
    size_t gt_num, gt_dim;
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

    // FIXME:load query
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                                 query_aligned_dim);
    // check whether gt is corresponded with query
    if (gt_num != query_num) {
        std::cout
            << "Error. Mismatch in number of queries and ground truth data"
            << std::endl;
        exit(0);
    }
    // FIXME:load data
    T *data_load = NULL;
    size_t num_points, dim, aligned_dim;
    diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                                 aligned_dim);

    // TODO:load tags
    tsl::robin_set<TagT> active_tags, inactive_tags;
    uint64_t *metadata;
    size_t nr, nc;
    diskann::load_bin<uint64_t>(index_file, metadata, nr, nc);
    active_tags.clear();
    inactive_tags.clear();
    for (size_t i = 0; i < nr; i++) {
        active_tags.insert(i);
    }
    for (size_t i = nr; i < num_points; i++) {
        inactive_tags.insert(i);
    }

    // TODO:run_single_iter
    diskann::DistanceL2 dist_cmp;
    diskann::Metric metric = diskann::Metric::L2;
    std::string mem_prefix = save_path + "_mem";
    std::string disk_prefix_in = save_path;
    std::string disk_prefix_out = save_path + "_merge";
    diskann::MergeInsert<T, TagT> merge_insert(
        paras, dim, mem_prefix, disk_prefix_in, disk_prefix_out, &dist_cmp,
        metric, false, save_path);
    for (unsigned i = 0; i < iter_times; i++) {
        std::cout << "Run iteration " << i << std::endl;
        // run_single_iter<T, TagT>(data_load, merge_insert, num_points, 100,
        //                          aligned_dim);
        std::cout << "Iteration " << i << "ended" << std::endl;
    }
    // TODO: search
    // sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
    // Lsearch,
    //                    merge_insert, inactive_tags);
}

template <typename T = float, typename TagT = uint32_t> void unit_test() {
    std::string data_path = "/home/chenyang.sun/FreshDiskANN_Baseline/build/"
                            "data/sift/sift_learn.fbin";
    std::string index_file = "/home/chenyang.sun/FreshDiskANN_Baseline/build/"
                             "data/sift/disk_sift_disk.index";
    std::string truthset_file = "/home/chenyang.sun/FreshDiskANN_Baseline/"
                                "build/data/sift/sift_query_learn_gt100";
    std::string query_file = "/home/chenyang.sun/FreshDiskANN_Baseline/build/"
                             "data/sift/sift_query.fbin";
    std::string save_path =
        "/home/chenyang.sun/FreshDiskANN_Baseline/build/data/sift/disk_sift";
    //
    // FIXME:summary parameters
    diskann::Parameters paras;
    paras.Set<unsigned>("L_mem", 50);
    paras.Set<unsigned>("R_mem", 32);
    paras.Set<float>("alpha_mem", 1.2);
    paras.Set<unsigned>("L_disk", 50);
    paras.Set<unsigned>("R_disk", 32);
    paras.Set<float>("alpha_disk", 1.2);
    paras.Set<unsigned>("C", 160);
    paras.Set<unsigned>("beamwidth", 5);
    paras.Set<unsigned>("nodes_to_cache", 0);
    paras.Set<unsigned>("num_search_threads", 2);
    // TODO:run_single_iter
    diskann::DistanceL2 dist_cmp;
    diskann::Metric metric = diskann::Metric::L2;
    std::string mem_prefix = save_path + "_mem";
    std::string disk_prefix_in = save_path;
    std::string disk_prefix_out = save_path + "_merge";
    diskann::MergeInsert<T, TagT> merge_insert(
        paras, 128, mem_prefix, disk_prefix_in, disk_prefix_out, &dist_cmp,
        metric, false, save_path);
    // FIXME:load truthset
    unsigned *gt_ids = nullptr;
    // uint32_t* gt_tags = nullptr; whether to use it
    float *gt_dists = nullptr;
    size_t gt_num, gt_dim;
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);

    // FIXME:load query
    T *query = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                                 query_aligned_dim);
    // check whether gt is corresponded with query
    if (gt_num != query_num) {
        std::cout
            << "Error. Mismatch in number of queries and ground truth data"
            << std::endl;
        exit(0);
    }
    // FIXME:load data
    T *data_load = NULL;
    size_t num_points, dim, aligned_dim;
    diskann::load_aligned_bin<T>(data_path.c_str(), data_load, num_points, dim,
                                 aligned_dim);

    // TODO:load tags
    tsl::robin_set<TagT> active_tags, inactive_tags;
    uint64_t *metadata;
    size_t nr, nc;
    diskann::load_bin<uint64_t>(index_file, metadata, nr, nc);
    active_tags.clear();
    inactive_tags.clear();
    for (size_t i = 0; i < nr; i++) {
        active_tags.insert(i);
    }
    for (size_t i = nr; i < num_points; i++) {
        inactive_tags.insert(i);
    }
    // run single iteration
    int iter_times = 1;
    for (unsigned i = 0; i < iter_times; i++) {
        std::cout << "Run iteration " << i << std::endl;
        // run_single_iter<T, TagT>(data_load, merge_insert, num_points, 100,
        //                          aligned_dim);
        std::cout << "Iteration " << i << "ended" << std::endl;
    }
    // TODO: search
    // sync_search_kernel(query, query_num, query_aligned_dim, recall_at,
    // Lsearch,
    //                    merge_insert, inactive_tags);
}

int main() { unit_test(); }
#else
./ tests / overall_performance float../../ main_DiskANN / DiskANN / build /
    data / sift /
    sift_learn.fbin 50 32 1.2 50 32 1.2 50000 10000 500 10000.. /../
    main_DiskANN / DiskANN / build / data / true false../../ main_DiskANN /
    DiskANN / build / data / sift / sift_learn.fbin../../ main_DiskANN /
    DiskANN / build / data / sift / sift_query.fbin../../ main_DiskANN /
    DiskANN / build / data / sift / sift_query_learn_gt100 10 20 5 1000
#endif