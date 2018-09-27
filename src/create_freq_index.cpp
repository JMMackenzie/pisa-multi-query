#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/optional.hpp>

#include "succinct/mapper.hpp"

#include "configuration.hpp"
#include "index_types.hpp"
#include "scorer/bm25.hpp"
#include "util/index_build_utils.hpp"
#include "util/util.hpp"
#include "util/verify_collection.hpp"

#include "CLI/CLI.hpp"

using ds2i::logger;

uint32_t quantize(float value) {
    float  quant = 1.f / ds2i::configuration::get().reference_size;
    size_t pos   = 1;
    while (value > quant * pos)
        pos++;
    return pos - 1;
}

template <typename Collection>
void dump_index_specific_stats(Collection const &, std::string const &) {}

void dump_index_specific_stats(ds2i::uniform_index const &coll, std::string const &type) {
    ds2i::stats_line()("type", type)("log_partition_size", int(coll.params().log_partition_size));
}

void dump_index_specific_stats(ds2i::opt_index const &coll, std::string const &type) {
    auto const &conf = ds2i::configuration::get();

    uint64_t length_threshold = 4096;
    double   long_postings    = 0;
    double   docs_partitions  = 0;
    double   freqs_partitions = 0;

    for (size_t s = 0; s < coll.size(); ++s) {
        auto const &list = coll[s];
        if (list.size() >= length_threshold) {
            long_postings += list.size();
            docs_partitions += list.docs_enum().num_partitions();
            freqs_partitions += list.freqs_enum().base().num_partitions();
        }
    }

    ds2i::stats_line()("type", type)("eps1", conf.eps1)("eps2", conf.eps2)(
        "fix_cost", conf.fix_cost)("docs_avg_part", long_postings / docs_partitions)(
        "freqs_avg_part", long_postings / freqs_partitions);
}

template <typename InputCollection, typename CollectionType, typename Scorer = ds2i::bm25>
void create_collection(InputCollection const &             input,
                       ds2i::global_parameters const &     params,
                       const boost::optional<std::string> &output_filename,
                       bool                                check,
                       std::string const &                 seq_type,
                       bool                                quantized,
                       std::vector<float> &                norm_lens) {
    using namespace ds2i;
    logger() << "Processing " << input.num_docs() << " documents" << std::endl;
    double tick = get_time_usecs();

    typename CollectionType::builder builder(input.num_docs(), params);
    progress_logger                  plog;
    uint64_t                         size = 0;

    for (auto const &plist : input) {
        uint64_t freqs_sum;
        size      = plist.docs.size();
        freqs_sum = std::accumulate(plist.freqs.begin(), plist.freqs.begin() + size, uint64_t(0));
        std::vector<uint32_t> freqs;
        auto                  i = 0;
        for (auto &&f : plist.freqs) {
            if (quantized) {
                float score = Scorer::doc_term_weight(f, norm_lens[*(plist.docs.begin() + i)]);
                freqs.push_back(quantize(score));
            } else {
                freqs.push_back(f);
            }
            ++i;
        }
        builder.add_posting_list(size, plist.docs.begin(), freqs.begin(), freqs_sum);

        plog.done_sequence(size);
    }

    plog.log();
    CollectionType coll;
    builder.build(coll);
    double elapsed_secs = (get_time_usecs() - tick) / 1000000;
    logger() << seq_type << " collection built in " << elapsed_secs << " seconds" << std::endl;

    stats_line()("type", seq_type)("worker_threads", configuration::get().worker_threads)(
        "construction_time", elapsed_secs);

    dump_stats(coll, seq_type, plog.postings);
    dump_index_specific_stats(coll, seq_type);

    if (output_filename) {
        mapper::freeze(coll, output_filename.value().c_str());
        if (check) {
            verify_collection<InputCollection, CollectionType, Scorer>(
                input, output_filename.value().c_str(), quantized, norm_lens);
        }
    }
}

int main(int argc, char **argv) {

    using namespace ds2i;
    std::string                  type;
    std::string                  input_basename;
    boost::optional<std::string> output_filename;
    bool                         check     = false;
    bool                         quantized = false;

    CLI::App app{"create_freq_index - a tool for creating an index."};
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-c,--collection", input_basename, "Collection basename")->required();
    app.add_option("-o,--output", output_filename, "Output filename")->required();
    app.add_flag("--check", check, "Check the correctness of the index");
    app.add_flag("--quantized", quantized, "Quantize index frequencies");
    CLI11_PARSE(app, argc, argv);

    binary_freq_collection input(input_basename.c_str());
    binary_collection      sizes_coll((input_basename + ".sizes").c_str());
    std::vector<float>     norm_lens(input.num_docs());
    double                 lens_sum = 0;
    auto                   len_it   = sizes_coll.begin()->begin();
    for (size_t i = 0; i < norm_lens.size(); ++i) {
        float len    = *len_it++;
        norm_lens[i] = len;
        lens_sum += len;
    }
    float avg_len = float(lens_sum / double(norm_lens.size()));
    for (auto &norm_len : norm_lens) {
        norm_len /= avg_len;
    }

    ds2i::global_parameters params;
    params.log_partition_size = configuration::get().log_partition_size;

    if (false) {
#define LOOP_BODY(R, DATA, T)                                                   \
    }                                                                           \
    else if (type == BOOST_PP_STRINGIZE(T)) {                                   \
        create_collection<binary_freq_collection, BOOST_PP_CAT(T, _index)>(     \
            input, params, output_filename, check, type, quantized, norm_lens); \
        /**/

        BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, DS2I_INDEX_TYPES);
#undef LOOP_BODY
    } else {
        logger() << "ERROR: Unknown type " << type << std::endl;
    }

    return 0;
}
