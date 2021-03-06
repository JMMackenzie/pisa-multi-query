#include <iostream>
#include <optional>

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include <functional>

#include "accumulator/lazy_accumulator.hpp"
#include "mio/mmap.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "mappable/mapper.hpp"

#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include <thread>

#include "cursor/block_max_scored_cursor.hpp"
#include "cursor/max_scored_cursor.hpp"
#include "cursor/scored_cursor.hpp"
#include "index_types.hpp"
#include "io.hpp"
#include "query/queries.hpp"
#include "util/util.hpp"
#include "wand_data_compressed.hpp"
#include "wand_data_raw.hpp"

#include "scorer/scorer.hpp"

#include "CLI/CLI.hpp"

using namespace pisa;
using ranges::views::enumerate;

void join_thread(std::thread& t) {
    t.join();
}

template <typename IndexType, typename WandType>
void evaluate_queries(const std::string &index_filename,
                      const std::optional<std::string> &wand_data_filename,
                      const std::vector<multi_query> &queries,
                      const std::optional<std::string> &thresholds_filename,
                      std::string const &type,
                      std::string const &query_type,
                      uint64_t k,
                      uint64_t fusion_k,
                      std::string const &documents_filename,
                      std::string const &scorer_name,
                      std::string const &run_id = "R0",
                      std::string const &iteration = "Q0")
{
    IndexType index;
    mio::mmap_source m(index_filename.c_str());
    mapper::map(index, m);

    WandType wdata;

    auto scorer = scorer::from_name(scorer_name, wdata);

    mio::mmap_source md;
    if (wand_data_filename) {
        std::error_code error;
        md.map(*wand_data_filename, error);
        if (error) {
            spdlog::error("error mapping file: {}, exiting...", error.message());
            std::abort();
        }
        mapper::map(wdata, md, mapper::map_flags::warmup);
    }

    std::function<std::vector<std::pair<float, uint64_t>>(Query)> query_fun;

    if (query_type == "wand" && wand_data_filename) {
        query_fun = [&](Query query) {
            topk_queue topk(k);
            wand_query wand_q(topk);
            wand_q(make_max_scored_cursors(index, wdata, *scorer, query),
                   index.num_docs());
            topk.finalize();
            return topk.topk();
        };
    } else if (query_type == "block_max_wand" && wand_data_filename) {
        query_fun = [&](Query query) {
            topk_queue topk(k);
            block_max_wand_query block_max_wand_q(topk);
            block_max_wand_q(make_block_max_scored_cursors(index, wdata, *scorer, query),
                             index.num_docs());
            topk.finalize();
            return topk.topk();
        };
    } else if (query_type == "block_max_maxscore" && wand_data_filename) {
        query_fun = [&](Query query) {
            topk_queue topk(k);
            block_max_maxscore_query block_max_maxscore_q(topk);
            block_max_maxscore_q(
                make_block_max_scored_cursors(index, wdata, *scorer, query), index.num_docs());
            topk.finalize();
            return topk.topk();
        };
    } else if (query_type == "ranked_or" && wand_data_filename) {
        query_fun = [&](Query query) {
            topk_queue topk(k);
            ranked_or_query ranked_or_q(topk);
            ranked_or_q(make_scored_cursors(index, *scorer, query), index.num_docs());
            topk.finalize();
            return topk.topk();
        };
    } else if (query_type == "maxscore" && wand_data_filename) {
        query_fun = [&](Query query) {
            topk_queue topk(k);
            maxscore_query maxscore_q(topk);
            maxscore_q(make_max_scored_cursors(index, wdata, *scorer, query),
                       index.num_docs());
            topk.finalize();
            return topk.topk();
        };
    } else {
        spdlog::error("Unsupported query type: {}", query_type);
        return;
    }

    auto source = std::make_shared<mio::mmap_source>(documents_filename.c_str());
    auto docmap = Payload_Vector<>::from(*source);

    std::vector<std::vector<std::pair<float, uint64_t>>> raw_results(queries.size());
    auto start_batch = std::chrono::steady_clock::now();
    size_t query_idx = 0;

    for (auto const & m_query : queries) {
               
        topk_queue fused_top_k(fusion_k);
        std::unordered_map<uint64_t, float> fusion_accumulators;
        std::vector<std::thread> query_threads;
        std::vector<std::vector<std::pair<float, uint64_t>>> mq_results(m_query.size());

        double tick = get_time_usecs();
        size_t idx = 0;
        for (auto const & query : m_query) {
            
            auto q_thread = std::thread( [&, idx] {            
                mq_results[idx] = query_fun(query);
            });

            query_threads.emplace_back(std::move(q_thread));
            ++idx;
        }
        
        std::for_each(query_threads.begin(), query_threads.end(), join_thread);

        // CombSUM fusion
        for (auto const & result : mq_results) {
            for (auto const & scored_pair : result) {
                fusion_accumulators[scored_pair.second] += scored_pair.first; 
            }
        } 
        // Now create fused top-k
        for (auto it = fusion_accumulators.begin(); it != fusion_accumulators.end(); ++it) {
            fused_top_k.insert(it->second, it->first);
        }
        fused_top_k.finalize();
        raw_results[query_idx] = fused_top_k.topk();
        ++query_idx;
    }
 
    auto end_batch = std::chrono::steady_clock::now();

    for (size_t query_idx = 0; query_idx < raw_results.size(); ++query_idx) {
        auto results = raw_results[query_idx];
        auto qid = queries[query_idx][0].id;
        for (auto &&[rank, result] : enumerate(results)) {
            std::cout << fmt::format("{}\t{}\t{}\t{}\t{}\t{}\n",
                                     qid.value_or(std::to_string(query_idx)),
                                     iteration,
                                     docmap[result.second],
                                     rank,
                                     result.first,
                                     run_id);
        }
    }
    auto end_print = std::chrono::steady_clock::now();
    double batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count();
    double batch_with_print_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_print - start_batch).count();
    spdlog::info("Time taken to process queries: {}ms", batch_ms);
    spdlog::info("Time taken to process queries with printing: {}ms", batch_with_print_ms);
}

using wand_raw_index = wand_data<wand_data_raw>;
using wand_uniform_index = wand_data<wand_data_compressed>;

int main(int argc, const char **argv)
{
    spdlog::set_default_logger(spdlog::stderr_color_mt("default"));

    std::string type;
    std::string query_type;
    std::string index_filename;
    std::optional<std::string> terms_file;
    std::string documents_file;
    std::string scorer_name;
    std::optional<std::string> wand_data_filename;
    std::optional<std::string> query_filename;
    std::optional<std::string> thresholds_filename;
    std::optional<std::string> stopwords_filename;
    std::optional<std::string> stemmer = std::nullopt;
    std::string run_id = "R0";
    uint64_t k = configuration::get().k;
    uint64_t fusion_k = 100;
    bool compressed = false;

    CLI::App app{"Retrieves query results in TREC format."};
    app.set_config("--config", "", "Configuration .ini file", false);
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-a,--algorithm", query_type, "Query algorithm")->required();
    app.add_option("-i,--index", index_filename, "Collection basename")->required();
    app.add_option("-w,--wand", wand_data_filename, "Wand data filename");
    app.add_option("-q,--query", query_filename, "Queries filename");
    app.add_option("-r,--run", run_id, "Run identifier");
    app.add_option("-s,--scorer", scorer_name, "Scorer function")->required();
    app.add_flag("--compressed-wand", compressed, "Compressed wand input file");
    app.add_option("-k", k, "k value");
    app.add_option("-z", fusion_k, "k value for final fused list");
    auto *terms_opt = app.add_option("--terms", terms_file, "Term lexicon");
    app.add_option("--stopwords", stopwords_filename, "File containing stopwords to ignore")
        ->needs(terms_opt);
    app.add_option("--stemmer", stemmer, "Stemmer type")->needs(terms_opt);
    app.add_option("--documents", documents_file, "Document lexicon")->required();
    CLI11_PARSE(app, argc, argv);

    if (run_id.empty()) {
        run_id = "R0";
    }

    std::vector<Query> queries;
    auto push_query = resolve_query_parser(queries, terms_file, stopwords_filename, stemmer);

    if (query_filename) {
        std::ifstream is(*query_filename);
        io::for_each_line(is, push_query);
    } else {
        io::for_each_line(std::cin, push_query);
    }
    auto multi_queries = generate_multi_queries(queries);
    auto spcs_queries = multi_query_to_spcs(multi_queries); 

    /**/
    if (false) { // NOLINT
#define LOOP_BODY(R, DATA, T)                                                                  \
    }                                                                                          \
    else if (type == BOOST_PP_STRINGIZE(T))                                                    \
    {                                                                                          \
        if (compressed) {                                                                      \
            evaluate_queries<BOOST_PP_CAT(T, _index), wand_uniform_index>(index_filename,      \
                                                                          wand_data_filename,  \
                                                                          multi_queries,       \
                                                                          thresholds_filename, \
                                                                          type,                \
                                                                          query_type,          \
                                                                          k,                   \
                                                                          fusion_k,            \
                                                                          documents_file,      \
                                                                          scorer_name,         \
                                                                          run_id);             \
        } else {                                                                               \
            evaluate_queries<BOOST_PP_CAT(T, _index), wand_raw_index>(index_filename,          \
                                                                      wand_data_filename,      \
                                                                      multi_queries,           \
                                                                      thresholds_filename,     \
                                                                      type,                    \
                                                                      query_type,              \
                                                                      k,                       \
                                                                      fusion_k,                \
                                                                      documents_file,          \
                                                                      scorer_name,             \
                                                                      run_id);                 \
        }                                                                                      \
        /**/

        BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY
    } else {
        spdlog::error("Unknown type {}", type);
    }
}
