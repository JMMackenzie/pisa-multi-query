#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <mio/mmap.hpp>
#include <range/v3/view/enumerate.hpp>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "mappable/mapper.hpp"

#include "accumulator/lazy_accumulator.hpp"
#include "cursor/block_max_scored_cursor.hpp"
#include "cursor/cursor.hpp"
#include "cursor/max_scored_cursor.hpp"
#include "cursor/scored_cursor.hpp"
#include "index_types.hpp"
#include "query/queries.hpp"
#include "timer.hpp"
#include "util/util.hpp"
#include "wand_data_compressed.hpp"
#include "wand_data_raw.hpp"

#include "CLI/CLI.hpp"
#include "scorer/scorer.hpp"

using namespace pisa;
using ranges::views::enumerate;

void join_thread(std::thread& t) {
    t.join();
}

template <typename Functor>
void extract_times(Functor query_func,
                   std::vector<multi_query> const &queries,
                   std::string const &index_type,
                   std::string const &query_type,
                   size_t fusion_k,
                   size_t runs,
                   std::ostream &os)
{
    std::vector<std::size_t> times(runs);
    
    for (auto const & m_query : queries) {
        for (size_t i = 0; i < runs; ++i) {
               
            topk_queue fused_top_k(fusion_k);
            std::unordered_map<uint64_t, float> fusion_accumulators;
            std::vector<std::thread> query_threads;
            std::vector<std::vector<std::pair<float, uint64_t>>> raw_results(m_query.size());
 
            double tick = get_time_usecs();
            size_t idx = 0;
            for (auto const & query : m_query) {
                
                auto q_thread = std::thread( [&, idx] {            
                    raw_results[idx] = query_func(query);
                });

                query_threads.emplace_back(std::move(q_thread));
                ++idx;
            }
            
            std::for_each(query_threads.begin(), query_threads.end(), join_thread);

            // CombSUM fusion
            for (auto const & result : raw_results) {
                for (auto const & scored_pair : result) {
                    fusion_accumulators[scored_pair.second] += scored_pair.first; 
                }
            } 
            // Now create fused top-k
            for (auto it = fusion_accumulators.begin(); it != fusion_accumulators.end(); ++it) {
                fused_top_k.insert(it->second, it->first);
            }
            double tock = get_time_usecs();
            double usecs = tock-tick;
            times[i] = usecs;
        }
        auto mean =
            std::accumulate(times.begin(), times.end(), std::size_t{0}, std::plus<>()) / runs;
        os << fmt::format("{}\t{}\n", m_query[0].id.value_or("0"), mean);
   }
}

template <typename Functor>
void op_perftest(Functor query_func,
                 std::vector<multi_query> const &queries,
                 std::string const &index_type,
                 std::string const &query_type,
                 size_t fusion_k,
                 size_t runs)
{

    std::vector<double> query_times;
    topk_queue fused_top_k(fusion_k);
    std::unordered_map<uint64_t, float> fusion_accumulators;

    for (size_t run = 0; run <= runs; ++run) {
        for (auto const & m_query : queries) {
                   
            std::vector<std::thread> query_threads;
            std::vector<std::vector<std::pair<float, uint64_t>>> raw_results(m_query.size());
 
            double tick = get_time_usecs();
            size_t idx = 0;
            for (auto const & query : m_query) {
                
                auto q_thread = std::thread( [&, idx] {            
                    raw_results[idx] = query_func(query);
                });

                query_threads.emplace_back(std::move(q_thread));
                ++idx;
            }
            
            std::for_each(query_threads.begin(), query_threads.end(), join_thread);

            // CombSUM fusion
            for (auto const & result : raw_results) {
                for (auto const & scored_pair : result) {
                    fusion_accumulators[scored_pair.second] += scored_pair.first; 
                }
            } 
            // Now create fused top-k
            for (auto it = fusion_accumulators.begin(); it != fusion_accumulators.end(); ++it) {
                fused_top_k.insert(it->second, it->first);
            }
            fused_top_k.finalize();
 
            double tock = get_time_usecs();
            double usecs = tock-tick;            
  
            if (run != 0) { // first run is not timed
                query_times.push_back(usecs);
            }
        }
    }

    if (false) {
        for (auto t : query_times) {
            std::cout << (t / 1000) << std::endl;
        }
    } else {
        std::sort(query_times.begin(), query_times.end());
        double avg =
            std::accumulate(query_times.begin(), query_times.end(), double()) / query_times.size();
        double q50 = query_times[query_times.size() / 2];
        double q90 = query_times[90 * query_times.size() / 100];
        double q95 = query_times[95 * query_times.size() / 100];

        spdlog::info("---- {} {}", index_type, query_type);
        spdlog::info("Mean: {}", avg);
        spdlog::info("50% quantile: {}", q50);
        spdlog::info("90% quantile: {}", q90);
        spdlog::info("95% quantile: {}", q95);

        stats_line()("type", index_type)("query", query_type)("avg", avg)("q50", q50)("q90", q90)(
            "q95", q95);
    }
}

template <typename IndexType, typename WandType>
void perftest(const std::string &index_filename,
              const std::optional<std::string> &wand_data_filename,
              const std::vector<multi_query> &queries,
              const std::optional<std::string> &thresholds_filename,
              std::string const &type,
              std::string const &query_type,
              uint64_t k,
              uint64_t fusion_k,
              std::string const &scorer_name,
              bool extract)
{
    IndexType index;
    spdlog::info("Loading index from {}", index_filename);
    mio::mmap_source m(index_filename.c_str());
    mapper::map(index, m);

    spdlog::info("Warming up posting lists");
    std::unordered_set<term_id_type> warmed_up;
    for (auto const & mq : queries) {
        for (auto const & q : mq) {
            for (auto t : q.terms) {
                if (!warmed_up.count(t)) {
                    index.warmup(t);
                    warmed_up.insert(t);
                }
            }
        }
    }

    WandType wdata;

    std::vector<std::string> query_types;
    boost::algorithm::split(query_types, query_type, boost::is_any_of(":"));
    mio::mmap_source md;
    if (wand_data_filename) {
        std::error_code error;
        md.map(*wand_data_filename, error);
        if (error) {
            std::cerr << "error mapping file: " << error.message() << ", exiting..." << std::endl;
            throw std::runtime_error("Error opening file");
        }
        mapper::map(wdata, md, mapper::map_flags::warmup);
    }

    std::vector<float> thresholds;
    if (thresholds_filename) {
        std::string t;
        std::ifstream tin(*thresholds_filename);
        while (std::getline(tin, t)) {
            thresholds.push_back(std::stof(t));
        }
    }

    auto scorer = scorer::from_name(scorer_name, wdata);

    spdlog::info("Performing {} queries", type);
    spdlog::info("K: {}", k);

    for (auto &&t : query_types) {
        spdlog::info("Query type: {}", t);
        std::function<std::vector<std::pair<float, uint64_t>>(Query)> query_fun;

        if (t == "wand" && wand_data_filename) {
            query_fun = [&](Query query) {
                topk_queue topk(k);
                wand_query wand_q(topk);
                wand_q(make_max_scored_cursors(index, wdata, *scorer, query),
                       index.num_docs());
                topk.finalize();
                return topk.topk();
            };
        } else if (t == "block_max_wand" && wand_data_filename) {
            query_fun = [&](Query query) {
                topk_queue topk(k);
                block_max_wand_query block_max_wand_q(topk);
                block_max_wand_q(make_block_max_scored_cursors(index, wdata, *scorer, query),
                                 index.num_docs());
                topk.finalize();
                return topk.topk();
            };
        } else if (t == "block_max_maxscore" && wand_data_filename) {
            query_fun = [&](Query query) {
                topk_queue topk(k);
                block_max_maxscore_query block_max_maxscore_q(topk);
                block_max_maxscore_q(
                    make_block_max_scored_cursors(index, wdata, *scorer, query), index.num_docs());
                topk.finalize();
                return topk.topk();
            };
        } else if (t == "ranked_or" && wand_data_filename) {
            query_fun = [&](Query query) {
                topk_queue topk(k);
                ranked_or_query ranked_or_q(topk);
                ranked_or_q(make_scored_cursors(index, *scorer, query), index.num_docs());
                topk.finalize();
                return topk.topk();
            };
        } else if (t == "maxscore" && wand_data_filename) {
            query_fun = [&](Query query) {
                topk_queue topk(k);
                maxscore_query maxscore_q(topk);
                maxscore_q(make_max_scored_cursors(index, wdata, *scorer, query),
                           index.num_docs());
                topk.finalize();
                return topk.topk();
            };
        } else {
            spdlog::error("Unsupported query type: {}", t);
            break;
        }
        
        if (extract) {
            extract_times(query_fun, queries, type, t, fusion_k, 2, std::cout);
        } else {
            op_perftest(query_fun, queries, type, t, fusion_k, 2);
        }
    }
}

using wand_raw_index = wand_data<wand_data_raw>;
using wand_uniform_index = wand_data<wand_data_compressed>;

int main(int argc, const char **argv)
{
    std::string type;
    std::string query_type;
    std::string index_filename;
    std::string scorer_name;
    std::optional<std::string> terms_file;
    std::optional<std::string> wand_data_filename;
    std::optional<std::string> query_filename;
    std::optional<std::string> thresholds_filename;
    std::optional<std::string> stopwords_filename;
    std::optional<std::string> stemmer = std::nullopt;
    uint64_t k = configuration::get().k;
    uint64_t fusion_k = 100;
    bool compressed = false;
    bool extract = false;
    bool silent = false;

    CLI::App app{"queries - a tool for performing queries on an index."};
    app.set_config("--config", "", "Configuration .ini file", false);
    app.add_option("-t,--type", type, "Index type")->required();
    app.add_option("-a,--algorithm", query_type, "Query algorithm")->required();
    app.add_option("-i,--index", index_filename, "Collection basename")->required();
    app.add_option("-w,--wand", wand_data_filename, "Wand data filename");
    app.add_option("-q,--query", query_filename, "Queries filename");
    app.add_option("-s,--scorer", scorer_name, "Scorer function")->required();
    app.add_flag("--compressed-wand", compressed, "Compressed wand input file");
    app.add_option("-k", k, "k value for per-variation top-k");
    app.add_option("-z", fusion_k, "k value for final fused list");
    app.add_option("-T,--thresholds", thresholds_filename, "k value");
    auto *terms_opt = app.add_option("--terms", terms_file, "Term lexicon");
    app.add_option("--stopwords", stopwords_filename, "File containing stopwords to ignore")
        ->needs(terms_opt);
    app.add_option("--stemmer", stemmer, "Stemmer type")->needs(terms_opt);
    app.add_flag("--extract", extract, "Extract individual query times");
    app.add_flag("--silent", silent, "Suppress logging");
    CLI11_PARSE(app, argc, argv);

    if (silent) {
        spdlog::set_default_logger(spdlog::create<spdlog::sinks::null_sink_mt>("stderr"));
    } else {
        spdlog::set_default_logger(spdlog::stderr_color_mt("stderr"));
    }
    if (extract) {
        std::cout << "qid\tusec\n";
    }

    std::vector<Query> queries;
    auto parse_query = resolve_query_parser(queries, terms_file, stopwords_filename, stemmer);
    if (query_filename) {
        std::ifstream is(*query_filename);
        io::for_each_line(is, parse_query);
    } else {
        io::for_each_line(std::cin, parse_query);
    }
    auto multi_queries = generate_multi_queries(queries);

    /**/
    if (false) {
#define LOOP_BODY(R, DATA, T)                                                          \
    }                                                                                  \
    else if (type == BOOST_PP_STRINGIZE(T))                                            \
    {                                                                                  \
        if (compressed) {                                                              \
            perftest<BOOST_PP_CAT(T, _index), wand_uniform_index>(index_filename,      \
                                                                  wand_data_filename,  \
                                                                  multi_queries,       \
                                                                  thresholds_filename, \
                                                                  type,                \
                                                                  query_type,          \
                                                                  k,                   \
                                                                  fusion_k,            \
                                                                  scorer_name,         \
                                                                  extract);            \
        } else {                                                                       \
            perftest<BOOST_PP_CAT(T, _index), wand_raw_index>(index_filename,          \
                                                              wand_data_filename,      \
                                                              multi_queries,           \
                                                              thresholds_filename,     \
                                                              type,                    \
                                                              query_type,              \
                                                              k,                       \
                                                              fusion_k,                \
                                                              scorer_name,             \
                                                              extract);                \
        }                                                                              \
        /**/

        BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY

    } else {
        spdlog::error("Unknown type {}", type);
    }
}
