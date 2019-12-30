#pragma once

#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <range/v3/view/enumerate.hpp>
#include <spdlog/spdlog.h>
#include "index_types.hpp"
#include "query/queries.hpp"
#include "term_processor.hpp"
#include "tokenizer.hpp"
#include "topk_queue.hpp"
#include "util/util.hpp"
#include "wand_data.hpp"
#include "wand_data_compressed.hpp"
#include "wand_data_raw.hpp"

namespace pisa {

using term_id_type = uint32_t;
using term_id_vec = std::vector<term_id_type>;

struct Query {
    std::optional<std::string> id;
    std::vector<term_id_type> terms;
    std::vector<float> term_weights;
};

[[nodiscard]] auto split_query_at_colon(std::string const &query_string)
    -> std::pair<std::optional<std::string>, std::string_view>
{
    // query id : terms (or ids)
    auto colon = std::find(query_string.begin(), query_string.end(), ':');
    std::optional<std::string> id;
    if (colon != query_string.end()) {
        id = std::string(query_string.begin(), colon);
    }
    auto pos = colon == query_string.end() ? query_string.begin() : std::next(colon);
    auto raw_query = std::string_view(&*pos, std::distance(pos, query_string.end()));
    return {std::move(id), std::move(raw_query)};
}

[[nodiscard]] auto parse_query_terms(std::string const &query_string, TermProcessor term_processor)
    -> Query
{
    auto [id, raw_query] = split_query_at_colon(query_string);
    TermTokenizer tokenizer(raw_query);
    std::vector<term_id_type> parsed_query;
    for (auto term_iter = tokenizer.begin(); term_iter != tokenizer.end(); ++term_iter) {
        auto raw_term = *term_iter;
        auto term = term_processor(raw_term);
        if (term) {
            if (!term_processor.is_stopword(*term)) {
                parsed_query.push_back(std::move(*term));
            } else {
                spdlog::warn("Term `{}` is a stopword and will be ignored", raw_term);
            }
        } else {
            spdlog::warn("Term `{}` not found and will be ignored", raw_term);
        }
    }
    return {std::move(id), std::move(parsed_query), {}};
}

[[nodiscard]] auto parse_query_ids(std::string const &query_string) -> Query
{
    auto [id, raw_query] = split_query_at_colon(query_string);
    std::vector<term_id_type> parsed_query;
    std::vector<std::string> term_ids;
    boost::split(term_ids, raw_query, boost::is_any_of("\t, ,\v,\f,\r,\n"));

    auto is_empty = [](const std::string &val) { return val.empty(); };
    // remove_if move matching elements to the end, preparing them for erase.
    term_ids.erase(std::remove_if(term_ids.begin(), term_ids.end(), is_empty), term_ids.end());

    try {
        auto to_int = [](const std::string &val) { return std::stoi(val); };
        std::transform(term_ids.begin(), term_ids.end(), std::back_inserter(parsed_query), to_int);
    } catch (std::invalid_argument &err) {
        spdlog::error("Could not parse term identifiers of query `{}`", raw_query);
        exit(1);
    }
    return {std::move(id), std::move(parsed_query), {}};
}

[[nodiscard]] std::function<void(const std::string)> resolve_query_parser(
    std::vector<Query> &queries,
    std::optional<std::string> const &terms_file,
    std::optional<std::string> const &stopwords_filename,
    std::optional<std::string> const &stemmer_type)
{
    if (terms_file) {
        auto term_processor = TermProcessor(terms_file, stopwords_filename, stemmer_type);
        return [&queries, term_processor = std::move(term_processor)](
                   std::string const &query_line) {
            queries.push_back(parse_query_terms(query_line, term_processor));
        };
    } else {
        return [&queries](std::string const &query_line) {
            queries.push_back(parse_query_ids(query_line));
        };
    }
}

bool read_query(term_id_vec &ret, std::istream &is = std::cin)
{
    ret.clear();
    std::string line;
    if (!std::getline(is, line)) {
        return false;
    }
    ret = parse_query_ids(line).terms;
    return true;
}

void remove_duplicate_terms(term_id_vec &terms)
{
    std::sort(terms.begin(), terms.end());
    terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
}

typedef std::pair<uint64_t, uint64_t> term_freq_pair;
typedef std::vector<term_freq_pair> term_freq_vec;

term_freq_vec query_freqs(term_id_vec terms)
{
    term_freq_vec query_term_freqs;
    std::sort(terms.begin(), terms.end());
    // count query term frequencies
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i == 0 || terms[i] != terms[i - 1]) {
            query_term_freqs.emplace_back(terms[i], 1);
        } else {
            query_term_freqs.back().second += 1;
        }
    }
    return query_term_freqs;
}

using multi_query = std::vector<Query>;
// Consume a vector of queries, and convert to multi-queries
std::vector<multi_query> generate_multi_queries (std::vector<Query> queries)
{
    std::vector<multi_query> multi_queries;

    std::map<std::string, multi_query> mapped_queries;
    for (auto &q : queries) {
        std::string id = q.id.value_or("");
        if (id == "") {
            spdlog::error("Error: Multi Queries must have IDs");
            exit(1);
        }
        remove_duplicate_terms(q.terms); // Ensure queries are unique terms only
        mapped_queries[id].push_back(q);
    }

    for (const auto &elem : mapped_queries)
        multi_queries.push_back(elem.second);
  
    spdlog::info("Read {} multi queries.", multi_queries.size());
    return multi_queries;

}

// Convert a multi-query into the SP-CS format
std::vector<Query> multi_query_to_spcs (std::vector<multi_query> queries) 
{
    std::vector<Query> spcs_queries;
    
    std::map<std::string, Query> q_map;
    size_t count = 0;
    for (const auto & multi : queries) {
        for (const auto & query : multi) {
            ++count;
            std::string id = query.id.value_or("");
            if (id == "") {
                spdlog::error("Error: Multi Queries must have IDs");
                exit(1);
            }
            q_map[id].id = id;
            for (const auto & term : query.terms) {
                q_map[id].terms.push_back(term);
            }
        }
    }
    
    for (const auto &elem : q_map) {
      spcs_queries.push_back(elem.second);
    }

    spdlog::info("Converted {} queries into {} SP-CS queries.", count, spcs_queries.size());
    return spcs_queries;
}

} // namespace pisa

#include "algorithm/and_query.hpp"
#include "algorithm/block_max_maxscore_query.hpp"
#include "algorithm/block_max_ranked_and_query.hpp"
#include "algorithm/block_max_wand_query.hpp"
#include "algorithm/maxscore_query.hpp"
#include "algorithm/or_query.hpp"
#include "algorithm/range_query.hpp"
#include "algorithm/range_taat_query.hpp"
#include "algorithm/ranked_and_query.hpp"
#include "algorithm/ranked_or_query.hpp"
#include "algorithm/ranked_or_taat_query.hpp"
#include "algorithm/wand_query.hpp"
