#pragma once

#include <gsl/span>

template <typename WandType>
struct block_max_maxscore_query {

    typedef bm25 scorer_type;

    block_max_maxscore_query(WandType const &wdata, uint64_t k) : m_wdata(&wdata), m_topk(k) {}

    template <class Enum>
    [[gnu::always_inline]] [[nodiscard]] auto score_essential(gsl::span<Enum *> essential_enums,
                                                              uint64_t cur_doc,
                                                              uint64_t num_docs,
                                                              float norm_len) const
        -> std::pair<float, uint64_t>
    {
        float score = 0.0;
        uint64_t next_doc = num_docs;
        for (auto &essential : essential_enums) {
            auto &cursor = essential->docs_enum;
            if (cursor.docid() == cur_doc) {
                score +=
                    essential->q_weight * scorer_type::doc_term_weight(cursor.freq(), norm_len);
                cursor.next();
            }
            if (cursor.docid() < next_doc) {
                next_doc = cursor.docid();
            }
        }
        return std::make_pair(score, next_doc);
    }

    template <class Enum>
    [[gnu::always_inline]] [[nodiscard]] auto current_block_upper_bound(
        gsl::span<Enum *> non_essential_enums,
        float block_upper_bound,
        uint64_t cur_doc,
        float score) -> double
    {
        auto enum_it = non_essential_enums.rbegin();
        for (; enum_it != non_essential_enums.rend(); ++enum_it) {
            auto &non_essential = *enum_it;
            if (non_essential->w.docid() < cur_doc) {
                non_essential->w.next_geq(cur_doc);
            }
            block_upper_bound -=
                non_essential->max_weight - non_essential->w.score() * non_essential->q_weight;
            if (not m_topk.would_enter(score + block_upper_bound)) {
                break;
            }
        }
        return block_upper_bound;
    }

    template <class Enum>
    [[gnu::always_inline]] [[nodiscard]] auto score_non_essential(
        gsl::span<Enum *> non_essential_enums,
        float block_upper_bound,
        uint64_t cur_doc,
        float score,
        float norm_len) -> double
    {
        auto enum_it = non_essential_enums.rbegin();
        for (; enum_it != non_essential_enums.rend(); ++enum_it) {
            auto &non_essential = *enum_it;
            non_essential->docs_enum.next_geq(cur_doc);
            if (non_essential->docs_enum.docid() == cur_doc) {
                auto s = non_essential->q_weight *
                         scorer_type::doc_term_weight(non_essential->docs_enum.freq(), norm_len);
                block_upper_bound += s;
            }
            block_upper_bound -= non_essential->w.score() * non_essential->q_weight;

            if (not m_topk.would_enter(score + block_upper_bound)) {
                break;
            }
        }
        return score + block_upper_bound;
    }

    template <typename Index>
    uint64_t operator()(Index const &index, term_id_vec const &terms) {
        m_topk.clear();
        if (terms.empty())
            return 0;

        auto query_term_freqs = query_freqs(terms);

        uint64_t                                        num_docs = index.num_docs();
        typedef typename Index::document_enumerator     enum_type;
        typedef typename WandType::wand_data_enumerator wdata_enum;

        struct scored_enum {
            enum_type  docs_enum;
            wdata_enum w;
            float      q_weight;
            float      max_weight;
        };

        std::vector<scored_enum> enums;
        enums.reserve(query_term_freqs.size());

        for (auto term : query_term_freqs) {
            auto list       = index[term.first];
            auto w_enum     = m_wdata->getenum(term.first);
            auto q_weight   = scorer_type::query_term_weight(term.second, list.size(), num_docs);
            auto max_weight = q_weight * m_wdata->max_term_weight(term.first);
            enums.push_back(scored_enum{std::move(list), w_enum, q_weight, max_weight});
        }

        std::vector<scored_enum *> ordered_enums;
        ordered_enums.reserve(enums.size());
        for (auto &en : enums) {
            ordered_enums.push_back(&en);
        }

        auto increasing_maxscore_order = [](auto *lhs, auto *rhs) -> bool {
            return lhs->max_weight < rhs->max_weight;
        };
        auto increasing_docid_order = [](auto const &lhs, auto const &rhs) -> bool {
            return lhs.docs_enum.docid() < rhs.docs_enum.docid();
        };

        std::sort(ordered_enums.begin(), ordered_enums.end(), increasing_maxscore_order);

        std::vector<float> upper_bounds(ordered_enums.size());
        std::transform(ordered_enums.begin(),
                       ordered_enums.end(),
                       upper_bounds.begin(),
                       [](auto const &elem) { return elem->max_weight; });
        std::partial_sum(upper_bounds.begin(), upper_bounds.end(), upper_bounds.begin());

        int non_essential_lists = 0;
        uint64_t cur_doc =
            std::min_element(enums.begin(), enums.end(), increasing_docid_order)->docs_enum.docid();

        while (non_essential_lists < ordered_enums.size() && cur_doc < index.num_docs()) {
            float norm_len = m_wdata->norm_len(cur_doc);

            auto [score, next_doc] = score_essential(
                gsl::span<scored_enum *>(ordered_enums).subspan(non_essential_lists),
                cur_doc,
                index.num_docs(),
                norm_len);

            auto block_upper_bound = current_block_upper_bound(
                gsl::span<scored_enum *>(ordered_enums).first(non_essential_lists),
                non_essential_lists > 0 ? upper_bounds[non_essential_lists - 1] : 0,
                cur_doc,
                score);

            if (m_topk.would_enter(score + block_upper_bound)) {
                score = score_non_essential(
                    gsl::span<scored_enum *>(ordered_enums).first(non_essential_lists),
                    block_upper_bound,
                    cur_doc,
                    score,
                    norm_len);
            }

            if (m_topk.insert(score)) {
                while (non_essential_lists < ordered_enums.size() &&
                       not m_topk.would_enter(upper_bounds[non_essential_lists])) {
                    non_essential_lists += 1;
                }
            }
            cur_doc = next_doc;
        }
        m_topk.finalize();
        return m_topk.topk().size();
    }

    std::vector<std::pair<float, uint64_t>> const &topk() const { return m_topk.topk(); }

   private:
    WandType const *m_wdata;
    topk_queue      m_topk;
};
