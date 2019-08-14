#include <vector>  // std::vector


namespace signatory {
    namespace fla_ops {
        inline int64_t num_lyndon_words(const std::vector<std::vector<LyndonWord>>& lyndon_words,
                                        const misc::LyndonSpec& lyndonspec) {
            int64_t num;
            if (lyndonspec.input_channels == 1) {
                // In this case there only exists a singe Lyndon word '0', at lyndon_words[0].back(). There are now
                // higher-depth words, i.e. lyndon_words[1], lyndon_words[2], ... etc. are all size-0 vectors.
                num = 1;
            }
            else {
                num = lyndon_words.back().back().compressed_index + 1;
            }
            return num;
        }
    }  // namespace signatory::fla_ops
}  // namespace signatory