#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <utility>
#include <string>

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

const std::string FILENAME = "../corpus.txt";

// Custom Hash for Pair<string,string>
struct PairHash {
    size_t operator()(const pair<string, string>& p) const {
        return hash<string>()(p.first) ^ (hash<string>()(p.second) << 1);
    }
};

// Class to handle BPE computation
class BPEProcessor {
private:
    vector<vector<string>> tokens;
    unordered_map<string, int> vocabulary;

public:
    void reserveSpace(size_t corpusSize) {
        tokens.reserve(corpusSize);
        vocabulary.reserve(corpusSize * 10);
    }

    vector<vector<string>>& getTokens() { return tokens; }
    const vector<vector<string>>& getTokens() const { return tokens; }

    // Add word methos that copies the input word.
    void addWord(const vector<string>& word) {
        tokens.push_back(word);
    }

    // More efficient alternative that uses std::move
    void addWord(vector<string>&& word) {
        tokens.push_back(std::move(word));
    }
};

std::vector<std::string> readCorpus(const std::string& filename) {
    std::vector<std::string> words;
    std::ifstream file(filename);
    std::string word;

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return words;
    }

    // Reserve expected space for efficiency.
    words.reserve(10000);

    while (file >> word) {
        words.push_back(std::move(word));
    }

    return words;
}

// Tokenize a word into characyers.
vector<string> bpeTokenize(const string& word) {
    vector<string> tokens;
    tokens.reserve(word.length() + 1);

    for (char c : word) {
        tokens.emplace_back(1, c);
    }

    if (!tokens.empty()) {
        tokens.back() += "</w>";
    }

    return tokens;
}

// Join tokens (for printing purposes)
string joinTokens(const vector<string>& tokens) {
    string result;
    size_t totalLength = 0;

    // Compute total length previously to be able to reserve space.
    for (const auto& token : tokens) {
        totalLength += token.length() + 1;
    }

    result.reserve(totalLength);

    for (const auto& token : tokens) {
        result += token + " ";
    }

    return result;
}

// Function to get the bigrams from a vector of words.
unordered_map<pair<string, string>, int, PairHash> getBigrams(const vector<vector<string>>& words) {
    unordered_map<pair<string, string>, int, PairHash> globalFreq;

#ifdef _OPENMP
#pragma omp parallel
    {
        unordered_map<pair<string, string>, int, PairHash> localFreq;

#pragma omp for nowait
        for (size_t idx = 0; idx < words.size(); ++idx) {
            const auto& word = words[idx];
            for (size_t i = 0; i < word.size() - 1; ++i) {
                localFreq[{word[i], word[i + 1]}]++;
            }
        }

        // More efficient manual reduction.
#pragma omp critical
        {
            for (const auto& p : localFreq) {
                globalFreq[p.first] += p.second;
            }
        }
    }
#else
    for (size_t idx = 0; idx < words.size(); ++idx) {
        const auto& word = words[idx];
        for (size_t i = 0; i < word.size() - 1; ++i) {
            globalFreq[{word[i], word[i + 1]}]++;
        }
    }
#endif

    return globalFreq;
}

// Optimized version of merge pair.
void mergePair(vector<vector<string>>& words, const pair<string, string>& bigram) {
    const string& first = bigram.first;
    const string& second = bigram.second;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(words.size()); ++i) {
        auto& word = words[i];
        vector<string> newWord;
        newWord.reserve(word.size()); // Reserve space.

        for (size_t j = 0; j < word.size(); ++j) {
            if (j < word.size() - 1 && word[j] == first && word[j + 1] == second) {
                newWord.push_back(first + second);
                ++j; // Jump to the next token.
            }
            else {
                newWord.push_back(std::move(word[j])); // Use move.
            }
        }
        word = std::move(newWord);
    }
}

// Function to find the best pair in  the set of bigrams.
// The best pair is the one that appears more times.
pair<string, string> findBestPair(const unordered_map<pair<string, string>, int, PairHash>& bigrams) {
    if (bigrams.empty()) {
        return { "", "" };
    }

    auto bestPair = std::max_element(
        bigrams.begin(), bigrams.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        }
    );

    return bestPair->first;
}

//
void printStats(const vector<vector<string>>& tokens, int iteration) {
    cout << "=== Iteración " << iteration << " ===" << endl;


    size_t maxWords = std::min(tokens.size(), size_t(10));
    for (size_t i = 0; i < maxWords; ++i) {
        cout << joinTokens(tokens[i]) << endl;
    }

    if (tokens.size() > 10) {
        cout << "... y " << (tokens.size() - 10) << " palabras más" << endl;
    }

    cout << "-------------------" << endl;
}

int main(int argc, char* argv[]) {
    auto start_time = chrono::high_resolution_clock::now();


    const char* filename = argc > 1 ? argv[1] : FILENAME.c_str();
    int numThreads = argc > 2 ? atoi(argv[2]) : 1;
    int numMerges = argc > 3 ? atoi(argv[3]) : 5;

    cout << "Reading corpus from: " << filename << endl;
    cout << "Number of threads: " << numThreads << endl;
    cout << "Number of merges: " << numMerges << endl;

    // Read corpus
    vector<string> corpus = readCorpus(filename);

    if (corpus.empty()) {
        cerr << "Error: Corpus is empty or cannot be read." << endl;
            return 1;
    }

    cout << "Corpus read: " << corpus.size() << " words" << endl;

    #ifdef _OPENMP
        omp_set_num_threads(numThreads);
        cout << "OpenMP enabled." << endl;
    #else
        cout << "OpenMP no está habilitado." << endl;
    #endif

    // Initialize BPEprocessor
    BPEProcessor processor;
    processor.reserveSpace(corpus.size());

    // Initial tokenization
    #ifdef _OPENMP
        double start1 = omp_get_wtime();
    #pragma omp parallel for
    #endif
        for (size_t i = 0; i < corpus.size(); ++i) {
            auto tokens = bpeTokenize(corpus[i]);
    #ifdef _OPENMP
    #pragma omp critical
    #endif
    {
        processor.addWord(std::move(tokens));
    }
    }

    #ifdef _OPENMP
        double end1 = omp_get_wtime();
        double duration1 = end1 - start1;
        cout << "Duración de tokenización inicial: " << duration1 << " segundos" << endl;
    #endif

    // Obtain a reference to the tokens
    auto& tokens = processor.getTokens();

    // Main BPE loop
    for (int i = 0; i < numMerges; ++i) {
    #ifdef _OPENMP
        double start2 = omp_get_wtime();
    #endif

        auto bigrams = getBigrams(tokens);

    #ifdef _OPENMP
        double end2 = omp_get_wtime();
        double duration2 = end2 - start2;
        cout << "Duration of the bigrams: " << duration2 << " seconds" << endl;
    #endif

        if (bigrams.empty()) {
            cout << "No more bigrams to fuse." << endl;
            break;
        }

        auto bestPair = findBestPair(bigrams);

        if (bestPair.first.empty()) {
            cout << "Did not found a valid pair to fuse." << endl;
            break;
        }

    #ifdef _OPENMP
        double start3 = omp_get_wtime();
    #endif

        cout << "Fusing: '" << bestPair.first << "' + '" << bestPair.second
            << "' (" << bigrams[bestPair] << " times)" << endl;

        mergePair(tokens, bestPair);

    #ifdef _OPENMP
        double end3 = omp_get_wtime();
        double duration3 = end3 - start3;
        cout << "Fusing duration: " << duration3 << " seconds" << endl;
    #endif

        // Print stats
        printStats(tokens, i + 1);
    }

    // Total time
    auto end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Total exeution time: " << total_duration.count() << " ms" << endl;

    cout << "\n=== FINAL STATS ===" << endl;
    cout << "Processed words: " << tokens.size() << endl;

    // Compute final vocab
    unordered_map<string, int> finalVocab;
    for (const auto& word : tokens) {
        for (const auto& token : word) {
            finalVocab[token]++;
        }
    }

    cout << "Final vocab size: " << finalVocab.size() << endl;

    return 0;
}
