#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Constants
constexpr int MAX_CHAR_VALUE = 1114112;  // Max Unicode code point
constexpr int INITIAL_VOCAB_SIZE = 256;  // Initial byte vocabulary

struct Token {
    int value;          // Value of the token (0-16777215, 24 bits)
    // we add padding up to 32 bits for better alignment
    bool is_end_word;   // Separated flag for </w>

    __device__ __host__ Token() : value(0), is_end_word(false) {}
    __device__ __host__ Token(int v, bool end = false) : value(v), is_end_word(end) {}
};

// I'm using flat arrays instead of nested vectors for easier GPU memory access patterns
struct WordData {
    Token* tokens;           // Token array
    int* lengths;           // Word lengths
    int* offsets;           // Offsets
    int num_words;
    int total_tokens;
    int max_tokens;         // Max validated limit
};

// Map of characters to utf8 tokens.
struct UTF8TokenMap {
    int* char_to_token;     // Map of characters to tokens
    int vocab_size;
    int max_char_value;     // For bounds validation.
};

//  Vocabulary representation on the Device
struct VocabularyGPU {
    char* token_data;      // Concatenated token data
    int* token_offsets;    // Offsets for each token
    int* token_lengths;    // Length of each token
    int vocab_size;
};

// Safety: My memory layout assumes that for every i (0 <= i < token_data.size()), every token_data[i] is at offset token_offsets[i] and has length token_lengths[i].
// It is the responsability of the user to guarantee that.

// Helper function to decode a utf8 character on the device
// Safety: assumes data is valid and has at least max_len size. The caller is responsible for guaranteeing this.
__device__ int decodeUTF8Char(const char* data, int max_len, int& unicode_value) {

    if (max_len <= 0) return 0;
    // This decoding pattern follows the utf8 decoding rules.
    // First decode the first byte.
    unsigned char first = data[0];

    // ASCII (1 byte). If the first byte starts with zero, the character is ascii-compatible.
    if (first < 0x80) {
        unicode_value = first;
        return 1;
    }

    // If the first character starts with 110, It is represented with two bytes.
    // 2 bytes: 110xxxxx 10xxxxxx
    if ((first & 0xE0) == 0xC0 && max_len >= 2) {
        if ((data[1] & 0xC0) == 0x80) {
            unicode_value = ((first & 0x1F) << 6) | (data[1] & 0x3F);
            return 2;
        }
    }

    // If it starts with 1110, the character is represented with three bytes.
    // 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
    if ((first & 0xF0) == 0xE0 && max_len >= 3) {
        if ((data[1] & 0xC0) == 0x80 && (data[2] & 0xC0) == 0x80) {
            unicode_value = ((first & 0x0F) << 12) |
                ((data[1] & 0x3F) << 6) |
                (data[2] & 0x3F);
            return 3;
        }
    }

    // 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    // If it starts with 111110, it is represented with 4 bytes.
    if ((first & 0xF8) == 0xF0 && max_len >= 4) {
        if ((data[1] & 0xC0) == 0x80 && (data[2] & 0xC0) == 0x80 && (data[3] & 0xC0) == 0x80) {
            unicode_value = ((first & 0x07) << 18) |
                ((data[1] & 0x3F) << 12) |
                ((data[2] & 0x3F) << 6) |
                (data[3] & 0x3F);
            return 4;
        }
    }

    // Invalid character.
    unicode_value = first;
    return 1;
}

/// Helper function to count the number of UTF8 tokens in a single word.
/// SAFETY: the caller must ensure word_data is valid and has word_len characters.
__device__ int countUTF8Tokens(const char* word_data, int word_len) {
    int token_count = 0;
    int pos = 0;

    while (pos < word_len) {
        int unicode_value;
        // Decode utf8 char.
        int char_len = decodeUTF8Char(word_data + pos, word_len - pos, unicode_value);

        if (char_len == 0) break; // Error on decoding, out

        token_count++;
        pos += char_len;
    }

    return token_count;
}

// Hash function for bigrams
__device__ __host__ inline uint32_t hash_bigram(uint64_t key, uint32_t table_size) {
    // MurmurHash-inspired hash function
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)(key % table_size);
}

// Initial tokenization.
__global__ void tokenizeWords(
    const char* corpus_data,
    int* word_offsets,
    int* word_lengths,
    WordData* result,
    UTF8TokenMap* token_map,
    int num_words,
    int* error_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_words) return;

    // Check bounds before you access.
    if (idx >= result->num_words) {
        if (error_flags) atomicOr(&error_flags[0], 1); // We use atomic or to set the first error flag to 1, indicating index out of range.
        return;
    }

    int word_start = word_offsets[idx];
    int word_len = word_lengths[idx];
    int token_offset = result->offsets[idx];

    // Validate we have enough space.
    if (token_offset < 0 || token_offset >= result->max_tokens) {
        if (error_flags) atomicOr(&error_flags[1], 1); // Error: Invalid offset
        return;
    }

    // Check corpus bounds
    if (word_start < 0 || word_start + word_len > result->total_tokens) {
        if (error_flags) atomicOr(&error_flags[2], 1); // Error: word out of corpus range.
        return;
    }
    // ============== PROCESSING SECTION ================= //
    int pos = 0;
    int token_count = 0;

    while (pos < word_len) {
        int unicode_value;
        int char_len = decodeUTF8Char(corpus_data + word_start + pos, word_len - pos, unicode_value);

        if (char_len == 0) {
            if (error_flags) atomicOr(&error_flags[3], 1); // Error: Invalid UTF8
            break;
        }

        // Here we perform the mapping from chars to tokens.
        int token_value;
        if (unicode_value < token_map->max_char_value && // The max char value should be defined by the user.
            token_map->char_to_token[unicode_value] != -1) {
            token_value = token_map->char_to_token[unicode_value];
        }
        else {
            // Use special token for unkown token values
            token_value = token_map->vocab_size - 1; // <UNK> token
        }

        // Validate that we do not exceed the available space
        if (token_offset + token_count >= result->max_tokens) {
            if (error_flags) atomicOr(&error_flags[4], 1); // Error: no space for tokens.
            break;
        }

        // Add marker for the last token.
        bool is_last_token = (pos + char_len >= word_len);
        result->tokens[token_offset + token_count] = Token(token_value, is_last_token);

        token_count++;
        pos += char_len;
    }

    // Update the length of the processed tokens.
    result->lengths[idx] = token_count;
}

// Kernel for counting bigrams.
// Uses a hash table to scale up to very big vocabularies.
// Also, as the access pattern is linear, which enforces a coalesced memory access.
template <int blockSize>
__global__ void countBigrams(
    const WordData* words,
    int* bigram_counts,
    int* bigram_keys_high,
    int* bigram_keys_low,
    int max_bigrams,
    int vocab_size
) {
    extern __shared__ int shared_mem[];

    // Shared memory hash table
    int hash_table_size = blockSize * 2;  // Load factor 0.5
    // Using unsigned long long to store 64-bit keys for atomicCAS compatibility
    unsigned long long* hash_table = (unsigned long long*)shared_mem;
    int* hash_counts = (int*)&hash_table[hash_table_size];

    int word_idx = blockIdx.x; // Each block processes one word.
    int tid = threadIdx.x;

    if (word_idx >= words->num_words) return;

    // Initialize hash table. Each thread initializes one slot of the shared memory.
    for (int i = tid; i < hash_table_size; i += blockSize) {
        hash_table[i] = 0xFFFFFFFFFFFFFFFFULL;  // Empty slot marker for unsigned long long
        hash_counts[i] = 0;
    }
    __syncthreads();

    int word_start = words->offsets[word_idx];
    int word_len = words->lengths[word_idx];

    if (word_len < 2) return;

    // Process bigrams.
    for (int i = tid; i < word_len - 1; i += blockSize) {
        // This ensures coalesced reads, as the loads from global memory are sequential
        // Aka each thread loads its token and the next one.
        Token token1 = words->tokens[word_start + i];
        Token token2 = words->tokens[word_start + i + 1];

        if (token1.value < vocab_size && token2.value < vocab_size) {
            // Compute 64bit hash
            uint64_t bigram_key = ((uint64_t)token1.value << 32) | token2.value;
            uint32_t hash = hash_bigram(bigram_key, hash_table_size);

            // Linear probing in shared memory
            bool inserted = false;
            for (int probe = 0; probe < hash_table_size && !inserted; probe++) {
                int slot = (hash + probe) % hash_table_size;

                // Atomic CAS stands for atomic Compare And Swap.
                // Here we compare the slot with empty marker (0xFFFFFFFFFFFFFFFF)
                // If matches, we replace with the bigram key and return the old value.
                // The atomic is required so that while linear probing there are no race conditions between threads.
                unsigned long long old_key = atomicCAS(&hash_table[slot], 0xFFFFFFFFFFFFFFFFULL, bigram_key);

                if (old_key == 0xFFFFFFFFFFFFFFFFULL || old_key == bigram_key) {
                    atomicAdd(&hash_counts[slot], 1);
                    inserted = true;
                }
            }
        }
    }
    __syncthreads();

    // Reduce and write to global memory
    for (int i = tid; i < hash_table_size; i += blockSize) {
        if (hash_table[i] != 0xFFFFFFFFFFFFFFFFULL && hash_counts[i] > 0) {
            uint64_t bigram_key = hash_table[i];

            // Find slot in global arrays using atomic operations
            int global_slot = atomicAdd(&bigram_counts[0], 1);  // Use first element as counter
            if (global_slot < max_bigrams) {
                bigram_keys_high[global_slot] = (int)(bigram_key >> 32);
                bigram_keys_low[global_slot] = (int)(bigram_key & 0xFFFFFFFF);
                bigram_counts[global_slot + 1] = hash_counts[i];  // +1 to skip counter
            }
        }
    }
}

// Warp-level prefix sum using shuffle operations
__device__ __forceinline__ int warpPrefixSum(int val) {
    int lane = threadIdx.x & 31; // Lane identifier can be obtained by and-ing the thread id with 31, as on most GPUs there are around 32 threads on each warp.

    // Warp-level prefix sum using shuffle
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        // Here __shfl_up_sync copies the value of val of the neighbor thread plced offset positions below in the same warp.
        int temp = __shfl_up_sync(0xFFFFFFFF, // Bit mask to indicate that all the warps are processing.
            val, offset);
        // Accumulate the value on each thread.
        if (lane >= offset) val += temp;
    }   //This is an inclusive scan. Each thread from 0 to 31 will contain the accumulated sum of all previous threads on the warp.

    return val;
}

// Block-level prefix sum using shared memory
template <int blockSize>
__device__ int blockPrefixSum(int val, int* shared_mem) {
    int tid = threadIdx.x;
    int warpId = tid / 32; //Id of the warp in the block.
    int laneId = tid & 31; // Id of the thread in the warp.

    // Warp-level prefix sum
    int warpSum = warpPrefixSum(val);

    // Store warp sums
    if (laneId == 31) {
        shared_mem[warpId] = warpSum;
    }
    __syncthreads();

    // Prefix sum of warp sums (Only performed by warp 0)
    // The issue here is that all other warps are idle.
    if (warpId == 0) {
        int warpVal = (tid < (blockSize / 32)) ? shared_mem[tid] : 0;
        shared_mem[tid] = warpPrefixSum(warpVal);
    }
    __syncthreads();

    // Add warp offset
    int warpOffset = (warpId > 0) ? shared_mem[warpId - 1] : 0;
    return warpSum + warpOffset;
}

template <int blockSize>
__global__ void mergePairs(
    WordData* words,
    int target_token1,
    int target_token2,
    int new_token,
    int* temp_buffer,      // Global temporary buffer
    int* new_lengths       // Output array for new lengths
) {
    extern __shared__ int shared_mem[];

    int word_idx = blockIdx.x; // Again each block processes one word.
    int tid = threadIdx.x;

    if (word_idx >= words->num_words) return;

    int word_start = words->offsets[word_idx];
    int word_len = words->lengths[word_idx];

    if (word_len == 0) {
        if (tid == 0) new_lengths[word_idx] = 0;
        return;
    }

    // Shared memory layout.
    int* pair_flags = shared_mem;                    // [blockSize]
    int* prefix_sums = &shared_mem[blockSize];       // [blockSize]
    int* temp_tokens = &shared_mem[blockSize * 2];   // [blockSize]
    int* scan_temp = &shared_mem[blockSize * 3];     // [blockSize/32]

    // Parallel pair detection
    int my_flag = 0;
    if (tid < word_len - 1) {
        Token token1 = words->tokens[word_start + tid];
        Token token2 = words->tokens[word_start + tid + 1];

        if (token1.value == target_token1 && token2.value == target_token2) {
            my_flag = 1; // Pair detected.
        }
    }

    pair_flags[tid] = my_flag;
    __syncthreads();

    // Prefix sum to calculate skip positions
    int skip_before = blockPrefixSum<blockSize>(my_flag, scan_temp);
    prefix_sums[tid] = skip_before;
    __syncthreads();

    // Parallel token processing
    int output_pos = -1;
    int token_to_write = -1;
    bool should_write = false;
    Token current_token;  // Declare here to be in scope for later use

    if (tid < word_len) {
        current_token = words->tokens[word_start + tid];

        // Check if this position starts a pair
        bool is_pair_start = (tid < word_len - 1) && (pair_flags[tid] == 1);

        // Check if this position is the second token of a pair
        bool is_pair_second = (tid > 0) && (pair_flags[tid - 1] == 1);

        if (is_pair_start) {
            // This is the start of a pair - write new token
            output_pos = tid - (skip_before - 1);  // Adjust for previous skips
            token_to_write = new_token;
            should_write = true;
        }
        else if (!is_pair_second) {
            // This is a regular token (not part of a pair)
            output_pos = tid - skip_before;  // Adjust for skips
            token_to_write = current_token.value;
            should_write = true;
        }
        // If is_pair_second, we skip this token (don't write)
    }

    // Write to temporary buffer
    if (should_write && output_pos >= 0 && output_pos < blockSize) {
        temp_tokens[output_pos] = token_to_write;
    }
    __syncthreads();

    // Calculate new length
    int total_pairs = (word_len > 0) ? prefix_sums[word_len - 1] : 0;
    int new_length = word_len - total_pairs;

    // Write back to global memory
    if (tid < new_length) {
        bool is_end = (tid == new_length - 1) || current_token.is_end_word;
        words->tokens[word_start + tid] = Token(temp_tokens[tid], is_end);
    }

    if (tid == 0) {
        new_lengths[word_idx] = new_length;
        words->lengths[word_idx] = new_length;
    }
}

// Kernel to find best bigram pair on GPU
__global__ void findBestBigram(
    int* bigram_keys_high,
    int* bigram_keys_low,
    int* bigram_counts,
    int num_bigrams,
    int* best_token1,
    int* best_token2,
    int* best_count
) {
    extern __shared__ int shared_mem[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx = bid * blockDim.x + tid;

    // Local variables for reduction
    int local_best_count = 0;
    int local_best_high = -1;
    int local_best_low = -1;

    // Grid-stride loop
    for (int i = idx; i < num_bigrams; i += stride) {
        if (bigram_counts[i + 1] > local_best_count) {  // +1 to skip counter
            local_best_count = bigram_counts[i + 1];
            local_best_high = bigram_keys_high[i];
            local_best_low = bigram_keys_low[i];
        }
    }

    // Store in shared memory
    shared_mem[tid * 3] = local_best_count;
    shared_mem[tid * 3 + 1] = local_best_high;
    shared_mem[tid * 3 + 2] = local_best_low;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_mem[(tid + s) * 3] > shared_mem[tid * 3]) {
                shared_mem[tid * 3] = shared_mem[(tid + s) * 3];
                shared_mem[tid * 3 + 1] = shared_mem[(tid + s) * 3 + 1];
                shared_mem[tid * 3 + 2] = shared_mem[(tid + s) * 3 + 2];
            }
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicMax(best_count, shared_mem[0]);
        // If this block found the best, update tokens
        if (*best_count == shared_mem[0]) {
            *best_token1 = shared_mem[1];
            *best_token2 = shared_mem[2];
        }
    }
}

// BPE Processor class.
// All d_ properties represent pointers to device data.
// Properties that do not start with d_ represent pointers to host data.
class BPEProcessorCUDA {
private:
    WordData d_words;
    VocabularyGPU d_vocab;
    thrust::device_vector<int> d_bigram_counts;
    thrust::device_vector<int> d_bigram_keys_high;
    thrust::device_vector<int> d_bigram_keys_low;

    // Device pointers for best bigram
    int* d_best_token1;
    int* d_best_token2;
    int* d_best_count;

    // Streams for pipelining.
    // We maintain a separate stream for computations and another stream for data transfers.
    // This allows for parallelizing transfers with computations.
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;

    // Host data
    std::vector<std::string> vocabulary;
    int current_vocab_size;

    int* d_error_flags;
    int* d_new_lengths;
    int* d_temp_buffer;
    char* d_corpus_data;
    int* d_word_offsets;
    int* d_word_lengths;
    UTF8TokenMap d_token_map;
    WordData* d_words_struct;
    UTF8TokenMap* d_token_map_struct;


    /// BATCHING SUPPORT
    int current_batch;
    int total_batches;
    std::vector<std::vector<std::string>> batched_corpus;
    thrust::device_vector<int> h_batch_offsets;


public:
    BPEProcessorCUDA() : current_vocab_size(INITIAL_VOCAB_SIZE), current_batch(0), total_batches(0) {
        // Initialize all pointers to nullptr
        d_words.tokens = nullptr;
        d_words.lengths = nullptr;
        d_words.offsets = nullptr;
        d_words.num_words = 0;
        d_words.total_tokens = 0;
        d_words.max_tokens = 0;

        d_token_map.char_to_token = nullptr;
        d_corpus_data = nullptr;
        d_word_offsets = nullptr;
        d_word_lengths = nullptr;
        d_new_lengths = nullptr;
        d_temp_buffer = nullptr;

        // Create streams
        CUDA_CHECK(cudaStreamCreate(&stream_compute));
        CUDA_CHECK(cudaStreamCreate(&stream_transfer));

        // Allocate error flags
        CUDA_CHECK(cudaMalloc(&d_error_flags, 5 * sizeof(int)));

        // Allocate best bigram trackers
        CUDA_CHECK(cudaMalloc(&d_best_token1, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_token2, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_words_struct, sizeof(WordData)));
        CUDA_CHECK(cudaMalloc(&d_token_map_struct, sizeof(UTF8TokenMap)));

        // Allocate token storage
        CUDA_CHECK(cudaMalloc(&d_words.tokens, d_words.max_tokens * sizeof(Token)));
        CUDA_CHECK(cudaMalloc(&d_words.lengths, d_words.num_words * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_words.offsets, d_words.num_words * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_new_lengths, d_words.num_words * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_temp_buffer, d_words.max_tokens * sizeof(int)));

        // Initialize vocabulary with byte tokens
        vocabulary.reserve(10000);
        for (int i = 0; i < 256; i++) {
            vocabulary.push_back(std::string(1, (char)i));
        }

        // Initialize d_vocab pointers to nullptr
        d_vocab.token_data = nullptr;
        d_vocab.token_offsets = nullptr;
        d_vocab.token_lengths = nullptr;
        d_vocab.vocab_size = 0;

        d_words_struct = nullptr;
        d_token_map_struct = nullptr;
    }


    // Free all allocated GPU memory
    ~BPEProcessorCUDA() {
        if (d_words.tokens) cudaFree(d_words.tokens);
        if (d_words.lengths) cudaFree(d_words.lengths);
        if (d_words.offsets) cudaFree(d_words.offsets);
        if (d_token_map.char_to_token) cudaFree(d_token_map.char_to_token);
        if (d_vocab.token_data) cudaFree(d_vocab.token_data);
        if (d_vocab.token_offsets) cudaFree(d_vocab.token_offsets);
        if (d_vocab.token_lengths) cudaFree(d_vocab.token_lengths);
        if (d_error_flags) cudaFree(d_error_flags);
        if (d_new_lengths) cudaFree(d_new_lengths);
        if (d_temp_buffer) cudaFree(d_temp_buffer);
        if (d_corpus_data) cudaFree(d_corpus_data);
        if (d_word_offsets) cudaFree(d_word_offsets);
        if (d_word_lengths) cudaFree(d_word_lengths);
        if (d_best_token1) cudaFree(d_best_token1);
        if (d_best_token2) cudaFree(d_best_token2);
        if (d_best_count) cudaFree(d_best_count);

        if (d_words_struct) cudaFree(d_words_struct);
        if (d_token_map_struct) cudaFree(d_token_map_struct);

        cudaStreamDestroy(stream_compute);
        cudaStreamDestroy(stream_transfer);
    }

    void initializeFromCorpus(const std::vector<std::string>& corpus) {

        const int batch_size = 1024;
        total_batches = (corpus.size() + batch_size - 1) / batch_size;

        // Prepare the batched corpus.
        batched_corpus.resize(total_batches);
        for (int i = 0; i < total_batches; i++) {
            int start = i * batch_size;
            int end = std::min(start + batch_size, (int)corpus.size());
            batched_corpus[i] = std::vector<std::string>(corpus.begin() + start, corpus.begin() + end);
        }

        processBatch(0);
        current_batch = 1;

    }

    void processBatch(int batch_idx) {

        if (batch_idx >= total_batches) return;
        // Prepare data for GPU
        prepareCorporaData(batched_corpus[batch_idx], 0, batched_corpus[batch_idx].size());

        // Initialize UTF8 token map
        initializeTokenMap(0, batched_corpus[batch_idx].size());


        // Initial tokenization on GPU
        dim3 blockSize(256);
        dim3 gridSize((batched_corpus[batch_idx].size() + blockSize.x - 1) / blockSize.x);

        tokenizeWords << <gridSize, blockSize, 0, stream_compute >> > (
            d_corpus_data,
            d_word_offsets,
            d_word_lengths,
            d_words_struct,
            d_token_map_struct,
            batched_corpus[batch_idx].size(),
            d_error_flags
            );

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    }

    void performBPE(int num_merges) {


        for (int iter = 0; iter < num_merges; iter++) {

            for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {
                // If we are not on the first batch, prepare the next one while we process this.

                if (batch_idx > 0) {
                    // Start data transfer for the next batch.
                    if (batch_idx + 1 < total_batches) {
                        prepareCorporaDataAsync(batched_corpus[batch_idx + 1], 0,
                            batched_corpus[batch_idx + 1].size());
                    }
                }

                // Count bigrams on this batch.
                dim3 blockSize(256);
                size_t shared_mem_size = blockSize.x * 2 * sizeof(unsigned long long) +
                    blockSize.x * 2 * sizeof(int);

                countBigrams<256> << <d_words.num_words, blockSize, shared_mem_size, stream_compute >> > (
                    d_words_struct,
                    thrust::raw_pointer_cast(d_bigram_counts.data()),
                    thrust::raw_pointer_cast(d_bigram_keys_high.data()),
                    thrust::raw_pointer_cast(d_bigram_keys_low.data()),
                    d_bigram_counts.size(),
                    current_vocab_size
                    );

                // 2. Find best pair (GPU reduction)
                auto best_pair = findBestPairGPU();

                // 3. Merge pairs
                if (best_pair.first != -1) {
                    // Create and track new vocabulary token
                    int new_token = createNewToken(best_pair.first, best_pair.second);

                    shared_mem_size = blockSize.x * sizeof(int) +
                        (blockSize.x / 32) * sizeof(int) +
                        blockSize.x * sizeof(int); // temp_tokens (as ints)

                    mergePairs<256> << <d_words.num_words, blockSize, shared_mem_size, stream_compute >> > (
                        d_words_struct,
                        best_pair.first,
                        best_pair.second,
                        new_token,
                        d_temp_buffer,
                        d_new_lengths
                        );
                }

                CUDA_CHECK(cudaStreamSynchronize(stream_compute));
                if (batch_idx + 1 < total_batches) {
                    initializeTokenMapAsync(0, batched_corpus[batch_idx + 1].size());
                }
            }
        }
    }

    // Get results back to CPU (only at the end)
    std::vector<std::vector<int>> getEncodedCorpus() {
        // Copy encoded data back to host
        thrust::device_vector<Token> d_tokens_vec(d_words.tokens, d_words.tokens + d_words.total_tokens);
        thrust::device_vector<int> d_lengths_vec(d_words.lengths, d_words.lengths + d_words.num_words);
        thrust::device_vector<int> d_offsets_vec(d_words.offsets, d_words.offsets + d_words.num_words);

        thrust::host_vector<Token> h_tokens = d_tokens_vec;
        thrust::host_vector<int> h_lengths = d_lengths_vec;
        thrust::host_vector<int> h_offsets = d_offsets_vec;

        std::vector<std::vector<int>> encoded_corpus;
        for (int i = 0; i < d_words.num_words; i++) {
            std::vector<int> word_tokens;
            for (int j = 0; j < h_lengths[i]; j++) {
                word_tokens.push_back(h_tokens[h_offsets[i] + j].value);
            }
            encoded_corpus.push_back(word_tokens);
        }

        return encoded_corpus;
    }

    const std::vector<std::string>& getVocabulary() const {
        return vocabulary;
    }

private:

    void prepareCorporaDataAsync(const std::vector<std::string>& corpus, int batch_start, int batch_end) {
        // Asynchronous variant for preparing corpora data.
        int batch_size = batch_end - batch_start;
        size_t total_chars = 0;
        std::vector<int> word_offsets(batch_size);
        std::vector<int> word_lengths(batch_size);

        for (int i = 0; i < batch_size; i++) {
            const std::string& word = corpus[batch_start + i];
            word_offsets[i] = total_chars;
            word_lengths[i] = word.length();
            total_chars += word.length();
        }

        std::string concatenated;
        concatenated.reserve(total_chars);
        for (int i = 0; i < batch_size; i++) {
            concatenated += corpus[batch_start + i];
        }

        // Usar stream_transfer para operaciones asíncronas
        CUDA_CHECK(cudaMemcpyAsync(d_corpus_data, concatenated.data(), total_chars,
            cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaMemcpyAsync(d_word_offsets, word_offsets.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaMemcpyAsync(d_word_lengths, word_lengths.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));

        d_words.num_words = batch_size;
        d_words.total_tokens = total_chars * 2;
        d_words.max_tokens = d_words.total_tokens;

        std::vector<int> token_offsets;
        int current_offset = 0;
        for (int i = 0; i < batch_size; i++) {
            token_offsets.push_back(current_offset);
            current_offset += word_lengths[i] * 2;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_words.offsets, token_offsets.data(),
            d_words.num_words * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));
    }

    // New method to initialize the token map asynchronously
    void initializeTokenMapAsync(int batch_start, int batch_end) {
        d_token_map.vocab_size = current_vocab_size;
        d_token_map.max_char_value = MAX_CHAR_VALUE;

        std::vector<int> char_to_token(MAX_CHAR_VALUE, -1);
        for (int i = 0; i < 256; i++) {
            char_to_token[i] = i;
        }

        CUDA_CHECK(cudaMemcpyAsync(d_token_map.char_to_token, char_to_token.data(),
            MAX_CHAR_VALUE * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));
    }


    std::pair<int, int> findBestPairGPU() {
        // Find best bigram pair using GPU reduction

        // Get bigram count
        int bigram_count;
        CUDA_CHECK(cudaMemcpy(&bigram_count, d_bigram_counts.data().get(),
            sizeof(int), cudaMemcpyDeviceToHost));

        if (bigram_count == 0) return { -1, -1 };

        // Find best bigram using GPU kernel
        dim3 blockSize(256);
        dim3 gridSize((bigram_count + blockSize.x - 1) / blockSize.x);

        findBestBigram << <gridSize, blockSize, blockSize.x * 3 * sizeof(int), stream_compute >> > (
            thrust::raw_pointer_cast(d_bigram_keys_high.data()),
            thrust::raw_pointer_cast(d_bigram_keys_low.data()),
            thrust::raw_pointer_cast(d_bigram_counts.data()),
            bigram_count,
            d_best_token1,
            d_best_token2,
            d_best_count
            );

        // Copy results
        int best_token1, best_token2, best_count;
        CUDA_CHECK(cudaMemcpy(&best_token1, d_best_token1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&best_token2, d_best_token2, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&best_count, d_best_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (best_count > 0) {
            return { best_token1, best_token2 };
        }

        return { -1, -1 };
    }

    void prepareCorporaData(const std::vector<std::string>& corpus, int batch_start, int batch_end) {
        // Allocate GPU memory, copy data from host to device,
        // initialize offsets and lengths arrays,
        // handle memory alignment for coalesced access

        // Calculate total size
        int batch_size = batch_end - batch_start;
        size_t total_chars = 0;
        std::vector<int> word_offsets(batch_size);
        std::vector<int> word_lengths(batch_size);

        for (int i = 0; i < batch_size; i++) {
            const std::string& word = corpus[batch_start + i];
            word_offsets[i] = total_chars;
            word_lengths[i] = word.length();
            total_chars += word.length();
        }

        // Prepare concatenated corpus for this batch.
        std::string concatenated;
        concatenated.reserve(total_chars);
        for (int i = 0; i < batch_size; i++) {
            concatenated += corpus[batch_start + i];
        }

        // Free previously allocated memory if needed.
        if (d_corpus_data) cudaFree(d_corpus_data);
        if (d_word_offsets) cudaFree(d_word_offsets);
        if (d_word_lengths) cudaFree(d_word_lengths);


        // Allocate device memory for this batch.
        CUDA_CHECK(cudaMalloc(&d_corpus_data, total_chars));
        CUDA_CHECK(cudaMalloc(&d_word_offsets, batch_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_word_lengths, batch_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_new_lengths, batch_size * sizeof(int)));

        // Copy to device using streams
        CUDA_CHECK(cudaMemcpyAsync(d_corpus_data, concatenated.data(), total_chars,
            cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaMemcpyAsync(d_word_offsets, word_offsets.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaMemcpyAsync(d_word_lengths, word_lengths.data(),
            batch_size * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));

        // Prepare WordData structure
        d_words.num_words = batch_size;
        d_words.total_tokens = total_chars * 2; // Conservative estimate
        d_words.max_tokens = d_words.total_tokens;


        // Reset token offsets.
        if (d_words.offsets) cudaFree(d_words.offsets);
        CUDA_CHECK(cudaMalloc(&d_words.offsets, batch_size * sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_new_lengths, 0, batch_size * sizeof(int), stream_transfer));

        // Calculate token offsets
        std::vector<int> token_offsets;
        int current_offset = 0;
        for (int i = batch_start; i < batch_end; i++) {
            token_offsets.push_back(current_offset);
            current_offset += word_lengths[i] * 2; // estimación conservadora
        }


        CUDA_CHECK(cudaMemcpyAsync(d_words.offsets, token_offsets.data(),
            d_words.num_words * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));

        // Initialize bigram storage
        size_t max_bigrams = d_words.num_words * 100; // Conservative estimate
        d_bigram_counts.resize(max_bigrams);
        d_bigram_keys_high.resize(max_bigrams);
        d_bigram_keys_low.resize(max_bigrams);

        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
    }

    void initializeTokenMap(int batch_start, int batch_end) {
        // Initialize UTF8 token map
        d_token_map.vocab_size = current_vocab_size;
        d_token_map.max_char_value = MAX_CHAR_VALUE;

        // Create mapping on host
        std::vector<int> char_to_token(MAX_CHAR_VALUE, -1);
        for (int i = 0; i < 256; i++) {
            char_to_token[i] = i;
        }

        // Allocate and copy to device
        CUDA_CHECK(cudaMalloc(&d_token_map.char_to_token, MAX_CHAR_VALUE * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_token_map.char_to_token, char_to_token.data(),
            MAX_CHAR_VALUE * sizeof(int),
            cudaMemcpyHostToDevice, stream_transfer));

        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));
    }

    int createNewToken(int token1, int token2) {
        // Create new token from merging two existing tokens
        std::string new_token_str = vocabulary[token1] + vocabulary[token2];
        vocabulary.push_back(new_token_str);

        int new_token_id = current_vocab_size++;

        // Update vocabulary on GPU if needed
        updateVocabularyGPU(new_token_id, new_token_str);

        return new_token_id;
    }

    void updateVocabularyGPU(int token_id, const std::string& token_str) {
        // Update vocabulary representation on GPU if needed
        // For now, we just track on CPU and sync at the end
        d_vocab.vocab_size = current_vocab_size;
    }
};

// Utility function to load corpus from file
std::vector<std::string> loadCorpusFromFile(const std::string& filename) {
    std::vector<std::string> corpus;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return corpus;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) {
            // Add end-of-word marker
            word += "</w>";
            corpus.push_back(word);
        }
    }

    file.close();
    return corpus;
}

// Create sample corpus for testing
std::vector<std::string> createSampleCorpus() {
    return {
        "low</w>", "low</w>", "low</w>", "low</w>", "low</w>",
        "lower</w>", "lower</w>", "newer</w>", "newer</w>", "newer</w>",
        "wider</w>", "wider</w>", "new</w>", "new</w>"
    };
}

// Main function with pipelining
int main(int argc, char** argv) {
    // Check CUDA availability
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    // Set device
    CUDA_CHECK(cudaSetDevice(0));

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // Load corpus (only CPU operation)
    std::vector<std::string> corpus;
    if (argc > 1) {
        std::cout << "Loading corpus from file: " << argv[1] << std::endl;
        corpus = loadCorpusFromFile(argv[1]);
    }
    else {
        std::cout << "Using sample corpus" << std::endl;
        corpus = createSampleCorpus();
    }

    if (corpus.empty()) {
        std::cerr << "Error: Empty corpus!" << std::endl;
        return 1;
    }

    std::cout << "Corpus size: " << corpus.size() << " words" << std::endl;

    // Initialize BPE processor
    BPEProcessorCUDA bpe_processor;

    // Measure time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize from corpus (transfers data to GPU)
    bpe_processor.initializeFromCorpus(corpus);

    // Perform BPE merges (all on GPU)
    int num_merges = (argc > 2) ? std::atoi(argv[2]) : 100;
    std::cout << "\nPerforming " << num_merges << " BPE merges..." << std::endl;

    bpe_processor.performBPE(num_merges);

    // Get results back (only transfer at the end)
    auto encoded_corpus = bpe_processor.getEncodedCorpus();
    const auto& vocabulary = bpe_processor.getVocabulary();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print results
    std::cout << "\nTotal processing time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final vocabulary size: " << vocabulary.size() << std::endl;

    // Print sample encoded words
    std::cout << "\nSample encoded words:" << std::endl;
    for (int i = 0; i < std::min(5, (int)encoded_corpus.size()); i++) {
        std::cout << "Word " << i << ": ";
        for (int token : encoded_corpus[i]) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
