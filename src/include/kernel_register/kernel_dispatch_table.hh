/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace dlp::kernel_frame {

constexpr std::size_t CACHE_LINE_SIZE = 64;

// Typecasting the function pointer to void* to avoid the dynamic type check.
// The Dispatch Table is not expected to be used directly by the user and the
// type checks will be done by the user of the Dispatch Table.
using voidFunctorPtr = void*;

/**
 * @brief Thread-safe hash table optimized for query-heavy workloads with rare
 * insertions
 *
 * ThreadSafeChainedDispatchTable implements a hybrid of chaining and open
 * addressing, providing the cache locality benefits of linear probing while
 * maintaining bounded probe distances through segmented chains.
 *
 * DESIGN PHILOSOPHY:
 * - Optimized for query performance (rare inserts, frequent queries)
 * - Segmented open addressing: each hash bucket gets CHAIN_SIZE consecutive
 * slots
 * - Backup overflow table for rare cases when chains fill up
 * - Cache-line aligned entries for optimal memory access patterns
 *
 * MEMORY LAYOUT:
 * - Main table: TABLE_SIZE chains of CHAIN_SIZE entries each
 * - Backup table: TABLE_SIZE entries for overflow handling
 * - Each entry: 64-byte cache-line aligned (key, value, occupied flag)
 * - Total memory: ~((TABLE_SIZE * CHAIN_SIZE + TABLE_SIZE) * 64) bytes
 *
 * @tparam TABLE_SIZE Number of hash buckets (recommend power of 2 for
 * performance)
 * @tparam CHAIN_SIZE Number of consecutive slots per hash bucket (affects cache
 * behavior)
 */
template<std::size_t TABLE_SIZE = 1000, std::size_t CHAIN_SIZE = 20>
class alignas(CACHE_LINE_SIZE) ThreadSafeChainedDispatchTable
{
  public:
    ThreadSafeChainedDispatchTable()
    {
        table.reserve(TABLE_SIZE * CHAIN_SIZE);
        table.resize(TABLE_SIZE * CHAIN_SIZE);
        for (auto& entry : table) {
            entry.isOccupied = false;
        }

        backupTable.reserve(TABLE_SIZE);
        backupTable.resize(TABLE_SIZE);
        for (auto& entry : backupTable) {
            entry.isOccupied = false;
        }
    }

    ~ThreadSafeChainedDispatchTable() = default;

    ThreadSafeChainedDispatchTable(
        ThreadSafeChainedDispatchTable&& other) noexcept
        : table(std::move(other.table))
        , backupTable(std::move(other.backupTable))
    {
        // Let the new mutex be default constructed
    }

    ThreadSafeChainedDispatchTable& operator=(
        ThreadSafeChainedDispatchTable&& other) noexcept
    {
        if (this != &other) {
            table       = std::move(other.table);
            backupTable = std::move(other.backupTable);
            // Leave mutex as is
        }
        return *this;
    }

    ThreadSafeChainedDispatchTable& operator=(
        const ThreadSafeChainedDispatchTable& other) = delete;
    ThreadSafeChainedDispatchTable(
        const ThreadSafeChainedDispatchTable& other) = delete;

    /**
     * @brief Thread-safe insertion of key-value pair with overflow handling
     *
     * Attempts to insert key-value pair into the appropriate hash chain.
     * If the chain is full, falls back to linear search in backup table.
     *
     * PERFORMANCE:
     * - Best case: O(1) - key found in main chain
     * - Average case: O(CHAIN_SIZE) - insert in main chain
     * - Worst case: O(CHAIN_SIZE + TABLE_SIZE) - overflow to backup table
     *
     * THREAD SAFETY:
     * - Fully thread-safe with multiple concurrent readers/single writer
     * - Uses mutex for insertion, lock-free for duplicate detection
     * - Release-acquire ordering ensures visibility to concurrent queries
     *
     * @tparam HASH_KEY_GETTER Callable that extracts hash key from KEY_TYPE*
     * @tparam KEY_COMPARATOR Callable that compares two KEY_TYPE* for equality
     * @tparam KEY_TYPE Type of the key object (pointer stored, not copied)
     * @param key Pointer to key object (must remain valid for table lifetime)
     * @param value Opaque value pointer to associate with key
     * @return Pointer to stored value on success, nullptr on table full
     */
    template<typename HASH_KEY_GETTER,
             typename KEY_COMPARATOR,
             typename KEY_TYPE,
             typename VALUE_REPLACER>
    voidFunctorPtr insert(KEY_TYPE* key, voidFunctorPtr value)
    {
        if ((!key) || (!value)) {
            return nullptr;
        }

        std::size_t index = hash_key(HASH_KEY_GETTER{}(key));

        // Check if the key is already in the chain
        // No need to update the value if the key is already in the chain.
        for (std::size_t i = index * CHAIN_SIZE; i < (index + 1) * CHAIN_SIZE;
             ++i) {
            // This will synchronize with the release operation in the insertion
            // path.
            if (table[i].isOccupied.load(std::memory_order_acquire)) {
                if (KEY_COMPARATOR{}(table[i].key, key)) {
                    // If the key is already in the chain, update the value.
                    if (table[i].value.load(std::memory_order_relaxed)
                        != value) {
                        VALUE_REPLACER{}(
                            table[i].value.load(std::memory_order_relaxed));
                        table[i].value.store(value, std::memory_order_relaxed);
                    }
                    return table[i].value.load(std::memory_order_relaxed);
                }
            }
        }

        // If the key is not in the chain, add it. Double check that
        // the key is not already in the chain
        std::lock_guard<std::mutex> lg{ mtx };
        for (std::size_t i = index * CHAIN_SIZE; i < (index + 1) * CHAIN_SIZE;
             ++i) {
            // No atomic acquire load here because we are inside the lock.
            if (table[i].isOccupied.load(std::memory_order_acquire)) {
                if (KEY_COMPARATOR{}(table[i].key, key)) {
                    // If the key is already in the chain, update the value.
                    if (table[i].value.load(std::memory_order_relaxed)
                        != value) {
                        VALUE_REPLACER{}(
                            table[i].value.load(std::memory_order_relaxed));
                        table[i].value.store(value, std::memory_order_relaxed);
                    }
                    return table[i].value.load(std::memory_order_relaxed);
                }
            } else {
                // The first non-occupied slot is found, so we need to add the
                // key to the first non-occupied slot.
                table[i] = Entry{ key, value, false };
                // This will synchronize with the acquire operation in the query
                // path.
                table[i].isOccupied.store(true, std::memory_order_release);
                return table[i].value;
            }
        }

        // If the chain is full, add the element to a new backup list of limited
        // size.
        for (std::size_t i = 0; i < TABLE_SIZE; ++i) {
            if (backupTable[i].isOccupied.load(std::memory_order_acquire)) {
                // If the key is already in the backup table, update the value.
                if (KEY_COMPARATOR{}(backupTable[i].key, key)) {
                    if (backupTable[i].value.load(std::memory_order_relaxed)
                        != value) {
                        VALUE_REPLACER{}(backupTable[i].value.load(
                            std::memory_order_relaxed));
                        backupTable[i].value.store(value,
                                                   std::memory_order_relaxed);
                    }
                    return backupTable[i].value.load(std::memory_order_relaxed);
                }
            } else {
                backupTable[i] = Entry{ key, value, false };
                // This will synchronize with the acquire operation in the query
                // path.
                backupTable[i].isOccupied.store(true,
                                                std::memory_order_release);
                return backupTable[i].value;
            }
        }

        // If the backup list is also full, return nullptr.
        return nullptr;
    }

    /**
     * @brief Lock-free query for key lookup optimized for high-frequency access
     *
     * Searches for key in main hash chain first, then falls back to backup
     * table if not found. Optimized for query-heavy workloads with minimal
     * latency.
     *
     * PERFORMANCE:
     * - Best case: O(1) - key found in first slot of main chain
     * - Average case: O(CHAIN_SIZE/2) - key found in main chain
     * - Worst case: O(CHAIN_SIZE + TABLE_SIZE) - key in backup table or not
     * found
     *
     * THREAD SAFETY:
     * - Fully lock-free and wait-free for readers
     * - Concurrent with insert operations (release-acquire synchronization)
     * - Multiple concurrent queries supported
     *
     * @tparam HASH_KEY_GETTER Callable that extracts hash key from KEY_TYPE*
     * @tparam KEY_COMPARATOR Callable that compares two KEY_TYPE* for equality
     * @tparam KEY_TYPE Type of the key object
     * @param key Pointer to key object to search for
     * @return Associated value pointer if found, nullptr if not found
     */
    template<typename HASH_KEY_GETTER,
             typename KEY_COMPARATOR,
             typename KEY_TYPE>
    voidFunctorPtr query(KEY_TYPE* key) const
    {
        if (!key) {
            return nullptr;
        }

        std::size_t index    = hash_key(HASH_KEY_GETTER{}(key));
        const auto  startIdx = index * CHAIN_SIZE;
        const auto  endIdx   = (index + 1) * CHAIN_SIZE;

        for (std::size_t i = startIdx; i < endIdx; ++i) {
            // This will synchronize with the release operation in the insertion
            // path.
            if (table[i].isOccupied.load(std::memory_order_acquire)) {
                if (KEY_COMPARATOR{}(table[i].key, key)) {
                    return table[i].value.load(std::memory_order_relaxed);
                }
            }
        }

        for (std::size_t i = 0; i < TABLE_SIZE; ++i) {
            if (backupTable[i].isOccupied.load(std::memory_order_acquire)) {
                if (KEY_COMPARATOR{}(backupTable[i].key, key)) {
                    return backupTable[i].value.load(std::memory_order_relaxed);
                }
            }
        }

        return nullptr;
    }

    /**
     * @brief Extracts all stored values from both main and backup tables
     *
     * Iterates through all table entries and collects value pointers from
     * occupied slots. Primarily used for resource cleanup in destructors.
     *
     * PERFORMANCE:
     * - Time: O(TABLE_SIZE * CHAIN_SIZE + TABLE_SIZE) - scans all entries
     * - Space: O(number of occupied entries) - proportional to actual usage
     * - Cache behavior: Sequential access, cache-friendly
     *
     * THREAD SAFETY:
     * - Lock-free reads with memory_order_acquire
     * - Safe to call concurrently with queries
     * - NOT safe to call concurrently with inserts (use external
     * synchronization)
     *
     * @tparam VALUE_TYPE Type to cast stored value pointers to
     * @return Vector of all stored value pointers cast to VALUE_TYPE*
     */
    template<typename VALUE_TYPE>
    std::vector<VALUE_TYPE*> getValues() const
    {
        std::vector<VALUE_TYPE*> values;
        for (auto& entry : table) {
            if (entry.isOccupied.load(std::memory_order_acquire)) {
                values.push_back(reinterpret_cast<VALUE_TYPE*>(
                    entry.value.load(std::memory_order_relaxed)));
            }
        }
        for (auto& entry : backupTable) {
            if (entry.isOccupied.load(std::memory_order_acquire)) {
                values.push_back(reinterpret_cast<VALUE_TYPE*>(
                    entry.value.load(std::memory_order_relaxed)));
            }
        }
        return values;
    }

    /**
     * @brief Extracts all stored keys from both main and backup tables
     *
     * Iterates through all table entries and collects key pointers from
     * occupied slots. Used for debugging, introspection, and iteration.
     *
     * PERFORMANCE:
     * - Time: O(TABLE_SIZE * CHAIN_SIZE + TABLE_SIZE) - scans all entries
     * - Space: O(number of occupied entries) - proportional to actual usage
     * - Cache behavior: Sequential access, cache-friendly
     *
     * THREAD SAFETY:
     * - Lock-free reads with memory_order_acquire
     * - Safe to call concurrently with queries
     * - NOT safe to call concurrently with inserts (use external
     * synchronization)
     *
     * @tparam KEY_TYPE Type to cast stored key pointers to
     * @return Vector of all stored key pointers cast to KEY_TYPE*
     */
    template<typename KEY_TYPE>
    std::vector<KEY_TYPE*> getKeys() const
    {
        std::vector<KEY_TYPE*> keys;
        for (auto& entry : table) {
            if (entry.isOccupied.load(std::memory_order_acquire)) {
                keys.push_back(reinterpret_cast<KEY_TYPE*>(entry.key));
            }
        }
        for (auto& entry : backupTable) {
            if (entry.isOccupied.load(std::memory_order_acquire)) {
                keys.push_back(reinterpret_cast<KEY_TYPE*>(entry.key));
            }
        }
        return keys;
    }

  private:
    /**
     * @brief Cache-line aligned hash table entry optimized for performance
     *
     * Each entry represents a single key-value pair with atomic occupancy flag.
     * Cache-line alignment prevents false sharing between entries and optimizes
     * memory access patterns for the segmented open addressing design.
     */
    struct alignas(CACHE_LINE_SIZE) Entry
    {
        // Not using std::any because it has an performance overhead due to the
        // dynamic type check.
        void*                       key;
        std::atomic<voidFunctorPtr> value;
        std::atomic<bool>           isOccupied;

        Entry()
            : key(nullptr)
            , value(nullptr)
            , isOccupied(false)
        {
        }

        Entry(void* key, voidFunctorPtr value, bool _isOccupied)
            : key(key)
            , value(value)
            , isOccupied(_isOccupied)
        {
        }

        Entry(const Entry& other)
            : key(other.key)
            , value(other.value.load(std::memory_order_relaxed))
            , isOccupied(other.isOccupied.load(std::memory_order_relaxed))
        {
        }

        Entry(Entry&& other)
            : key(other.key)
            , value(other.value.load(std::memory_order_relaxed))
            , isOccupied(other.isOccupied.load(std::memory_order_relaxed))
        {
        }

        Entry& operator=(const Entry& other)
        {
            key = other.key;
            value.store(other.value.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
            isOccupied = other.isOccupied.load(std::memory_order_relaxed);
            return *this;
        }

        Entry& operator=(Entry&& other)
        {
            key = other.key;
            value.store(other.value.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
            isOccupied.store(other.isOccupied.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
            return *this;
        }
    };

    std::vector<Entry> table;
    std::vector<Entry> backupTable;
    std::mutex         mtx;

    /**
     * @brief Hash function mapping key tuples to table buckets
     *
     * Maps std::tuple<uint64_t, uint64_t> to bucket index [0, TABLE_SIZE).
     * Uses golden ratio constant for good distribution properties.
     *
     * PERFORMANCE: O(1) constexpr evaluation when possible
     *
     * @param key Tuple containing two 64-bit hash components
     * @return Bucket index in range [0, TABLE_SIZE)
     */
    constexpr std::size_t hash_key(
        const std::tuple<uint64_t, uint64_t>& key) const noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed, std::get<0>(key));
        hash_combine(seed, std::get<1>(key));
        return (seed % TABLE_SIZE);
    }

    /**
     * @brief Combines hash values using golden ratio mixing
     *
     * Implements boost-style hash combination with golden ratio constant
     * for optimal bit distribution and avalanche properties.
     *
     * PERFORMANCE: O(1) constexpr evaluation when possible
     *
     * @tparam T Type of value to hash (must be std::hash compatible)
     * @param seed Hash seed to modify (input/output parameter)
     * @param value Value to combine into the hash
     */
    template<typename T>
    constexpr void hash_combine(std::size_t& seed,
                                const T&     value) const noexcept
    {
        std::hash<T> hasher;
        // 0x9e3779b9 is derived from the golden ratio (phi ≈ 1.61803...)
        // multiplied by 2^32 The golden ratio provides mathematically optimal
        // distribution properties The bit shifts (seed<<6, seed>>2) help mix
        // the bits and reduce clustering This combination creates an avalanche
        // effect where small input changes cause significant output
        // differences, minimizing hash collisions
        seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

} // namespace dlp::kernel_frame
