#include "LSH.h"
#include <string.h>
#include <stddef.h>
#include <thread>
#include <iostream>

LSH::LSH(int K_, int L_, int T_) : counter(0), K(K_), L(L_), THREADS(T_)
{
	for(int idx = 0; idx < L; ++idx)
	{
		std::unordered_map<int, std::vector<int>> table;
		tables.emplace_back(std::move(table));
	}
}

// Insert N elements into LSH hash tables
void LSH::insert_multi(const int* fp, const int N)
{
	std::vector<std::thread> thread_list;
	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { add_multi(fp, N, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}
        counter += N;
}

// Parallel Insert: Each thread handle an independent set of tables - (N x L)
void LSH::add_multi(const int* fp, const int N, const int tdx)
{
	// For each example
	for(int idx = 0; idx < N; ++idx)
	{
		// For each table
		for(int jdx = tdx; jdx < L; jdx+=THREADS)
		{
			add(fp[idx * L + jdx], jdx, idx+counter);
		}
	}
}

// Remove a single element from LSH hash tables
void LSH::remove(const int* fp, const int item_id)
{
	for(int idx = 0; idx < L; ++idx)
	{
		std::vector<int>& bucket = tables[idx][fp[idx]];
		for(size_t jdx = 0; jdx < bucket.size(); ++jdx)
		{
			if(bucket[jdx] == item_id)
			{
				bucket[jdx] = -1;
			}
		}
	}
}

// Insert a single element into LSH hash tables
void LSH::insert(const int* fp, const int item_id)
{
	for(int idx = 0; idx < L; ++idx)
	{
		add(fp[idx], idx, item_id);
	}
}

// Insert a item; Check if bucket exists first
void LSH::add(const int key, const int idx, const int item_id)
{
	std::unordered_map<int, std::vector<int>>& table = tables[idx];
	if(table.find(key) == table.end())
	{
		std::vector<int> value;
		table.emplace(key, std::move(value));
	}
	table[key].push_back(item_id);
}

std::unordered_set<int> LSH::query(const int* fp)
{
	std::unordered_set<int> result;
	for(int idx = 0; idx < L; ++idx)
	{
		retrieve(result, idx, fp[idx]);
	}
	result.erase(-1);
	return result;
}

void LSH::retrieve(std::unordered_set<int>& result, const int table_idx, const int bucket_idx)
{
	std::unordered_map<int, std::vector<int>>& table = tables[table_idx];
	if(table.find(bucket_idx) != table.end())
	{
		const std::vector<int>& bucket = table[bucket_idx];
		result.insert(bucket.begin(), bucket.end());
	}
}

std::unordered_set<int> LSH::query_multi(const int* fp, const int N)
{
	int offset = -1;
	std::unordered_set<int> result;
	for(int idx = 0; idx < N; ++idx)
	{
		for(int jdx = 0; jdx < L; ++jdx)
		{
			retrieve(result, jdx, fp[++offset]);
		}
	}
	result.erase(-1);
	return result;
}

void LSH::query_multi_mask(const int* fp, float* mask, const int M, const int N)
{
	std::vector<std::thread> thread_list;
	for(int tdx = 0; tdx < THREADS; ++tdx)
	{
		std::thread t([=] { retrieve_mask(fp, mask, M, N, tdx); });
		thread_list.emplace_back(std::move(t));
	}

	for(auto& t : thread_list)
	{
		t.join();
	}
}

void LSH::retrieve_mask(const int* fp, float* mask, const int M, const int N, const int tdx)
{

	for(int rnd = tdx; rnd < M; rnd+=THREADS)
	{
		const int fp_offset = rnd * L;
		const int mask_offset = rnd * N;

		std::unordered_set<int> result;
		for(int idx = 0; idx < L; ++idx)
		{
			retrieve(result, idx, fp[fp_offset + idx]);
		}
		result.erase(-1);

		for(int index : result)
		{
			mask[mask_offset + index] = 1.0;
		}
	}
}

std::vector<std::unordered_set<int>> LSH::query_multiset(const int* fp, const int N)
{
        std::vector<std::unordered_set<int>> results;
	int offset = -1;
	for(int idx = 0; idx < N; ++idx)
	{
	        std::unordered_set<int> local_result;
		for(int jdx = 0; jdx < L; ++jdx)
		{
			retrieve(local_result, jdx, fp[++offset]);
		}
	        local_result.erase(-1);
                results.push_back(local_result);
	}
	return results;
}

void LSH::clear()
{
	counter = 0;
	for(int idx = 0; idx < L; ++idx)
	{
		tables[idx].clear();
	}

}
