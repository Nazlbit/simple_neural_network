#pragma once
#include <cassert>
#include <cstring>

class binary_data
{
	char* data = nullptr;
	unsigned long long size = 0;
public:
	binary_data() {}
	binary_data(unsigned long long size) : size(size)
	{
		assert(size > 0);
		data = new char[size];
	}
	binary_data(binary_data&& bd) noexcept
	{
		size = bd.size;
		data = bd.data;
		bd.data = nullptr;
		bd.size = 0;
	}
	binary_data(const binary_data& bd)
	{
		size = bd.size;
		if (size > 0)
		{
			data = new char[size];
			memcpy(data, bd.data, size);
		}
	}
	~binary_data()
	{
		if (data) delete[] data;
	}

	binary_data& operator=(const binary_data& bd)
	{
		if (data) delete[] data;
		size = bd.size;
		if (size > 0)
		{
			data = new char[size];
			memcpy(data, bd.data, size);
		}
		return *this;
	}
	binary_data& operator=(binary_data&& bd) noexcept
	{
		if (data) delete[] data;
		size = bd.size;
		data = bd.data;
		bd.data = nullptr;
		bd.size = 0;
		return *this;
	}
	char* get_data() const
	{
		return data;
	}
	unsigned long long get_size() const
	{
		return size;
	}
};