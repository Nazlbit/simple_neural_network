#include <chrono>
#include <random>
#include <fstream>

#include "auxiliary.h"

float random_float(float min, float max)
{
	static std::mt19937_64 g1(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> dis(min, max);
	return dis(g1);
}

bool is_little_endian()
{
	static uint16_t value = 0x1;
	static uint8_t* value_ptr = (uint8_t*)&value;
	static bool little_endian = value_ptr[0] == 0x1;
	return little_endian;
}

void swap_byte_order(char* value, size_t size)
{
	char* end = value + size - 1;
	while (value < end)
	{
		char tmp = *value;
		*value = *end;
		*end = tmp;
		value++;
		end--;
	}
}

binary_data read_file(const char* file_name)
{
	std::ifstream f(file_name, std::ios::binary | std::ios::ate);
	if (!f.is_open())
	{
		return binary_data();
	}
	std::streampos size = f.tellg();
	f.seekg(0, std::ios::beg);
	binary_data data(size);
	f.read(reinterpret_cast<char*>(data.get_data()), size);
	return data;
}