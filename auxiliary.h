#pragma once
#include <ostream>

#include "binary_data.h"

float random_float(float min, float max);

bool is_little_endian();

void swap_byte_order(char* value, size_t size);

binary_data read_file(const char* file_name);

template<typename _Elem, typename T>
void write_var(std::basic_ostream<_Elem>& stream, T var)
{
	stream.write(reinterpret_cast<const _Elem*>(&var), sizeof(T));
}

