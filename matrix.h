#pragma once
#include <cassert>

class matrix
{
private:
	float* values;
	unsigned width;
	unsigned height;

public:
	matrix(unsigned width, unsigned height);

	matrix(unsigned width, unsigned height, float fill_value);

	matrix(const matrix& m);

	matrix(matrix&& m) noexcept;

	~matrix();

	matrix& operator=(const matrix& m);

	matrix& operator=(matrix&& m) noexcept;

	unsigned get_width() const
	{
		return width;
	}

	unsigned get_height() const
	{
		return height;
	}

	bool is_alive() const
	{
		return values;
	}

	float& at(unsigned row, unsigned column)
	{
		assert(row < height);
		assert(column < width);
		return values[column * height + row];
	}

	const float& at(unsigned row, unsigned column) const
	{
		assert(row < height);
		assert(column < width);
		return values[column * height + row];
	}
};

matrix hadamard_product(const matrix& a, const matrix& b);

matrix operator*(const matrix& a, const matrix& b);

matrix operator*(const matrix& m, float v);

matrix operator+(const matrix& a, const matrix& b);

matrix operator-(const matrix& a, const matrix& b);

matrix operator-(const matrix& m);

matrix operator+(float v, const matrix& m);

matrix operator+(const matrix& m, float v);

matrix operator-(float v, const matrix& m);

matrix operator-(const matrix& m, float v);

matrix transpose(const matrix& m);
