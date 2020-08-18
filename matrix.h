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

	float& at(unsigned index)
	{
		assert(index < height* width);
		return values[index];
	}

	const float& at(unsigned index) const
	{
		assert(index < height* width);
		return values[index];
	}

	matrix get_submatrix(unsigned row_a, unsigned column_a, unsigned row_b, unsigned column_b) const;

	float* get_data()
	{
		return values;
	}

	const float* get_data() const
	{
		return values;
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
