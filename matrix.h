#pragma once
#include <cassert>

class matrix;

class matrix
{
private:
	float* values;
	unsigned width;
	unsigned height;
	bool view;

public:
	matrix() : values(nullptr), width(0), height(0), view(false) {}

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
		return values[row * width + column];
	}

	const float& at(unsigned row, unsigned column) const
	{
		assert(row < height);
		assert(column < width);
		return values[row * width + column];
	}

	float& at(unsigned index)
	{
		assert(index < height * width);
		return values[index];
	}

	const float& at(unsigned index) const
	{
		assert(index < height * width);
		return values[index];
	}

	float* get_data()
	{
		return values;
	}

	const float* get_data() const
	{
		return values;
	}

	matrix submatrix(unsigned row_a, unsigned row_b, unsigned column_a, unsigned column_b) const;

	matrix submatrix(unsigned row_a, unsigned row_b) const;

	matrix transpose() const;
};

// Functions that generate another matrix and do not modify the original matrix should be outside of the class

matrix hadamard_product(const matrix& a, const matrix& b);

matrix operator*(const matrix& a, const matrix& b);

matrix operator*(const matrix& m, float v);

matrix operator/(const matrix& m, float v);

matrix operator+(const matrix& a, const matrix& b);

matrix operator-(const matrix& a, const matrix& b);

matrix operator-(const matrix& m);

matrix operator+(float v, const matrix& m);

matrix operator+(const matrix& m, float v);

matrix operator-(float v, const matrix& m);

matrix operator-(const matrix& m, float v);

matrix transpose(const matrix& m);

matrix submatrix(const matrix& m, unsigned row_a, unsigned row_b, unsigned column_a, unsigned column_b);