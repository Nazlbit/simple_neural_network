#pragma once
#include <cassert>

class matrix;

class matrix
{
private:
	float* values;
	int width;
	int height;

public:
	matrix() : values(nullptr), width(0), height(0) {}

	matrix(int width, int height);

	matrix(int width, int height, float fill_value);

	matrix(const matrix& m);

	matrix(matrix&& m) noexcept;

	~matrix();

	matrix& operator=(const matrix& m);

	matrix& operator=(matrix&& m) noexcept;

	int get_width() const
	{
		return width;
	}

	int get_height() const
	{
		return height;
	}

	bool is_alive() const
	{
		return values;
	}

	float& at(int row, int column)
	{
		assert(row >= 0 && row < height);
		assert(column >= 0 && column < width);
		return values[(size_t)row * width + column];
	}

	const float& at(int row, int column) const
	{
		assert(row >= 0 && row < height);
		assert(column >= 0 && column < width);
		return values[(size_t)row * width + column];
	}

	float& at(size_t index)
	{
		assert(index < (size_t)height * width);
		return values[index];
	}

	const float& at(size_t index) const
	{
		assert(index < (size_t)height * width);
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

	matrix submatrix(int row_a, int row_b, int column_a, int column_b) const;

	matrix submatrix(int row_a, int row_b) const;

	matrix transpose() const;
};

// Functions that generate another matrix and do not modify the original matrix should be outside of the class

matrix hadamard_product(const matrix& a, const matrix& b);

matrix operator*(const matrix& a, const matrix& b);

matrix operator*(const matrix& m, float v);

inline matrix operator*(float v, const matrix& m)
{
	return m * v;
}

inline matrix operator/(const matrix& m, float v)
{
	return m * (1.f / v);
}

matrix operator/(float v, const matrix& m);

matrix operator+(const matrix& a, const matrix& b);

matrix operator-(const matrix& a, const matrix& b);

matrix operator-(const matrix& m);

matrix operator+(const matrix& m, float v);

inline matrix operator+(float v, const matrix& m)
{
	return m + v;
}

matrix operator-(float v, const matrix& m);

inline matrix operator-(const matrix& m, float v)
{
	return m + -v;
}

matrix transpose(const matrix& m);

matrix submatrix(const matrix& m, int row_a, int row_b, int column_a, int column_b);

matrix sqrt(const matrix& m);