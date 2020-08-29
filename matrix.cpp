#include "matrix.h"
#include <cstring>
#include <cmath>

matrix::matrix(int width, int height) : width(width), height(height)
{
	assert(width > 0);
	assert(height > 0);

	values = new float[width * height];
}

matrix::matrix(int width, int height, float fill_value) : width(width), height(height)
{
	assert(width > 0);
	assert(height > 0);

	values = new float[(size_t)width * height];

	for (size_t i = 0; i < (size_t)width * height; i++)
		values[i] = fill_value;
}

matrix::matrix(const matrix& m) : width(m.width), height(m.height)
{
	assert(m.is_alive());
	values = new float[(size_t)width * height];

	memcpy(values, m.values, (size_t)width * height * sizeof(float));
}

matrix::matrix(matrix&& m) noexcept : width(m.width), height(m.height)
{
	assert(m.is_alive());
	values = m.values;
	m.values = nullptr;
}

matrix::~matrix()
{
	if (values) delete[] values;
}

matrix& matrix::operator=(const matrix& m)
{
	assert(m.is_alive());

	if (m.width != width || m.height != height)
	{
		if (values) delete[] values;

		width = m.width;
		height = m.height;

		values = new float[(size_t)width * height];
	}

	memcpy(values, m.values, (size_t)width * height * sizeof(float));

	return *this;
}

matrix& matrix::operator=(matrix&& m) noexcept
{
	assert(m.is_alive());

	if (values) delete[] values;

	width = m.width;
	height = m.height;

	values = m.values;
	m.values = nullptr;

	return *this;
}

matrix matrix::submatrix(int row_a, int row_b, int column_a, int column_b) const
{
	assert(row_a >= 0 && row_a < row_b);
	assert(column_a >= 0 && column_a < column_b);

	matrix result(column_b - column_a, row_b - row_a);
	for (int i = row_a; i < row_b; i++)
	{
		for (int j = column_a; j < column_b; j++)
		{
			result.at(i - row_a, j - column_a) = at(i, j);
		}
	}
	return result;
}

matrix matrix::submatrix(int row_a, int row_b) const
{
	assert(row_a >= 0 && row_a < row_b);

	const int new_height = row_b - row_a;

	matrix result(width, new_height);
	memcpy(result.get_data(), values + row_a * width, (size_t)new_height * width * sizeof(float));
	return result;
}

matrix submatrix(const matrix& m, int row_a, int row_b, int column_a, int column_b)
{
	assert(row_a >= 0 && row_a < row_b);
	assert(column_a >= 0 && column_a < column_b);

	matrix result(column_b - column_a, row_b - row_a);
	for (int i = row_a; i < row_b; i++)
	{
		for (int j = column_a; j < column_b; j++)
		{
			result.at(i - row_a, j - column_a) = m.at(i, j);
		}
	}
	return result;
}

matrix sqrt(const matrix& m)
{
	matrix result(m.get_width(), m.get_height());
	for (size_t i = 0; i < (size_t)result.get_width() * result.get_height(); i++)
	{
		result.at(i) = sqrtf(m.at(i));
	}

	return result;
}

matrix hadamard_product(const matrix& a, const matrix& b)
{
	assert(a.is_alive() && b.is_alive());
	assert(a.get_width() == b.get_width());
	assert(a.get_height() == b.get_height());

	matrix result(a.get_width(), a.get_height());

	for (size_t i = 0; i < (size_t)result.get_height() * a.get_width(); i++)
	{
		result.at(i) = a.at(i) * b.at(i);
	}

	return result;
}

matrix operator*(const matrix& a, const matrix& b)
{
	assert(a.is_alive() && b.is_alive());
	assert(a.get_width() == b.get_height());
	
	matrix result(b.get_width(), a.get_height());

	for (int i = 0; i < result.get_height(); i++)
	{
		for (int j = 0; j < result.get_width(); j++)
		{
			result.at(i, j) = 0;

			for (int k = 0; k < a.get_width(); k++)
			{
				result.at(i, j) += a.at(i, k) * b.at(k, j);
			}
		}
	}

	return result;
}

matrix operator*(const matrix& m, float v)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (size_t i = 0; i < (size_t)result.get_width()*result.get_height(); i++)
	{
		result.at(i) = m.at(i) * v;
	}

	return result;
}

matrix operator/(float v, const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (size_t i = 0; i < (size_t)result.get_width() * result.get_height(); i++)
	{
		result.at(i) = v / m.at(i);
	}

	return result;
}

matrix operator+(const matrix& a, const matrix& b)
{
	assert(a.is_alive() && b.is_alive());
	assert(a.get_width() == b.get_width());
	assert(a.get_height() == b.get_height());

	matrix result(a.get_width(), a.get_height());

	for (size_t i = 0; i < (size_t)result.get_height() * result.get_width(); i++)
	{
		result.at(i) = a.at(i) + b.at(i);
	}

	return result;
}

matrix operator-(const matrix& a, const matrix& b)
{
	assert(a.is_alive() && b.is_alive());
	assert(a.get_width() == b.get_width());
	assert(a.get_height() == b.get_height());

	matrix result(a.get_width(), a.get_height());

	for (size_t i = 0; i < (size_t)result.get_height() * result.get_width(); i++)
	{
		result.at(i) = a.at(i) - b.at(i);
	}

	return result;
}

matrix operator-(const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (size_t i = 0; i < (size_t)result.get_height() * result.get_width(); i++)
	{
		result.at(i) = -m.at(i);
	}

	return result;
}

matrix operator+(const matrix& m, float v)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (size_t i = 0; i < (size_t)result.get_height() * result.get_width(); i++)
	{
		result.at(i) = m.at(i) + v;
	}

	return result;
}

matrix operator-(float v, const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (size_t i = 0; i < (size_t)result.get_height() * result.get_width(); i++)
	{
		result.at(i) = v - m.at(i);
	}

	return result;
}

matrix transpose(const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_height(), m.get_width());

	for (int i = 0; i < result.get_height(); i++)
	{
		for (int j = 0; j < result.get_width(); j++)
		{
			result.at(i, j) = m.at(j, i);
		}
	}

	return result;
}

matrix matrix::transpose() const
{
	assert(is_alive());

	matrix result(get_height(), get_width());

	for (int i = 0; i < result.get_height(); i++)
	{
		for (int j = 0; j < result.get_width(); j++)
		{
			result.at(i, j) = at(j, i);
		}
	}

	return result;
}
