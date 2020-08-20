#include "matrix.h"
#include <cstring>
#include <cmath>

matrix::matrix(unsigned width, unsigned height) : width(width), height(height)
{
	assert(width > 0);
	assert(height > 0);

	values = new float[width * height];
}

matrix::matrix(unsigned width, unsigned height, float fill_value) : width(width), height(height)
{
	assert(width > 0);
	assert(height > 0);

	values = new float[width * height];

	for (unsigned long long i = 0; i < (unsigned long long)width * height; i++)
		values[i] = fill_value;
}

matrix::matrix(const matrix& m) : width(m.width), height(m.height)
{
	assert(m.is_alive());
	values = new float[width * height];

	memcpy(values, m.values, width * height * sizeof(float));

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

		values = new float[width * height];
	}

	memcpy(values, m.values, width * height * sizeof(float));

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

matrix submatrix(const matrix& m, unsigned row_a, unsigned row_b, unsigned column_a, unsigned column_b)
{
	assert(row_a < row_b);
	assert(column_a < column_b);

	matrix result(column_b - column_a, row_b - row_a);
	for (int i = row_a; i < row_b; i++)
	{
		for (unsigned j = column_a; j < column_b; j++)
		{
			result.at(i - row_a, j - column_a) = m.at(i, j);
		}
	}
	return result;
}

matrix hadamard_product(const matrix& a, const matrix& b)
{
	assert(a.is_alive() && b.is_alive());
	assert(a.get_width() == b.get_width());
	assert(a.get_height() == b.get_height());

	matrix result(a.get_width(), a.get_height());

	for (unsigned i = 0; i < result.get_height()* a.get_width(); i++)
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

	for (unsigned i = 0; i < result.get_height(); i++)
	{
		for (unsigned j = 0; j < result.get_width(); j++)
		{
			result.at(i, j) = 0;

			for (unsigned k = 0; k < a.get_width(); k++)
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

	for (unsigned i = 0; i < result.get_width()*result.get_height(); i++)
	{
		result.at(i) = m.at(i) * v;
	}

	return result;
}

matrix operator/(const matrix& m, float v)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (unsigned i = 0; i < result.get_width() * result.get_height(); i++)
	{
			result.at(i) = m.at(i) / v;
	}

	return result;
}

matrix operator+(const matrix& a, const matrix& b)
{
	assert(a.is_alive() && b.is_alive());
	assert(a.get_width() == b.get_width());
	assert(a.get_height() == b.get_height());

	matrix result(a.get_width(), a.get_height());

	for (unsigned i = 0; i < result.get_height() * result.get_width(); i++)
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

	for (unsigned i = 0; i < result.get_height() * result.get_width(); i++)
	{
		result.at(i) = a.at(i) - b.at(i);
	}

	return result;
}

matrix operator-(const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (unsigned i = 0; i < result.get_height() * result.get_width(); i++)
	{
		result.at(i) = -m.at(i);
	}

	return result;
}

matrix operator+(float v, const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (unsigned i = 0; i < result.get_height()* result.get_width(); i++)
	{
		result.at(i) = v + m.at(i);
	}

	return result;
}

matrix operator+(const matrix& m, float v)
{
	return v + m;
}

matrix operator-(float v, const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (unsigned i = 0; i < result.get_height() * result.get_width(); i++)
	{
		result.at(i) = v - m.at(i);
	}

	return result;
}

matrix operator-(const matrix& m, float v)
{
	assert(m.is_alive());

	matrix result(m.get_width(), m.get_height());

	for (unsigned i = 0; i < result.get_height() * result.get_width(); i++)
	{
		result.at(i) = m.at(i) - v;
	}

	return result;
}

matrix transpose(const matrix& m)
{
	assert(m.is_alive());

	matrix result(m.get_height(), m.get_width());

	for (unsigned i = 0; i < result.get_height(); i++)
	{
		for (unsigned j = 0; j < result.get_width(); j++)
		{
			result.at(i, j) = m.at(j, i);
		}
	}

	return result;
}
