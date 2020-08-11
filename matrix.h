#pragma once

template<typename T, bool row_major = true>
class matrix
{
private:
	T* values;
	unsigned width;
	unsigned height;

public:
	matrix(unsigned width, unsigned height) : width(width), height(height)
	{
		assert(width > 0);
		assert(height > 0);

		values = new T[width * height];
	}

	matrix(const matrix& m) : width(m.width), height(m.height)
	{
		values = new T[width * height];

		memcpy(values, m.values, width * height * sizeof(T));

	}

	matrix(matrix&& m) noexcept : width(m.width), height(m.height)
	{
		values = m.values;
		m.values = nullptr;
	}

	~matrix()
	{
		if(values) delete[] values;
	}

	matrix& operator=(const matrix& m)
	{
		if (m.width != width || m.height != height)
		{
			if (values) delete[] values;

			width = m.width;
			height = m.height;

			values = new T[width*height];
		}

		memcpy(values, m.values, width * height * sizeof(T));

		return *this;
	}

	matrix& operator=(matrix&& m) noexcept
	{
		if (values) delete[] values;

		width = m.width;
		height = m.height;

		values = m.values;
		m.values = nullptr;

		return *this;
	}

	unsigned get_width() const
	{
		return width;
	}

	unsigned get_height() const
	{
		return height;
	}

	T& at(unsigned row, unsigned column)
	{
		assert(row < height);
		assert(column < width);

		if constexpr (row_major)
			return values[row * width + column];
		else
			return values[column * height + row];
	}

	const T& at(unsigned row, unsigned column) const
	{
		assert(row < height);
		assert(column < width);

		if constexpr (row_major)
			return values[row * width + column];
		else
			return values[column * height + row];
	}

	matrix operator*(const matrix& m) const
	{
		assert(width == m.height);

		matrix result(m.width, height);

		for (unsigned i = 0; i < result.height; i++)
		{
			for (unsigned j = 0; j < result.width; j++)
			{
				result.at(i, j) = 0;

				for (unsigned k = 0; k < width; k++)
				{
					result.at(i, j) += at(i, k) * m.at(k, j);
				}
			}
		}

		return result;
	}

	matrix operator+(matrix m) const
	{
		assert(width == m.width);
		assert(height == m.height);

		if constexpr (row_major)
		{
			for (unsigned i = 0; i < height; i++)
			{
				for (unsigned j = 0; j < width; j++)
				{
					m.at(i, j) = at(i, j) + m.at(i, j);
				}
			}
		}
		else
		{
			for (unsigned j = 0; j < width; j++)
			{
				for (unsigned i = 0; i < height; i++)
				{
					m.at(i, j) = at(i, j) + m.at(i, j);
				}
			}
		}

		return m;
	}
};