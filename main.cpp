#include <chrono>
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

#include "matrix.h"

float random_float(float min, float max)
{
	static std::mt19937_64 g1(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> dis(min, max);
	return dis(g1);
}

struct neural_net
{
	struct layer
	{
		const unsigned size;
		matrix weights;
		matrix biases;

		layer(unsigned size, unsigned prev_layer_size) : size(size), biases(size, 1, 0.f), weights(size, prev_layer_size, 0.f) 
		{ 
			init(); 
		}
		void init()
		{
			for (unsigned i = 0; i < size; i++)
			{
				biases.at(0, i) = random_float(-1.f, 1.f);

				for (unsigned j = 0; j < weights.get_height(); j++)
				{
					weights.at(j, i) = random_float(-1.f, 1.f);
				}
			}
		}
	};

	std::vector<layer> layers;
	unsigned input_layer_size;

	neural_net(const unsigned num_layers, const unsigned* const layer_sizes)
	{
		assert(num_layers > 1);
		assert(layer_sizes != nullptr);
		
		input_layer_size = layer_sizes[0];

		layers.reserve(num_layers - 1);

		//Init layers
		for (unsigned i = 1; i < num_layers; i++)
		{
			layers.emplace_back(layer_sizes[i], layer_sizes[i - 1]);
		}
	}

	static float sigmoid(float input)
	{
		return 1.f / (1.f + expf(-input));
	}

	static float sigmoid_derivative(float input)
	{
		return input * (1 - input);
	}

	static float relu(float input)
	{
		return input > 0 ? input : 0;
	}

	static float relu_derivative(float input)
	{
		return input > 0 ? 1 : 0;
	}

	static matrix activation_function(matrix input)
	{
		for (unsigned i = 0; i < input.get_height()*input.get_width(); i++)
		{
			input.at(i) = sigmoid(input.at(i));
		}
		return input;
	}

	static matrix activation_function_derivative(matrix input)
	{
		for (unsigned i = 0; i < input.get_height() * input.get_width(); i++)
		{
			input.at(i) = sigmoid_derivative(input.at(i));
		}
		return input;
	}

	matrix run(const matrix& input) const
	{
		assert(input.get_width() == input_layer_size);

		matrix result = input;

		for (const layer& l : layers)
		{
			result = activation_function(result * l.weights + matrix(1, input.get_height(), 1.f) * l.biases);
		}

		return result;
	}

	std::vector<matrix> run_ext_output(const matrix& input) const
	{
		assert(input.get_width() == input_layer_size);

		std::vector<matrix> result;
		result.reserve(layers.size() + 1);
		result.push_back(input);

		for (const layer& l : layers)
		{
			result.push_back(activation_function(result.back() * l.weights + matrix(1, input.get_height(), 1.f)*l.biases));
		}

		return result;
	}

	void backpropagation(const matrix& input, const matrix& required_output, float rate)
	{
		assert(input.is_alive() && required_output.is_alive());
		assert(input.get_width() == input_layer_size);
		assert(required_output.get_width() == layers.back().size);
		assert(required_output.get_height() == input.get_height());
		assert(rate > 0);

		std::vector<matrix> values = run_ext_output(input); // Calculate initial neurons activation values

		float delta_square_sum = 0;

		matrix x = values.back() - required_output; // Delta

		for (unsigned i = layers.size(); i > 0; i--) // For every layer starting from the last
		{
			x = hadamard_product(x, activation_function_derivative(values[i]));

			layers[i - 1].biases = layers[i - 1].biases - matrix(input.get_height(), 1, 1.f)*x*rate; // Calculate new bias

			matrix weights_derivatives = transpose(values[i - 1]) * x; // Weights partial derivative

			x = transpose(layers[i - 1].weights * transpose(x)); // Neuron connection partial derivative

			layers[i - 1].weights = layers[i - 1].weights - weights_derivatives * rate; // Calculate new weights
		}
	}
};

float calculate_error(matrix values, matrix required_values)
{
	matrix delta = required_values - values;

	float delta_sqr_sum = 0;
	unsigned size = delta.get_height() * delta.get_width();
	for (int i = 0; i < size; i++)
	{
		delta_sqr_sum += delta.at(i) * delta.at(i);
	}
	delta_sqr_sum /= delta.get_height();
	return delta_sqr_sum;
}

void print(const matrix& values)
{
	for (unsigned j = 0; j < values.get_height(); j++)
	{
		for (unsigned i = 0; i < values.get_width(); i++)
		{
			std::cout << values.at(j, i) << " ";
		}
		std::cout << std::endl;
	}
}

class binary_data
{
	std::byte* data = nullptr;
	unsigned long long size = 0;
public:
	binary_data() {}
	binary_data(unsigned long long size) : size(size)
	{
		assert(size > 0);
		data = new std::byte[size];
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
			data = new std::byte[size];
			memcpy(data, bd.data, size);
		}
	}
	~binary_data()
	{
		if(data) delete[] data;
	}

	binary_data& operator=(const binary_data& bd)
	{
		if (data) delete[] data;
		size = bd.size;
		if (size > 0)
		{
			data = new std::byte[size];
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
	std::byte* get_data() const
	{
		return data;
	}
	unsigned long long get_size() const
	{
		return size;
	}
};

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

bool is_little_endian()
{
	uint16_t value = 0x1;
	uint8_t* value_ptr = (uint8_t*)&value;
	return (value_ptr[0] == 0x1);
}

void digits()
{
	//Load data

	struct
	{
		binary_data data;
		int32_t magic_number;
		int32_t num_of_items;
	} labels;
	struct
	{
		binary_data data;
		int32_t magic_number;
		int32_t num_of_items;
		int32_t num_of_rows;
		int32_t num_of_columns;
		
	} images;

	//Data source is http://yann.lecun.com/exdb/mnist/
	labels.data = read_file("train-labels.idx1-ubyte");
	images.data = read_file("train-images.idx3-ubyte");

	if (!labels.data.get_data() || !images.data.get_data())
	{
		std::cout << "ERROR: couldn't load data!\n";
		return;
	}

	labels.magic_number = *(int32_t*)labels.data.get_data();
	labels.num_of_items = *(int32_t*)(labels.data.get_data() + 4);

	images.magic_number = *(int32_t*)images.data.get_data();
	images.num_of_items = *(int32_t*)(images.data.get_data() + 4);
	images.num_of_rows = *(int32_t*)(images.data.get_data() + 8);
	images.num_of_columns = *(int32_t*)(images.data.get_data() + 12);

	if (is_little_endian())
	{
		swap_byte_order((char*)&labels.magic_number, sizeof(int32_t));
		swap_byte_order((char*)&labels.num_of_items, sizeof(int32_t));
		swap_byte_order((char*)&images.magic_number, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_items, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_rows, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_columns, sizeof(int32_t));
	}

	if (labels.magic_number != 0x00000801 || images.magic_number != 0x00000803)
	{
		std::cout << "ERROR: magic numbers don't match!\n";
		return;
	}

	if (labels.num_of_items != images.num_of_items)
	{
		std::cout << "ERROR: data sizes don't match\n";
		return;
	}

	//Prepare data

	const unsigned input_layer_size = images.num_of_rows * images.num_of_columns;
	const unsigned items_num = 10000; //Limit the number of items for faster training

	matrix input(input_layer_size, items_num);

	matrix required_output(10, items_num, 0.f);

	matrix input_test(input_layer_size, 100);

	matrix required_output_test(10, 100, 0.f);

	const float mul = 1.f / 255;

	//Train data
	for (int32_t i = 0; i < items_num; i++)
	{
		//Input layer

		std::byte* img_start_addr = images.data.get_data() + i * input_layer_size + 16;
		
		for (unsigned j = 0; j < input_layer_size; j++)
		{
			input.at(i, j) = *reinterpret_cast<uint8_t*>(img_start_addr + j) * mul;
		}

		//Output layer

		required_output.at(i, *reinterpret_cast<uint8_t*>(labels.data.get_data() + i + 8)) = 1.f;
	}

	// Test data
	for (int32_t i = 0; i < 100; i++)
	{
		//Input layer

		std::byte* img_start_addr = images.data.get_data() + (i + items_num) * input_layer_size + 16;

		for (unsigned j = 0; j < input_layer_size; j++)
		{
			input_test.at(i, j) = *reinterpret_cast<uint8_t*>(img_start_addr + j) * mul; //Shifted by items_num images
		}

		//Output layer

		required_output_test.at(i, *reinterpret_cast<uint8_t*>(labels.data.get_data() + items_num + i + 8)) = 1.f;
	}

	const unsigned num_layers = 4;
	const unsigned layer_sizes[num_layers] = {input_layer_size, 16, 16, 10};
	neural_net net(num_layers, layer_sizes);

	for (int i = 0; i < 1000; i++)
	{
		std::cout << "Iteration #" << i << std::endl;
		net.backpropagation(input, required_output, 10.f/items_num);

		std::cout << calculate_error(net.run(input_test), required_output_test) << std::endl;
	}

	std::cout << "TEST:\n";

	matrix result = net.run(input_test);
	print(result);
	std::cout << std::endl;
	print(required_output_test);

	std::cout << calculate_error(result, required_output_test) << std::endl;
}

void simple_example()
{
	const unsigned num_layers = 4;
	const unsigned layer_sizes[num_layers] = { 3, 4, 3, 2 };

	neural_net test(num_layers, layer_sizes);

	matrix input(3, 1);
	input.at(0, 0) = 0.5f;
	input.at(0, 1) = 0.2f;
	input.at(0, 2) = 0.8f;

	matrix required_output(2, 1);
	required_output.at(0, 0) = 1.f;
	required_output.at(0, 1) = 0.f;

	// Print initial output
	std::cout << "Initial output:" << std::endl;
	print(test.run(input));
	std::cout << std::endl;

	// Run backpropagation
	for (int i = 0; i < 100; i++)
	{
		test.backpropagation(input, required_output, 0.02f);
	}

	// Calculate Sum of square deltas
	matrix result = test.run(input);
	matrix delta = required_output - result;
	float delta_square_sum = (delta * transpose(delta)).at(0, 0);

	// Print results
	std::cout << "Num iterations: " << 10 << std::endl;
	std::cout << "Sum of squared deltas : " << delta_square_sum << std::endl;
	print(result);
}

int main()
{
	//simple_example();
	digits();

	return 0;
}
