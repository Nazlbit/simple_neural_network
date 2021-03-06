#include "neural_net.h"
#include "auxiliary.h"

void neural_net::layer::init()
{
	for (int i = 0; i < size; i++)
	{
		biases.at(0, i) = random_float(-1.f, 1.f);

		for (int j = 0; j < prev_layer_size; j++)
		{
			weights.at(j, i) = random_float(-1.f, 1.f);
		}
	}
}

neural_net::neural_net(const int num_layers, const int* const layer_sizes)
{
	assert(num_layers > 1);
	assert(layer_sizes != nullptr);
	assert(layer_sizes[0] > 0);

	input_layer_size = layer_sizes[0];

	layers.reserve(num_layers - 1);

	//Init layers
	for (int i = 1; i < num_layers; i++)
	{
		assert(layer_sizes[i] > 0);

		layers.emplace_back(layer_sizes[i], layer_sizes[i - 1]);
		layers.back().init();
	}
}

neural_net::neural_net(const char* const file_name)
{
	load_from_file(file_name);
}

matrix neural_net::run(matrix input) const
{
	assert(input.get_width() == input_layer_size);

	for (const layer& l : layers)
	{
		input = activation_function(input * l.weights + matrix(1, input.get_height(), 1.f) * l.biases);
	}

	return input;
}

std::vector<matrix> neural_net::run_ext_output(matrix input) const
{
	assert(input.get_width() == input_layer_size);

	std::vector<matrix> result;
	result.reserve(layers.size() + 1);
	result.push_back(std::move(input));

	for (const layer& l : layers)
	{
		result.push_back(activation_function(result.back() * l.weights + matrix(1, input.get_height(), 1.f) * l.biases));
	}

	return result;
}

std::vector<neural_net::layer> neural_net::backpropagation(const matrix& input, const matrix& required_output)
{
	assert(input.is_alive() && required_output.is_alive());
	assert(input.get_width() == input_layer_size);
	assert(required_output.get_width() == layers.back().size);
	assert(required_output.get_height() == input.get_height());

	std::vector<matrix> values = run_ext_output(input); // Calculate initial neurons activation values
	std::vector<layer> gradient(layers.size());

	matrix x = values.back() - required_output; // Delta

	for (int i = layers.size(); i > 0; i--) // For every layer starting from the last
	{
		x = hadamard_product(x, activation_function_derivative(values[i])); // Activation function derivative
		gradient[i - 1].size = layers[i - 1].size;
		gradient[i - 1].prev_layer_size = layers[i - 1].prev_layer_size;
		gradient[i - 1].weights = transpose(values[i - 1]) * x; // Weights partial derivative
		gradient[i - 1].biases = matrix(input.get_height(), 1, 1.f) * x; // Biases partial derivative
		x = transpose(layers[i - 1].weights * transpose(x)); // Neuron connection partial derivative
	}

	return gradient;
}

void neural_net::backpropagation(const matrix& input, const matrix& required_output, float rate)
{
	assert(input.is_alive() && required_output.is_alive());
	assert(input.get_width() == input_layer_size);
	assert(required_output.get_width() == layers.back().size);
	assert(required_output.get_height() == input.get_height());

	std::vector<matrix> values = run_ext_output(input); // Calculate initial neurons activation values

	matrix x = values.back() - required_output; // Delta

	for (int i = layers.size(); i > 0; i--) // For every layer starting from the last
	{
		x = hadamard_product(x, activation_function_derivative(values[i])); // Activation function derivative
		matrix weights_derivative = transpose(values[i - 1]) * x; // Weights partial derivative
		layers[i - 1].biases = layers[i - 1].biases - matrix(input.get_height(), 1, 1.f) * x * rate; // Biases partial derivative
		x = transpose(layers[i - 1].weights * transpose(x)); // Neuron connection partial derivative
		layers[i - 1].weights = layers[i - 1].weights - weights_derivative * rate;
	}
}

void neural_net::train_batch(const matrix& input, const matrix& required_output, int iter_num, float rate)
{
	assert(input.is_alive() && required_output.is_alive());
	assert(input.get_width() == input_layer_size);
	assert(required_output.get_width() == layers.back().size);
	assert(required_output.get_height() == input.get_height());
	assert(rate > 0);
	assert(iter_num > 0);

	for (int i = 0; i < iter_num; i++)
	{
		backpropagation(input, required_output, rate);
	}
}

void neural_net::train_stochastic(const matrix& input, const matrix& required_output, int iter_num, float rate)
{
	assert(input.is_alive() && required_output.is_alive());
	assert(input.get_width() == input_layer_size);
	assert(required_output.get_width() == layers.back().size);
	assert(required_output.get_height() == input.get_height());
	assert(rate > 0);
	assert(iter_num > 0);

	const int num_samples = input.get_height();

	static std::vector<layer> grad = backpropagation(input.submatrix(0, 1), required_output.submatrix(0, 1));

	const float fraction = 0.7f;
	for (int i = 0; i < iter_num; i++)
	{
		const unsigned sample_index = random_int(0, num_samples - 1);
		auto new_grad = backpropagation(input.submatrix(sample_index, sample_index+1), required_output.submatrix(sample_index, sample_index+1));

		for (int j = 0; j < (int)layers.size(); j++)
		{
			grad[j].weights = grad[j].weights * fraction + new_grad[j].weights * rate;
			grad[j].biases = grad[j].biases * fraction + new_grad[j].biases * rate;

			layers[j].weights = layers[j].weights - grad[j].weights;
			layers[j].biases = layers[j].biases - grad[j].biases;
		}
	}
}

void neural_net::train_mini_batch(const std::vector<matrix>& input, const std::vector<matrix>& required_output, int iter_num, float rate)
{
	assert(input.size() == required_output.size());
	assert(rate > 0);
	assert(iter_num > 0);
	
	const int num_batches = (int)input.size();

	for (int i = 0; i < iter_num; i++)
	{
		const int batch_index = random_int(0, num_batches-1);
		backpropagation(input[batch_index], required_output[batch_index], rate);
	}
}

bool neural_net::save_to_file(const char* const file_name)
{
	std::ofstream f(file_name, std::ios::binary | std::ios::trunc);
	if (!f.is_open()) return false;
	//Magic number
	write_var(f, (uint32_t)0x00230298u);
	//Is little endian
	write_var(f, is_little_endian());
	//Number of layers
	write_var(f, int32_t(layers.size() + 1));
	//Input layer size
	write_var(f, (int32_t)input_layer_size);
	//Other layers sizes
	for (const layer& l : layers)
		write_var(f, (int32_t)l.size);
	//Weights and biases
	for (const layer& l : layers)
	{
		//Weights
		f.write(reinterpret_cast<const char*>(l.weights.get_data()),
			(std::streamsize)l.weights.get_width() * l.weights.get_height() * sizeof(float));
		//Biases
		f.write(reinterpret_cast<const char*>(l.biases.get_data()),
			(std::streamsize)l.biases.get_width() * sizeof(float));
	}

	return true;
}

bool neural_net::load_from_file(const char* const file_name)
{
	binary_data net_data = read_file(file_name);

	if (!net_data.get_data()) return false;
	char* file_pointer = net_data.get_data();

	//Magic number
	const uint32_t magic_number = *reinterpret_cast<unsigned*>(file_pointer);
	file_pointer += sizeof(unsigned);

	if (magic_number != (uint32_t)0x00230298 && magic_number != (uint32_t)0x98022300)
		return false;

	const bool little_endian = *reinterpret_cast<bool*>(file_pointer);
	file_pointer += sizeof(bool);

	//If endianness doesn't match
	if (little_endian != is_little_endian())
	{
		assert((net_data.get_data() + net_data.get_size() - file_pointer) % 4 == 0);
		for (char* i = file_pointer; i < net_data.get_data() + net_data.get_size(); i += 4)
		{
			swap_byte_order(i, 4);
		}
	}

	const int32_t num_layers = *reinterpret_cast<int32_t*>(file_pointer);
	file_pointer += 4;

	const int32_t* layers_sizes = reinterpret_cast<int32_t*>(file_pointer);
	file_pointer += 4 * num_layers;

	input_layer_size = layers_sizes[0];

	layers.clear();
	layers.reserve(num_layers - 1);

	for (int i = 1; i < num_layers; i++)
	{
		layer& l = layers.emplace_back(layers_sizes[i], layers_sizes[i - 1]);

		const size_t weights_size = (size_t)l.weights.get_width() * l.weights.get_height() * sizeof(float);;
		memcpy(l.weights.get_data(), file_pointer, weights_size);
		file_pointer += weights_size;

		const size_t biases_size = (size_t)l.biases.get_width() * sizeof(float);
		memcpy(l.biases.get_data(), file_pointer, biases_size);
		file_pointer += biases_size;
	}

	return true;
}

float neural_net::calculate_error(matrix values, matrix required_values)
{
	matrix delta = required_values - values;

	float delta_sqr_sum = 0;
	for (size_t i = 0; i < (size_t)delta.get_height() * delta.get_width(); i++)
	{
		delta_sqr_sum += delta.at(i) * delta.at(i);
	}
	delta_sqr_sum /= delta.get_height();
	return delta_sqr_sum;
}

matrix neural_net::activation_function(matrix input)
{
	for (size_t i = 0; i < (size_t)input.get_height() * input.get_width(); i++)
	{
		input.at(i) = sigmoid(input.at(i));
	}
	return input;
}

matrix neural_net::activation_function_derivative(matrix input)
{
	for (size_t i = 0; i < (size_t)input.get_height() * input.get_width(); i++)
	{
		input.at(i) = sigmoid_derivative(input.at(i));
	}
	return input;
}
