#include "neural_net.h"
#include "auxiliary.h"

void neural_net::layer::init()
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

neural_net::neural_net(const unsigned num_layers, const unsigned* const layer_sizes)
{
	assert(num_layers > 1);
	assert(layer_sizes != nullptr);

	input_layer_size = layer_sizes[0];

	layers.reserve(num_layers - 1);

	//Init layers
	for (unsigned i = 1; i < num_layers; i++)
	{
		layers.emplace_back(layer_sizes[i], layer_sizes[i - 1]);
		layers.back().init();
	}
}

neural_net::neural_net(const char* const file_name)
{
	load_from_file(file_name);
}

matrix neural_net::run(const matrix& input) const
{
	assert(input.get_width() == input_layer_size);

	matrix result = input;

	for (const layer& l : layers)
	{
		result = activation_function(result * l.weights + matrix(1, input.get_height(), 1.f) * l.biases);
	}

	return result;
}

std::vector<matrix> neural_net::run_ext_output(const matrix& input) const
{
	assert(input.get_width() == input_layer_size);

	std::vector<matrix> result;
	result.reserve(layers.size() + 1);
	result.push_back(input);

	for (const layer& l : layers)
	{
		result.push_back(activation_function(result.back() * l.weights + matrix(1, input.get_height(), 1.f) * l.biases));
	}

	return result;
}

void neural_net::backpropagation(const matrix& input, const matrix& required_output, float rate)
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

		layers[i - 1].biases = layers[i - 1].biases - matrix(input.get_height(), 1, 1.f) * x * rate; // Calculate new bias

		matrix weights_derivatives = transpose(values[i - 1]) * x; // Weights partial derivative

		x = transpose(layers[i - 1].weights * transpose(x)); // Neuron connection partial derivative

		layers[i - 1].weights = layers[i - 1].weights - weights_derivatives * rate; // Calculate new weights
	}
}

bool neural_net::save_to_file(const char* const file_name)
{
	std::ofstream f(file_name, std::ios::binary | std::ios::trunc);
	if (!f.is_open()) return false;
	//Magic number
	write_var(f, 0x00230298u);
	//Is little endian
	write_var(f, is_little_endian());
	//Number of layers
	write_var(f, unsigned(layers.size() + 1));
	//Input layer size
	write_var(f, input_layer_size);
	//Other layers sizes
	for (const layer& l : layers)
		write_var(f, l.size);
	//Weights and biases
	for (const layer& l : layers)
	{
		//Weights
		f.write(reinterpret_cast<const char*>(l.weights.get_data()),
			(uint64_t)l.weights.get_width() * l.weights.get_height() * sizeof(float));
		//Biases
		f.write(reinterpret_cast<const char*>(l.biases.get_data()),
			(uint64_t)l.biases.get_width() * sizeof(float));
	}

	return true;
}

bool neural_net::load_from_file(const char* const file_name)
{
	binary_data net_data = read_file(file_name);

	if (!net_data.get_data()) return false;
	char* file_pointer = net_data.get_data();

	//Magic number
	const unsigned magic_number = *reinterpret_cast<unsigned*>(file_pointer);
	file_pointer += sizeof(unsigned);

	if (magic_number != 0x00230298u && magic_number != 0x98022300u)
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

	const unsigned num_layers = *reinterpret_cast<unsigned*>(file_pointer);
	file_pointer += 4;

	const unsigned* layers_sizes = reinterpret_cast<unsigned*>(file_pointer);
	file_pointer += 4 * num_layers;

	input_layer_size = layers_sizes[0];

	layers.clear();
	layers.reserve(num_layers - 1);

	for (unsigned i = 1; i < num_layers; i++)
	{
		layer& l = layers.emplace_back(layers_sizes[i], layers_sizes[i - 1]);

		const unsigned weights_size = l.weights.get_width() * l.weights.get_height() * sizeof(float);;
		memcpy(l.weights.get_data(), file_pointer, weights_size);
		file_pointer += weights_size;

		const unsigned biases_size = l.biases.get_width() * sizeof(float);
		memcpy(l.biases.get_data(), file_pointer, biases_size);
		file_pointer += biases_size;
	}

	return true;
}

float neural_net::calculate_error(matrix values, matrix required_values)
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

matrix neural_net::activation_function(matrix input)
{
	for (unsigned i = 0; i < input.get_height() * input.get_width(); i++)
	{
		input.at(i) = sigmoid(input.at(i));
	}
	return input;
}

matrix neural_net::activation_function_derivative(matrix input)
{
	for (unsigned i = 0; i < input.get_height() * input.get_width(); i++)
	{
		input.at(i) = sigmoid_derivative(input.at(i));
	}
	return input;
}
