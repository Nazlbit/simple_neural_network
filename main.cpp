#include <iostream>
#include <chrono>

#include "neural_net.h"
#include "auxiliary.h"

void print(const matrix& values)
{
	for (unsigned j = 0; j < values.get_height(); j++)
	{
		for (unsigned i = 0; i < values.get_width(); i++)
		{
			std::cout << values.at(j, i) << " ";
		}
		std::cout << '\n';
	}
}

void print_image(const float* data, unsigned width, unsigned height)
{
	for (unsigned i = 0; i < height; i++)
	{
		for (unsigned j = 0; j < width; j++)
		{
			if (data[i * width + j] > 0.5)
			{
				std::cout << (char)178 << ' ';
			}
			else
			{
				std::cout << "  ";
			}
		}
		std::cout << '\n';
	}
}

float calculate_error(const matrix& output, const matrix& required_output)
{
	assert(output.get_height() == required_output.get_height());
	assert(output.get_width() == required_output.get_width());

	float error = 0;

	for (unsigned j = 0; j < output.get_height(); j++)
	{
		unsigned top = 0;
		unsigned required_top = 0;
		for (unsigned k = 1; k < output.get_width(); k++)
		{
			if (output.at(j, top) < output.at(j, k)) top = k;
			if (required_output.at(j, required_top) < required_output.at(j, k)) required_top = k;
		}
		if (top != required_top)
			error++;
	}
	error /= output.get_height();
	return error;
}

void digits()
{
	//Load data

	struct
	{
		binary_data data;
		int32_t magic_number;
		int32_t items_num;
	} labels;
	struct
	{
		binary_data data;
		int32_t magic_number;
		int32_t items_num;
		int32_t num_of_rows;
		int32_t num_of_columns;
		
	} images;

	std::cout << "Loading data into memory...\n";

	//Data source is http://yann.lecun.com/exdb/mnist/
	labels.data = read_file("train-labels.idx1-ubyte");
	images.data = read_file("train-images.idx3-ubyte");

	if (!labels.data.get_data() || !images.data.get_data())
	{
		std::cout << "ERROR: couldn't load data!\n";
		return;
	}

	std::cout << "Processing data...\n";

	labels.magic_number = *(int32_t*)labels.data.get_data();
	labels.items_num = *(int32_t*)(labels.data.get_data() + 4);

	images.magic_number = *(int32_t*)images.data.get_data();
	images.items_num = *(int32_t*)(images.data.get_data() + 4);
	images.num_of_rows = *(int32_t*)(images.data.get_data() + 8);
	images.num_of_columns = *(int32_t*)(images.data.get_data() + 12);

	if (is_little_endian())
	{
		swap_byte_order((char*)&labels.magic_number, sizeof(int32_t));
		swap_byte_order((char*)&labels.items_num, sizeof(int32_t));
		swap_byte_order((char*)&images.magic_number, sizeof(int32_t));
		swap_byte_order((char*)&images.items_num, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_rows, sizeof(int32_t));
		swap_byte_order((char*)&images.num_of_columns, sizeof(int32_t));
	}

	if (labels.magic_number != 0x00000801 || images.magic_number != 0x00000803)
	{
		std::cout << "ERROR: magic numbers don't match!\n";
		return;
	}

	if (labels.items_num != images.items_num)
	{
		std::cout << "ERROR: data sizes don't match\n";
		return;
	}

	const unsigned input_layer_size = images.num_of_rows * images.num_of_columns;
	const unsigned samples_num = images.items_num;

	matrix input(input_layer_size, samples_num);
	matrix required_output(10, samples_num);

	const float mul = 1.f / 255;

	//Init training data
	for (unsigned j = 0; j < samples_num; j++)
	{
		for (unsigned k = 0; k < input_layer_size; k++)
		{
			const char* pixel_addr = images.data.get_data() + 16 + input_layer_size * j + k;
			input.at(j, k) = *reinterpret_cast<const uint8_t*>(pixel_addr) * mul;
		}
		const char* label_addr = labels.data.get_data() + 8 + j;
		required_output.at(j, *reinterpret_cast<const uint8_t*>(label_addr)) = 1.f;
	}

	//Construct neural network
	const unsigned num_layers = 4;
	const unsigned layer_sizes[num_layers] = {input_layer_size, 40, 20, 10};
	neural_net net(num_layers, layer_sizes);


	//Start training
	std::cout << "Training...\n";

	for (unsigned h = 0; h < 100; h++)
	{
		net.train_stochastic(input, required_output, 1000, 0.1);
		
		net.save_to_file("digits_net.bin");

		//Calculate error
		const unsigned num_test_samples = 100;
		unsigned sample_beg = random_int(0, samples_num - 1 - num_test_samples);
		float error = calculate_error(net.run(input.submatrix(sample_beg, sample_beg + num_test_samples)), required_output.submatrix(sample_beg, sample_beg + num_test_samples));

		std::cout << "Iteration #" << h*1000 << '\n';
		std::cout << "Error: " << error << '\n';
	}

	// Print results
	const unsigned num_test_samples = 50;
	matrix test_input = input.submatrix(0, num_test_samples);
	matrix test_required_output = required_output.submatrix(0, num_test_samples);
	matrix output = net.run(test_input);

	//For every image print the result
	for (unsigned j = 0; j < num_test_samples; j++)
	{
		print_image(test_input.get_data() + j * input_layer_size, images.num_of_columns, images.num_of_rows);
		print(submatrix(test_required_output, j, j + 1, 0, test_required_output.get_width()));

		matrix normalized_output = submatrix(output, j, j + 1, 0, output.get_width());
		normalized_output = normalized_output / (normalized_output * matrix(1, normalized_output.get_width(), 1.f)).at(0);
		print(normalized_output);

		system("pause");
	}
}

void simple_example()
{
	const unsigned num_layers = 4;
	const unsigned layer_sizes[num_layers] = { 3, 4, 3, 2 };

	neural_net net(num_layers, layer_sizes);

	matrix input(3, 1);
	input.at(0, 0) = 0.5f;
	input.at(0, 1) = 0.2f;
	input.at(0, 2) = 0.8f;

	matrix required_output(2, 1);
	required_output.at(0, 0) = 1.f;
	required_output.at(0, 1) = 0.f;

	//Print initial output
	std::cout << "Initial output:\n";
	print(net.run(input));
	std::cout << '\n';

	//Run backpropagation
	for (int i = 0; i < 100; i++)
	{
		std::cout << "Iteration #" << i << '\n';

		net.backpropagation(input, required_output, 20.f);

		//Calculate Error
		matrix result = net.run(input);
		matrix delta = required_output - result;
		float delta_square_sum = (delta * transpose(delta)).at(0, 0);

		//Print results
		std::cout << "Error : " << delta_square_sum << '\n';
		print(result);
	}

	//Test file saving and loading functionality
	std::cout << "Test file saving and loading functionality\n";
	net.save_to_file("simple_example.bin");
	net.load_from_file("simple_example.bin");
	print(net.run(input));
}

int main()
{
	//simple_example();
	digits();

	return 0;
}
