#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

static void load_file(const char *filename, void **data, size_t *size)
{
	FILE *f = fopen(filename, "rb");

	fseek(f, 0, SEEK_END);
	*size = ftell(f);
	*data = malloc(*size);

	fseek(f, 0, SEEK_SET);
	fread(*data, 1, *size, f);

	fclose(f);
}

static void load_mnist(const char *data_filename, const char *labels_filename,
			int size, float **p_inputs, float **p_outputs)
{
	void *		file;
	size_t		file_size;
	uint8_t *	udata;

	load_file(data_filename, &file, &file_size);
	udata = file;
	udata += 16;

	float *inputs = malloc(28 * 28 * size * sizeof(*inputs));

	for (int i = 0; i < 28 * 28 * size; i++)
	{
		inputs[i] = udata[i] / 255.f;
	}

	free(file);

	load_file(labels_filename, &file, &file_size);
	udata = file;
	udata += 8;

	float *outputs = malloc(10 * size * sizeof(*outputs));

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			outputs[i * 10 + j] = udata[i] == j ? 1.f : 0.f;
		}
	}

	free(file);

	*p_inputs = inputs;
	*p_outputs = outputs;
}

void load_training_data(float **p_inputs, float **p_outputs)
{
	load_mnist(	"data/train-images.idx3-ubyte",
			"data/train-labels.idx1-ubyte",
			60000, p_inputs, p_outputs);
}

void load_testing_data(float **p_inputs, float **p_outputs)
{
	load_mnist(	"data/t10k-images.idx3-ubyte",
			"data/t10k-labels.idx1-ubyte",
			10000, p_inputs, p_outputs);
}
