#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "mnist_loader.h"
#include "net.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static void epoch_callback(net_t *net, int epoch)
{
	static float *	inputs	= NULL;
	static float *	outputs	= NULL;
	int		hits	= 0;

	if (inputs == NULL)
	{
		load_testing_data(&inputs, &outputs);
	}

	for (int i = 0; i < 10000; i++)
	{
		const float *	result	= net_ff(net, &inputs[28 * 28 * i]);
		float		max	= -1.f;
		int		max_n	= 0;

		for (int j = 0; j < 10; j++)
		{
			if (result[j] > max)
			{
				max = result[j];
				max_n = j;
			}
		}

		if (outputs[10 * i + max_n] == 1.f)
		{
			hits++;
		}
	}

	printf("epoch %i: %i / %i\n", epoch + 1, hits, 10000);
}

int main()
{
	net_t	net;
	float *	inputs;
	float *	outputs;
	int	sizes[]	= {784, 800, 10};

	srand(time(NULL));

	load_training_data(&inputs, &outputs);

	net_init(&net, 3, sizes, cost_crossentropy);
	net_sgd(&net, 60000, inputs, outputs,
		10, 100, 0.2f, 1.f,
		epoch_callback);

	free(inputs);
	free(outputs);

	inputs = malloc(28 * 28 * sizeof(*inputs));

	for (;;)
	{
		int		x;
		int		y;
		int		c;
		char		s[128];
		const float *	result;
		uint8_t *	image;

		gets(s);

		if (strlen(s) == 0)
		{
			break;
		}

		image = stbi_load(s, &x, &y, &c, 1);

		for (int i = 0; i < 28 * 28; i++)
		{
			inputs[i] = 1.f - image[i] / 255.f;
		}

		stbi_image_free(image);

		result = net_ff(&net, inputs);

		for (int j = 0; j < 10; j++)
		{
			printf("%i: %.2f%%\n", j, result[j] * 100.f);
		}
	}

	free(inputs);
}
