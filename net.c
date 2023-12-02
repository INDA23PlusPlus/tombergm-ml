#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "net.h"

static float sigmoid(float z)
{
	return 1.f / (1.f + expf(-z));
}

static float sigmoid_prime(float z)
{
	float s = sigmoid(z);
	return s * (1.f - s);
}

static float cc_quadratic_fn(float a, float y)
{
	return 0.f;
}

static float cc_quadratic_delta_fn(float z, float a, float y)
{
	return (a - y) * sigmoid_prime(z);
}

static const net_cost_t cc_quadratic =
{
	cc_quadratic_fn,
	cc_quadratic_delta_fn,
};

static float cc_crossentropy_fn(float a, float y)
{
	return 0.f;
}

static float cc_crossentropy_delta_fn(float z, float a, float y)
{
	return a - y;
}

static const net_cost_t cc_crossentropy =
{
	cc_crossentropy_fn,
	cc_crossentropy_delta_fn,
};

const net_cost_t *const cost_quadratic = &cc_quadratic;
const net_cost_t *const cost_crossentropy = &cc_crossentropy;

void net_init(net_t *net, int depth, const int *sizes,
		const net_cost_t *cost)
{
	net->depth = depth;
	net->layers = malloc(sizeof(*net->layers) * net->depth);
	net->cost = cost;

	for (int i = 0; i < depth; i++)
	{
		net_layer_t *l = &net->layers[i];

		l->size = sizes[i];

		if (i != 0)
		{
			net_layer_t *b = &net->layers[i - 1];

			l->weights = malloc(sizeof(*l->weights) *
						l->size * b->size);
			l->biases = malloc(sizeof(*l->biases) * l->size);

			/* initialize biases to -1..1, uniform distribution
			 * weights to -d..d, uniform distribution,
			 * d = 1 / sqrt(n^l - 1) */
			for (int y = 0; y < l->size; y++)
			{
				float d = 1.f / sqrtf(b->size);

				l->biases[y] = 1.f - rand() / (RAND_MAX / 2.f);

				for (int x = 0; x < b->size; x++)
				{
					float w = d - rand() / (RAND_MAX / 2.f) * d;
					l->weights[y * b->size + x] = w;
				}
			}

			l->weighted_inputs = malloc(sizeof(*l->weighted_inputs) * l->size);
			l->dnabla_w = malloc(sizeof(*l->dnabla_w) * l->size * b->size);
			l->dnabla_b = malloc(sizeof(*l->dnabla_b) * l->size);
			l->nabla_w = malloc(sizeof(*l->nabla_w) * l->size * b->size);
			l->nabla_b = malloc(sizeof(*l->nabla_b) * l->size);
		}
		else
		{
			l->weights		= NULL;
			l->biases		= NULL;
			l->weighted_inputs	= NULL;
			l->dnabla_w		= NULL;
			l->dnabla_b		= NULL;
			l->nabla_w		= NULL;
			l->nabla_b		= NULL;
		}

		l->outputs = malloc(sizeof(*l->outputs) * l->size);
	}
}

const float *net_ff(const net_t *net, const float *inputs)
{
	memcpy(net->layers[0].outputs, inputs,
		sizeof(*inputs) * net->layers[0].size);

	for (int i = 1; i < net->depth; i++)
	{
		net_layer_t *l = &net->layers[i];
		net_layer_t *b = &net->layers[i - 1];

		for (int y = 0; y < l->size; y++)
		{
			float z = 0.f;

			for (int x = 0; x < b->size; x++)
			{
				float n = l->weights[y * b->size + x];
				z += n * b->outputs[x];
			}

			l->weighted_inputs[y] = z + l->biases[y];
			l->outputs[y] = sigmoid(l->weighted_inputs[y]);
		}
	}

	return net->layers[net->depth - 1].outputs;
}

void net_backprop(const net_t *net, float *desired_outputs)
{
	/* compute errors in output layer */
	{
		net_layer_t *l = &net->layers[net->depth - 1];
		net_layer_t *b = &net->layers[net->depth - 2];

		for (int y = 0; y < l->size; y++)
		{
			float d = net->cost->dn(l->weighted_inputs[y],
						l->outputs[y],
						desired_outputs[y]);

			l->dnabla_b[y] = d;

			for (int x = 0; x < b->size; x++)
			{
				float g = b->outputs[x] * l->dnabla_b[y];
				l->dnabla_w[y * b->size + x] = g;
			}
		}
	}

	/* backpropagate errors */
	for (int i = net->depth - 2; i > 0; i--)
	{
		net_layer_t *l = &net->layers[i];
		net_layer_t *f = &net->layers[i + 1];
		net_layer_t *b = &net->layers[i - 1];

		for (int y = 0; y < l->size; y++)
		{
			float d = 0.f;

			for (int x = 0; x < f->size; x++)
			{
				float dx = f->weights[x * f->size + y];
				dx *= f->dnabla_b[x];
				d += dx;
			}

			l->dnabla_b[y] = d * sigmoid_prime(l->weighted_inputs[y]);

			for (int x = 0; x < b->size; x++)
			{
				float g = b->outputs[x] * l->dnabla_b[y];
				l->dnabla_w[y * b->size + x] = g;
			}
		}
	}
}

void net_sgd(net_t *net, int data_size, float *data_inputs,
		float *data_outputs, int epochs, int batch_size, float rate,
		float lambda, net_callback_t callback)
{
	int 	input_size	= net->layers[0].size;
	int 	output_size	= net->layers[net->depth - 1].size;
	int 	no_batches	= (data_size + batch_size - 1) / batch_size;
	int *	indices		= malloc(sizeof(*indices) * data_size);

	for (int i = 0; i < data_size; i++)
	{
		indices[i] = i;
	}

	/* epoch loop */
	for (int i = 0; i < epochs; i++)
	{
		/* shuffle indices */
		for (int j = 0; j < data_size - 1; j++)
		{
			int n = j + rand() % (data_size - j);
			int t = indices[n];
			indices[n] = indices[j];
			indices[j] = t;
		}

		/* process batches */
		int index = 0;

		for (int j = 0; j < no_batches; j++)
		{
			/* the last batch may have less training data */
			int current_batch_size = batch_size;

			if (index + current_batch_size > data_size)
			{
				current_batch_size = data_size - index;
			}

			/* clear nabla sums */
			for (int k = 1; k < net->depth; k++)
			{
				net_layer_t *l = &net->layers[k];
				net_layer_t *b = &net->layers[k - 1];
				for (int m = 0; m < l->size; m++)
				{
					l->nabla_b[m] = 0.f;
					for (int n = 0; n < b->size; n++)
					{
						l->nabla_w[m * b->size + n] = 0.f;
					}
				}
			}

			/* train on current batch */
			for (int k = 0; k < current_batch_size; k++)
			{
				/* perform feedforward and backpropagation
				 * on the current training data */
				int n = indices[index++];
				net_ff(net, &data_inputs[n * input_size]);
				net_backprop(net, &data_outputs[n * output_size]);

				/* add nabla sums */
				for (int m = 1; m < net->depth; m++)
				{
					net_layer_t *l = &net->layers[m];
					net_layer_t *b = &net->layers[m - 1];

					for (int n = 0; n < l->size; ++n)
					{
						l->nabla_b[n] += l->dnabla_b[n];
						for (int o = 0; o < b->size; o++)
						{
							float dnw = l->dnabla_w[n * b->size + o];
							l->nabla_w[n * b->size + o] += dnw;
						}
					}
				}
			}

			/* adjust weights and biases */
			for (int k = 1; k < net->depth; k++)
			{
				net_layer_t *l = &net->layers[k];
				net_layer_t *b = &net->layers[k - 1];

				for (int m = 0; m < l->size; m++)
				{
					l->biases[m] -= rate / current_batch_size * l->nabla_b[m];

					for (int n = 0; n < b->size; n++)
					{
						float r = 1.f - rate * (lambda / data_size);
						float nw = l->nabla_w[m * b->size + n];
						float dw = rate / current_batch_size * nw;

						l->weights[m * b->size + n] *= r;
						l->weights[m * b->size + n] -= dw;
					}
				}
			}
		}

		rate *= 0.95;

		if (callback != NULL)
		{
			callback(net, i);
		}
	}

	free(indices);
}
