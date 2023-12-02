#ifndef NET_H
#define NET_H

typedef float (*net_cost_fn_t)(float a, float y);
typedef float (*net_cost_dn_t)(float z, float a, float y);

typedef struct
{
	net_cost_fn_t	fn;
	net_cost_dn_t	dn;
} net_cost_t;

typedef struct
{
	int	size;
	float *	weights;
	float *	biases;
	float *	weighted_inputs;
	float *	outputs;
	float *	dnabla_w;
	float *	dnabla_b;
	float *	nabla_w;
	float *	nabla_b;
} net_layer_t;

typedef struct
{
	int			depth;
	net_layer_t *		layers;
	const net_cost_t *	cost;
} net_t;

typedef void (*	net_callback_t)	(net_t *net, int epoch);

void		net_init	(net_t *net, int depth, const int *sizes,
					const net_cost_t *cost);
const float *	net_ff		(const net_t *net, const float *inputs);
void		net_backprop	(const net_t *net, float *desired_outputs);
void		net_sgd		(net_t *net, int data_size, float *data_inputs,
					float *data_outputs, int epochs,
					int batch_size, float rate,
					float lambda,
					net_callback_t callback);

extern const net_cost_t *const cost_quadratic;
extern const net_cost_t *const cost_crossentropy;

#endif
