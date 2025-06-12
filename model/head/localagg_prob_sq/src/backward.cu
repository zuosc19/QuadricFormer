#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	int voxel_idx = points_xyz[3 * idx] * grid.y * grid.z + points_xyz[3 * idx + 1] * grid.z + points_xyz[3 * idx + 2];
	voxel2pts[voxel_idx] = idx;
}


template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const int P,
	const uint32_t* __restrict__ offsets,
	const uint32_t* __restrict__ point_list_keys_unsorted,
	const int* __restrict__ voxel2pts,
	const float* __restrict__ pts,
	const float* __restrict__ means3D,
	const float* __restrict__ scales3D,
	const float* __restrict__ rot3D,
	const float* __restrict__ opas,
	const float* __restrict__ u,
	const float* __restrict__ v,
	const float* __restrict__ semantic,
	const float* __restrict__ logits,
	const float* __restrict__ bin_logits,
	const float* __restrict__ density,
	const float* __restrict__ probability,
	const float* __restrict__ logits_grad,
	const float* __restrict__ bin_logits_grad,
	const float* __restrict__ density_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opas_grad,
	float* __restrict__ u_grad,
	float* __restrict__ v_grad,
	float* __restrict__ semantics_grad,
	float* __restrict__ rot3D_grad,
	float* __restrict__ scale3D_grad)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
	    return;

	uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
	uint32_t end = offsets[idx];
	
	const float3 means = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float3 rot1 = {rot3D[idx * 9 + 0], rot3D[idx * 9 + 1], rot3D[idx * 9 + 2]};
	const float3 rot2 = {rot3D[idx * 9 + 3], rot3D[idx * 9 + 4], rot3D[idx * 9 + 5]};
	const float3 rot3 = {rot3D[idx * 9 + 6], rot3D[idx * 9 + 7], rot3D[idx * 9 + 8]}; 
	const float3 s = {scales3D[idx * 3], scales3D[idx * 3 + 1], scales3D[idx * 3 + 2]};
	const float opa = opas[idx];
	const float uu = u[idx];
	const float vv = v[idx];
	float sem[CHANNELS] = {0};
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		sem[ch] = semantic[idx * CHANNELS + ch];
	}

	float means_grad[3] = {0};
	float scales_grad[3] = {0};
	float opa_grad = 0;
	float uu_grad = 0;
	float vv_grad = 0;
	float semantic_grad[CHANNELS] = {0};
	float rot_grad[9] = {0};

	for (int i = start; i < end; i++)
	{
		int voxel_idx = point_list_keys_unsorted[i];
		int pts_idx = voxel2pts[voxel_idx];
		if (pts_idx >= 0)
		{
			float3 d = {- means.x + pts[pts_idx * 3], - means.y + pts[pts_idx * 3 + 1], - means.z + pts[pts_idx * 3 + 2]};
			float3 trans = {rot1.x * d.x + rot1.y * d.y + rot1.z * d.z, rot2.x * d.x + rot2.y * d.y + rot2.z * d.z, rot3.x * d.x + rot3.y * d.y + rot3.z * d.z};
			float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1 / uu);
			float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1 / uu);
			float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1 / vv);
			float f = powf(term_x + term_y, uu / vv) + term_z;
			float power = exp(-0.5f * f);
			float prob = power;
			
			float f_grad = 0.;
			float x_grad = 0.;
			float y_grad = 0.;
			float z_grad = 0.;
			float prob_grad = 0.;
			float prob_sum = probability[pts_idx];

			if (prob_sum > 1e-9) {
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					semantic_grad[ch] += logits_grad[pts_idx * CHANNELS + ch] * prob * opa / prob_sum;
					prob_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * opa / prob_sum;
					opa_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * prob / prob_sum;
				}
			} 
			prob_grad += (1 - bin_logits[pts_idx]) / (1 - power + 1e-9) *  bin_logits_grad[pts_idx];
			f_grad -= 0.5f * prob_grad * power;
			uu_grad += f_grad * powf(term_x + term_y, uu / vv) * ((log(term_x + term_y + 1e-9) / vv) - (term_x * log((trans.x / s.x) * (trans.x / s.x) + 1e-9) + term_y * log((trans.y / s.y) * (trans.y / s.y) + 1e-9)) / uu / vv / (term_x + term_y));
			vv_grad -= f_grad * (uu * powf(term_x + term_y, uu / vv) * log(term_x + term_y + 1e-9) / vv / vv + term_z * log((trans.z / s.z) * (trans.z / s.z) + 1e-9) / vv / vv);

			scales_grad[0] -= f_grad * 2 * term_x * powf(term_x + term_y, uu / vv - 1) / vv / s.x;
			scales_grad[1] -= f_grad * 2 * term_y * powf(term_x + term_y, uu / vv - 1) / vv / s.y;
			scales_grad[2] -= f_grad * 2 * term_z / vv / s.z;

			x_grad += f_grad * 2 * term_x * powf(term_x + term_y, uu / vv - 1) / vv / trans.x;
			y_grad += f_grad * 2 * term_y * powf(term_x + term_y, uu / vv - 1) / vv / trans.y;
			z_grad += f_grad * 2 * term_z / vv / trans.z;

			means_grad[0] -= (rot1.x * x_grad + rot2.x * y_grad + rot3.x * z_grad);
			means_grad[1] -= (rot1.y * x_grad + rot2.y * y_grad + rot3.y * z_grad);
			means_grad[2] -= (rot1.z * x_grad + rot2.z * y_grad + rot3.z * z_grad);

			rot_grad[0] += x_grad * d.x;
			rot_grad[1] += x_grad * d.y;
			rot_grad[2] += x_grad * d.z;
			rot_grad[3] += y_grad * d.x;
			rot_grad[4] += y_grad * d.y;
			rot_grad[5] += y_grad * d.z;
			rot_grad[6] += z_grad * d.x;
			rot_grad[7] += z_grad * d.y;
			rot_grad[8] += z_grad * d.z;
		}
	}

	means3D_grad[idx * 3] = means_grad[0];
	means3D_grad[idx * 3 + 1] = means_grad[1];
	means3D_grad[idx * 3 + 2] = means_grad[2];

	scale3D_grad[idx * 3] = scales_grad[0];
	scale3D_grad[idx * 3 + 1] = scales_grad[1];
	scale3D_grad[idx * 3 + 2] = scales_grad[2];

	opas_grad[idx] = opa_grad;
	u_grad[idx] = uu_grad;
	v_grad[idx] = vv_grad;
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		semantics_grad[idx * CHANNELS + ch] = semantic_grad[ch];
	}
	for (int ch = 0; ch < 9; ch++)
	{
		rot3D_grad[idx * 9 + ch] = rot_grad[ch];
	}
}


void BACKWARD::render(
	const int P,
	const uint32_t* offsets,
	const uint32_t* point_list_keys_unsorted,
	const int* voxel2pts,
	const float* pts,
	const float* means3D,
	const float* scales3D,
	const float* rot3D,
	const float* opas,
	const float* u,
	const float* v,
	const float* semantic,
	const float* logits,
	const float* bin_logits,
	const float* density,
	const float* probability,
	const float* logits_grad,
	const float* bin_logits_grad,
	const float* density_grad,
	float* means3D_grad,
	float* opas_grad,
	float* u_grad,
	float* v_grad,
	float* semantics_grad,
	float* rot3D_grad,
	float* scale3D_grad)
{
	renderCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P,
		offsets,
		point_list_keys_unsorted,
		voxel2pts,
		pts,
		means3D,
		scales3D,
		rot3D,
		opas,
		u,
		v,
		semantic,
		logits,
		bin_logits,
		density,
		probability,
		logits_grad,
		bin_logits_grad,
		density_grad,
		means3D_grad,
		opas_grad,
		u_grad,
		v_grad,
		semantics_grad,
		rot3D_grad,
		scale3D_grad);
}

void BACKWARD::preprocess(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	preprocessCUDA << <(N + 255) / 256, 256 >> > (
		N,
		points_xyz,
		grid,
		voxel2pts
	);
}