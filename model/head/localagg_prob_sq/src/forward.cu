#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getRect(points_xyz + 3 * idx, radii[idx], rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0)
		return;

	tiles_touched[idx] = (rect_max.z - rect_min.z) * (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void renderCUDA(
	const int N,
	const float* __restrict__ pts,
	const int* __restrict__ points_int,
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ means3D,
	const float* __restrict__ scales3D,
	const float* __restrict__ rot3D,
	const float* __restrict__ opas,
	const float* __restrict__ u,
	const float* __restrict__ v,
	const float* __restrict__ semantic,
	float* __restrict__ out_logits,
	float* __restrict__ out_bin_logits,
	float* __restrict__ out_density,
	float* __restrict__ out_probability)
{
    auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
	    return;

	const int* point_int = points_int + idx * 3;
	const int voxel_idx = point_int[0] * grid.y * grid.z + point_int[1] * grid.z + point_int[2];
	const float3 point = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[voxel_idx];

	// Initialize helper variables
	float C[CHANNELS] = { 0 };
	float bin_logit = 1.0;
	float density = 0.0;
	float prob_sum = 0.0;

	for (int i = range.x; i < range.y; i++)
	{
		int gs_idx = point_list[i];
		float3 rot1 = { rot3D[gs_idx * 9 + 0], rot3D[gs_idx * 9 + 1], rot3D[gs_idx * 9 + 2] };
		float3 rot2 = { rot3D[gs_idx * 9 + 3], rot3D[gs_idx * 9 + 4], rot3D[gs_idx * 9 + 5] };
		float3 rot3 = { rot3D[gs_idx * 9 + 6], rot3D[gs_idx * 9 + 7], rot3D[gs_idx * 9 + 8] }; 
		float3 d = { - means3D[gs_idx * 3] + point.x, - means3D[gs_idx * 3 + 1] + point.y, - means3D[gs_idx * 3 + 2] + point.z };
		float3 s = { scales3D[gs_idx * 3], scales3D[gs_idx * 3 + 1], scales3D[gs_idx * 3 + 2] };
		float3 trans = { rot1.x * d.x + rot1.y * d.y + rot1.z * d.z, rot2.x * d.x + rot2.y * d.y + rot2.z * d.z, rot3.x * d.x + rot3.y * d.y + rot3.z * d.z };
		float term_x = powf((trans.x / s.x) * (trans.x / s.x), 1 / u[gs_idx]);
		float term_y = powf((trans.y / s.y) * (trans.y / s.y), 1 / u[gs_idx]);
		float term_z = powf((trans.z / s.z) * (trans.z / s.z), 1 / v[gs_idx]);
		float f = powf(term_x + term_y, u[gs_idx] / v[gs_idx]) + term_z;
		float power = exp(-0.5f * f);
		float prob = power * opas[gs_idx];

		for (int ch = 0; ch < CHANNELS; ch++)
		{
			C[ch] += semantic[CHANNELS * gs_idx + ch] * prob;
		}
		bin_logit = (1 - power) * bin_logit;
		density = power + density;
		prob_sum = prob + prob_sum;
	}

	// Iterate over batches until all done or range is complete
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (prob_sum > 1e-9) {
		for (int ch = 0; ch < CHANNELS; ch++)
			out_logits[idx * CHANNELS + ch] = C[ch] / prob_sum;
	} else {
		for (int ch = 0; ch < CHANNELS - 1; ch++)
			out_logits[idx * CHANNELS + ch] = 1.0 / (CHANNELS - 1);
	}
	out_bin_logits[idx] = 1 - bin_logit;
	out_density[idx] = density;
	out_probability[idx] = prob_sum;
}


void FORWARD::render(
	const int N,
	const float* pts,
	const int* points_int,
	const dim3 grid,
	const uint2* ranges,
	const uint32_t* point_list,
	const float* means3D,
	const float* scales3D,
	const float* rot3D,
	const float* opas,
	const float* u,
	const float* v,
	const float* semantic,
	float* out_logits,
	float* out_bin_logits,
	float* out_density,
	float* out_probability)
{
	renderCUDA<NUM_CHANNELS> << <(N + 255) / 256, 256 >> > (
		N, 
		pts,
		points_int,
		grid,
		ranges,
		point_list,
		means3D,
		scales3D,
		rot3D,
		opas,
		u,
		v,
		semantic,
		out_logits,
		out_bin_logits,
		out_density,
		out_probability);
}


void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		points_xyz,
		radii,
		grid,
		tiles_touched
	);
}