#include "common.h"
#include "lstm.h"
#include <math.h>
#include <stdio.h>

__device__ static inline float sigmoid(float x) {
  return 1.000000e+00f / (1.000000e+00f + __expf(0.000000e+00f - x));
}

__global__ void gemvw(StepInput *d_input, CellModel *d_model,
                      CellRuntime *d_runtime, int gate_num) {
  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx = blockIdx.x * kColumnsPerBlock + lane_idx;

  if (warp_idx == 0) {
    d_runtime->gemvw_temp[gate_num][col_idx] = 0.000000e+00f;
  }
  __syncthreads();

  float temp = 0.000000e+00f;
  const int row_start_idx = kRowsPerWarp * warp_idx;
  const int row_end_idx = row_start_idx + kRowsPerWarp;
  for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
    float input_data = d_input->data[row_idx];
    temp =
        fma(d_model->weights_w[gate_num][row_idx][col_idx], input_data, temp);
  }

  atomicAdd(&d_runtime->gemvw_temp[gate_num][col_idx], temp);
}

__global__ void gemvu(CellModel *d_model, CellRuntime *d_runtime,
                      int gate_num) {
  const int warp_idx = threadIdx.x / kThreadsPerWarp;
  const int lane_idx = threadIdx.x % kThreadsPerWarp;
  const int col_idx = blockIdx.x * kColumnsPerBlock + lane_idx;

  if (warp_idx == 0) {
    d_runtime->gemvu_temp[gate_num][col_idx] = 0.000000e+00f;
  }
  __syncthreads();

  float temp = 0.000000e+00f;
  const int row_start_idx = kRowsPerWarp * warp_idx;
  const int row_end_idx = row_start_idx + kRowsPerWarp;
  for (int row_idx = row_start_idx; row_idx < row_end_idx; ++row_idx) {
    float state_h_data = d_runtime->state_h[row_idx];
    temp =
        fma(d_model->weights_u[gate_num][row_idx][col_idx], state_h_data, temp);
  }

  atomicAdd(&d_runtime->gemvu_temp[gate_num][col_idx], temp);
}

__global__ void solve(StepInput *d_output, CellModel *d_model,
                      CellRuntime *d_runtime) {
  const int col_idx = threadIdx.x;

  float input_gate_x = d_runtime->gemvw_temp[0][col_idx] +
                       d_runtime->gemvu_temp[0][col_idx] +
                       d_model->bias[0][col_idx];
  float input_gate_y = d_runtime->gemvw_temp[1][col_idx] +
                       d_runtime->gemvu_temp[1][col_idx] +
                       d_model->bias[1][col_idx];
  float forget_gate = d_runtime->gemvw_temp[2][col_idx] +
                      d_runtime->gemvu_temp[2][col_idx] +
                      d_model->bias[2][col_idx];
  float output_gate = d_runtime->gemvw_temp[3][col_idx] +
                      d_runtime->gemvu_temp[3][col_idx] +
                      d_model->bias[3][col_idx];
  input_gate_x = sigmoid(input_gate_x);
  input_gate_y = tanh(input_gate_y);
  output_gate = sigmoid(output_gate);
  forget_gate =
      sigmoid(forget_gate + 1.000000e+00f) * d_runtime->state_c[col_idx];
  d_runtime->state_c[col_idx] = fma(input_gate_x, input_gate_y, forget_gate);
  d_runtime->state_h[col_idx] =
      (tanh(d_runtime->state_c[col_idx])) * output_gate;
  d_output->data[col_idx] = d_runtime->state_h[col_idx];
}

ModelParams *g_h_model, *g_d_model;
InputParams *g_h_input, *g_d_input;
InputParams *g_h_output, *g_h_expect_output;
CellParams *g_h_cells, *g_d_cells;

void CellCompute(int step_idx, int cell_idx) {
  StepInput *step_input = &g_d_input->step_input[step_idx];
  CellModel *cell_model = &g_d_model->cell_model[cell_idx];
  CellRuntime *cell_runtime = &g_d_cells->cell_runtime[cell_idx];

  for (int i = 0; i < kLstmGateNumber; ++i) {
    void *gemvw_params[] = {&step_input, &cell_model, &cell_runtime, &i};
    CUDA_CHECK(cudaLaunchKernel((void *)gemvw, (dim3)kGemvBlockNumber,
                                (dim3)kHiddenSize, gemvw_params, 0, 0));
  }

  for (int i = 0; i < kLstmGateNumber; ++i) {
    void *gemvu_params[] = {&cell_model, &cell_runtime, &i};
    CUDA_CHECK(cudaLaunchKernel((void *)gemvu, (dim3)kGemvBlockNumber,
                                (dim3)kHiddenSize, gemvu_params, 0, 0));
  }

  void *solve_params[] = {&step_input, &cell_model, &cell_runtime};
  CUDA_CHECK(cudaLaunchKernel((void *)solve, (dim3)1, (dim3)kHiddenSize,
                              solve_params, 0, 0));
}

void Initialize() {
  CUDA_CHECK(cudaMemcpy(g_d_input, g_h_input, sizeof(InputParams),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(g_d_cells, 0.0f, sizeof(CellParams) / sizeof(float)));
}

void Solve() {
  for (int i = 0; i < kLstmTimestep; ++i) {
    for (int j = 0; j < kCellNumber; ++j) {
      CellCompute(i, j);
    }
  }
}

void Fetch() {
  CUDA_CHECK(cudaMemcpy(g_h_output, g_d_input, sizeof(InputParams),
                        cudaMemcpyDeviceToHost));
}

int main() {
  CUDA_CHECK(cudaMallocHost((void **)&g_h_model, sizeof(ModelParams)));
  CUDA_CHECK(cudaMallocHost((void **)&g_h_input, sizeof(InputParams)));
  CUDA_CHECK(cudaMallocHost((void **)&g_h_output, sizeof(InputParams)));
  CUDA_CHECK(cudaMallocHost((void **)&g_h_expect_output, sizeof(InputParams)));
  CUDA_CHECK(cudaMallocHost((void **)&g_h_cells, sizeof(CellParams)));
  CUDA_CHECK(cudaMalloc((void **)&g_d_model, sizeof(ModelParams)));
  CUDA_CHECK(cudaMalloc((void **)&g_d_input, sizeof(InputParams)));
  CUDA_CHECK(cudaMalloc((void **)&g_d_cells, sizeof(CellParams)));

  FILE *model_file, *input_file, *output_file;
  model_file = fopen("model_params.txt", "r");
  input_file = fopen("input_params.txt", "r");
  output_file = fopen("expect_results.txt", "r");
  for (int i = 0; !feof(model_file); ++i) {
    fscanf(model_file, "%f", (float *)g_h_model + i);
  }
  for (int i = 0; !feof(input_file); ++i) {
    fscanf(input_file, "%f", (float *)g_h_input + i);
  }
  for (int i = 0; !feof(output_file); ++i) {
    fscanf(output_file, "%f", (float *)g_h_expect_output + i);
  }

  CUDA_CHECK(cudaMemcpy(g_d_model, g_h_model, sizeof(ModelParams),
                        cudaMemcpyHostToDevice));

  // correctness test
  Initialize();
  Solve();
  Fetch();
  for (unsigned int i = 0; i < sizeof(InputParams) / sizeof(float); ++i) {
    if (fabs(((float *)g_h_expect_output)[i] - ((float *)g_h_output)[i]) >
        1e-5) {
      printf("The result is incorrect.\n");
      exit(1);
    }
  }

  cudaEvent_t start, stop;
  float elapsedTime, duration = 0.0f;
  const int warm_up = 200, exec = 1000;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // warm up
  for (int i = 0; i < warm_up; ++i) {
    Initialize();
    Solve();
    Fetch();
  }
  cudaStreamSynchronize(0);

  // benchmark
  for (int i = 0; i < exec; ++i) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    Initialize();
    Solve();
    Fetch();
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Iteration time %f us\n", elapsedTime);
    duration += elapsedTime;
  }
  printf("Mean time: %fms\n", duration / exec);

  CUDA_CHECK(cudaFreeHost(g_h_model));
  CUDA_CHECK(cudaFreeHost(g_h_input));
  CUDA_CHECK(cudaFreeHost(g_h_output));
  CUDA_CHECK(cudaFreeHost(g_h_expect_output));
  CUDA_CHECK(cudaFreeHost(g_h_cells));
  CUDA_CHECK(cudaFree(g_d_model));
  CUDA_CHECK(cudaFree(g_d_input));
  CUDA_CHECK(cudaFree(g_d_cells));
  fclose(model_file);
  fclose(input_file);
  fclose(output_file);

  return 0;
}