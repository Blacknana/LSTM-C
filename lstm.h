#ifndef __LSTM_H__
#define __LSTM_H__

#include "cuda_runtime.h"

enum LSTMScaleParams {
  kLstmGateNumber = 4,
  kHiddenSize = 256,
  kInputSize = 256,
  kCellNumber = 10,
  kLstmTimestep = 100,
};

enum LSTMKernelScaleParams {
  kThreadsPerWarp = 32,
  kWarpsPerBlock = 8,
  kColumnsPerBlock = kThreadsPerWarp,
  kGemvBlockNumber = kHiddenSize / kColumnsPerBlock,
  kRowsPerWarp = kHiddenSize / kWarpsPerBlock,
};

typedef struct {
  float weights_w[kLstmGateNumber][kInputSize][kHiddenSize];
  float weights_u[kLstmGateNumber][kHiddenSize][kHiddenSize];
  float bias[kLstmGateNumber][kHiddenSize];
} CellModel;

typedef struct {
  float data[kHiddenSize];
} StepInput;

typedef struct {
  float state_h[kHiddenSize];
  float state_c[kHiddenSize];
  float gemvw_temp[kLstmGateNumber][kHiddenSize];
  float gemvu_temp[kLstmGateNumber][kHiddenSize];
} CellRuntime;

typedef struct {
  CellModel cell_model[kCellNumber];
} ModelParams;

typedef struct {
  StepInput step_input[kLstmTimestep];
} InputParams;

typedef struct {
  CellRuntime cell_runtime[kCellNumber];
} CellParams;

#endif