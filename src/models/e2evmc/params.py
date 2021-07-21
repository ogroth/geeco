"""Hyper-parameter definition of E2E-VMC."""

import collections
import copy


E2E_VMC_DEFAULT_PARAM_DICT = {
    'img_height' : 256,
    'img_width' : 256,
    'img_channels' : 3,
    'dim_jnt_state' : 7,  # dimension of joint state vector
    'dim_grp_command' : 2,  # dimension of command vector of gripper
    'control_mode' : 'cartesian',  # cartesian | velocity -> defines output of decoder
    'num_grp_states' : 3,  # discrete gripper actions: close / no-op / open
    'dim_action' : 4,  # dimension of action vector (relative cartesian EE pos, discrete gripper action)
    'proc_obs' : 'sequence',  # sequence | dynimg; processing of frame buffer
    'proc_tgt' : 'constant',  # constant | residual | dyndiff; processing of target frame
    'dim_s_obs' : 256,  # state embedding size of current observation
    'dim_s_dyn' : 256,  # state embedding size of buffer dynamics
    'dim_s_diff' : 256,  # state embedding size of target difference
    'dim_h_lstm' : 128,  # dimension of LSTM hidden state for frame aggregation
    'dim_h_fc' : 128,  # dimension of FC layer after LSTM before output decoding
    'window_size' : 4,  # num. of frames to aggregate before command prediction
    'l2_regularizer' : 0.0,  # weight of the L2 variable regularization loss
    'lambda_aux' : 1.0,  # weight of the auxiliary pose prediction losses
    'batch_size' : 32,  # num. of frame windows per batch
    'lr' : 1e-4,  # learning rate for Adam optimizer
}

E2EVMCConfig = collections.namedtuple(
    typename='E2EVMCConfig',
    field_names=list(E2E_VMC_DEFAULT_PARAM_DICT.keys())
)
E2E_VMC_DEFAULT_CONFIG = E2EVMCConfig(**E2E_VMC_DEFAULT_PARAM_DICT)


def create_e2evmc_config(custom_params: dict) -> E2EVMCConfig:
  """
  Customizes the default parameters of E2E VMC with the parameters specified in
  'custom_params'.
  """
  customized_keys = list(
      set(custom_params.keys()).intersection(set(E2E_VMC_DEFAULT_PARAM_DICT.keys())))
  customized_params = copy.deepcopy(E2E_VMC_DEFAULT_PARAM_DICT)
  for k in customized_keys:
    customized_params[k] = custom_params[k]
  return E2EVMCConfig(**customized_params)
