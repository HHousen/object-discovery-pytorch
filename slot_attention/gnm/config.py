import yaml
import json
import io
from argparse import Namespace


def dict_to_ns(d):
    return Namespace(**d)


CONFIG_YAML = """exp_name: ''
data_dir: ''
summary_dir: ''
model_dir: ''
last_ckpt: ''
data:
  img_w: 128
  img_h: 128
  inp_channel: 3
  blender_dir_list_train: []
  blender_dir_list_test: []
  dataset: 'mnist'
z:
  z_global_dim: 32
  z_what_dim: 64
  z_where_scale_dim: 2
  z_where_shift_dim: 2
  z_where_dim: 4
  z_pres_dim: 1
  z_depth_dim: 1
  z_local_dim: 64
  z_bg_dim: 10
arch:
  glimpse_size: 64
  num_cell: 4
  phase_overlap: True
  phase_background: True
  img_enc_dim: 128
  p_global_decoder_type: 'MLP'
  draw_step: 4
  phase_graph_net_on_global_decoder: False
  phase_graph_net_on_global_encoder: False
  conv:
    img_encoder_filters: [16, 16, 32, 32, 64, 64, 128, 128, 128]
    img_encoder_groups: [1, 1, 1, 1, 1, 1, 1, 1, 1]
    img_encoder_strides: [2, 1, 2, 1, 2, 1, 2, 1, 2]
    img_encoder_kernel_sizes: [4, 3, 4, 3, 4, 3, 4, 3, 4]
    p_what_decoder_filters: [128, 64, 32, 16, 8, 4]
    p_what_decoder_kernel_sizes: [3, 3, 3, 3, 3, 3]
    p_what_decoder_upscales: [2, 2, 2, 2, 2, 2]
    p_what_decoder_groups: [1, 1, 1, 1, 1, 1]
    p_bg_decoder_filters: [128, 64, 32, 16, 8, 3]
    p_bg_decoder_kernel_sizes: [1, 1, 1, 1, 1, 3]
    p_bg_decoder_upscales: [4, 2, 4, 2, 2, 1]
    p_bg_decoder_groups: [1, 1, 1, 1, 1, 1]
  deconv:
    p_global_decoder_filters: [128, 128, 128]
    p_global_decoder_kernel_sizes: [1, 1, 1]
    p_global_decoder_upscales: [2, 1, 2]
    p_global_decoder_groups: [1, 1, 1]
  mlp:
    p_global_decoder_filters: [512, 1024, 2048]
    q_global_encoder_filters: [512, 512, 64]
    p_global_encoder_filters: [512, 512, 64]
    p_bg_generator_filters: [128, 64, 20]
    q_bg_encoder_filters: [512, 256, 20]
  pwdw:
    pwdw_filters: [128, 128]
    pwdw_kernel_sizes: [1, 1]
    pwdw_strides: [1, 1]
    pwdw_groups: [1, 1]
  structdraw:
    kernel_size: 1
    rnn_decoder_hid_dim: 128
    rnn_encoder_hid_dim: 128
    hid_to_dec_filters: [128]
    hid_to_dec_kernel_sizes: [3]
    hid_to_dec_strides: [1]
    hid_to_dec_groups: [1]
log:
  num_summary_img: 15
  num_img_per_row: 5
  save_epoch_freq: 10
  print_step_freq: 2000
  num_sample: 50
  compute_nll_freq: 20
  phase_nll: False
  nll_num_sample: 30
  phase_log: True
const:
  pres_logit_scale: 8.8
  scale_mean: -1.5
  scale_std: 0.1
  ratio_mean: 0
  ratio_std: 0.3
  shift_std: 1
  eps: 0.000000000000001
  likelihood_sigma: 0.2
  bg_likelihood_sigma: 0.3
train:
  start_epoch: 0
  epoch: 600
  batch_size: 32
  lr: 0.0001
  cp: 1.0
  beta_global_anneal_start_step: 0
  beta_global_anneal_end_step: 100000
  beta_global_anneal_start_value: 0.
  beta_global_anneal_end_value: 1.
  beta_pres_anneal_start_step: 0
  beta_pres_anneal_end_step: 0
  beta_pres_anneal_start_value: 1.
  beta_pres_anneal_end_value: 0.
  beta_where_anneal_start_step: 0
  beta_where_anneal_end_step: 0
  beta_where_anneal_start_value: 1.
  beta_where_anneal_end_value: 0.
  beta_what_anneal_start_step: 0
  beta_what_anneal_end_step: 0
  beta_what_anneal_start_value: 1.
  beta_what_anneal_end_value: 0.
  beta_depth_anneal_start_step: 0
  beta_depth_anneal_end_step: 0
  beta_depth_anneal_start_value: 1.
  beta_depth_anneal_end_value: 0.
  beta_bg_anneal_start_step: 1000
  beta_bg_anneal_end_step: 0
  beta_bg_anneal_start_value: 1.
  beta_bg_anneal_end_value: 0.
  beta_aux_pres_anneal_start_step: 1000
  beta_aux_pres_anneal_end_step: 0
  beta_aux_pres_anneal_start_value: 1.
  beta_aux_pres_anneal_end_value: 0.
  beta_aux_where_anneal_start_step: 0
  beta_aux_where_anneal_end_step: 500
  beta_aux_where_anneal_start_value: 10.
  beta_aux_where_anneal_end_value: 1.
  beta_aux_what_anneal_start_step: 1000
  beta_aux_what_anneal_end_step: 0
  beta_aux_what_anneal_start_value: 1.
  beta_aux_what_anneal_end_value: 0.
  beta_aux_depth_anneal_start_step: 1000
  beta_aux_depth_anneal_end_step: 0
  beta_aux_depth_anneal_start_value: 1.
  beta_aux_depth_anneal_end_value: 0.
  beta_aux_global_anneal_start_step: 0
  beta_aux_global_anneal_end_step: 100000
  beta_aux_global_anneal_start_value: 0.
  beta_aux_global_anneal_end_value: 1.
  beta_aux_bg_anneal_start_step: 0
  beta_aux_bg_anneal_end_step: 50000
  beta_aux_bg_anneal_start_value: 50.
  beta_aux_bg_anneal_end_value: 1.
  tau_pres_anneal_start_step: 1000
  tau_pres_anneal_end_step: 20000
  tau_pres_anneal_start_value: 1.
  tau_pres_anneal_end_value: 0.5
  tau_pres: 1.
  p_pres_anneal_start_step: 0
  p_pres_anneal_end_step: 4000
  p_pres_anneal_start_value: 0.1
  p_pres_anneal_end_value: 0.001
  aux_p_scale_anneal_start_step: 0
  aux_p_scale_anneal_end_step: 0
  aux_p_scale_anneal_start_value: -1.5
  aux_p_scale_anneal_end_value: -1.5
  phase_bg_alpha_curriculum: True
  bg_alpha_curriculum_period: [0, 500]
  bg_alpha_curriculum_value: 0.9
  seed: 666
"""


def get_arrow_args():
    arrow_args = json.loads(
        json.dumps(yaml.safe_load(io.StringIO(CONFIG_YAML))), object_hook=dict_to_ns
    )
    return arrow_args
