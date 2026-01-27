/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-27T16:46:53+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x54bf030d97ad5a35ad1977c3eddbd6ff"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-01-27T16:46:53+0100"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 49152, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  Conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  Conv1_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  block_1_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 196608, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  block_1_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 196608, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  block_1_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  block_1_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  block_1_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  block_2_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  block_2_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  block_2_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  block_2_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  block_2_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  block_2_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  block_3_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  block_3_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  block_3_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  block_3_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  block_3_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  block_4_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  block_4_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  block_4_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  block_4_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  block_4_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  block_4_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  block_5_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  block_5_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  block_5_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  block_5_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  block_5_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  block_5_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  block_6_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  block_6_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24576, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  block_6_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  block_6_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  block_6_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  block_7_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  block_7_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  block_7_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  block_7_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  block_7_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  block_7_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  block_8_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  block_8_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  block_8_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  block_8_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  block_8_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  block_8_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  block_9_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  block_9_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  block_9_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  block_9_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  block_9_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  block_9_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  block_10_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  block_10_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  block_10_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  block_10_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  block_10_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  block_11_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  block_11_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  block_11_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  block_11_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  block_11_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  block_11_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  block_12_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  block_12_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  block_12_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  block_12_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  block_12_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  block_12_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  block_13_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  block_13_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  block_13_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  block_13_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3072, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  block_13_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  block_14_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  block_14_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  block_14_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  block_14_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  block_14_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  block_14_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  block_15_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  block_15_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  block_15_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  block_15_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  block_15_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  block_15_add_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  block_16_expand_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  block_16_expand_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  block_16_depthwise_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  block_16_depthwise_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  block_16_project_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1792, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  Conv_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20480, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  out_relu_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 20480, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  global_average_pooling2d_pool_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1280, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  predictions_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1000, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  predictions_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1000, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  Conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  Conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  block_1_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  block_1_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  block_1_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  block_1_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  block_1_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  block_1_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  block_2_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  block_2_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  block_2_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  block_2_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  block_2_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  block_2_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  block_3_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 384, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  block_3_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  block_3_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  block_3_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  block_3_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  block_3_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  block_4_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  block_4_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  block_4_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 864, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  block_4_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  block_4_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  block_4_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  block_5_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  block_5_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  block_5_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 864, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  block_5_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  block_5_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  block_5_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  block_6_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  block_6_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  block_6_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 864, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  block_6_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  block_6_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  block_6_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  block_7_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  block_7_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  block_7_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1296, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  block_7_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  block_7_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  block_7_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  block_8_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#150 */
AI_ARRAY_OBJ_DECLARE(
  block_8_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#151 */
AI_ARRAY_OBJ_DECLARE(
  block_8_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1296, AI_STATIC)

/* Array#152 */
AI_ARRAY_OBJ_DECLARE(
  block_8_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#153 */
AI_ARRAY_OBJ_DECLARE(
  block_8_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#154 */
AI_ARRAY_OBJ_DECLARE(
  block_8_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#155 */
AI_ARRAY_OBJ_DECLARE(
  block_9_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#156 */
AI_ARRAY_OBJ_DECLARE(
  block_9_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#157 */
AI_ARRAY_OBJ_DECLARE(
  block_9_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1296, AI_STATIC)

/* Array#158 */
AI_ARRAY_OBJ_DECLARE(
  block_9_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#159 */
AI_ARRAY_OBJ_DECLARE(
  block_9_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#160 */
AI_ARRAY_OBJ_DECLARE(
  block_9_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#161 */
AI_ARRAY_OBJ_DECLARE(
  block_10_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3456, AI_STATIC)

/* Array#162 */
AI_ARRAY_OBJ_DECLARE(
  block_10_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#163 */
AI_ARRAY_OBJ_DECLARE(
  block_10_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1296, AI_STATIC)

/* Array#164 */
AI_ARRAY_OBJ_DECLARE(
  block_10_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#165 */
AI_ARRAY_OBJ_DECLARE(
  block_10_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4608, AI_STATIC)

/* Array#166 */
AI_ARRAY_OBJ_DECLARE(
  block_10_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#167 */
AI_ARRAY_OBJ_DECLARE(
  block_11_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#168 */
AI_ARRAY_OBJ_DECLARE(
  block_11_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#169 */
AI_ARRAY_OBJ_DECLARE(
  block_11_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1728, AI_STATIC)

/* Array#170 */
AI_ARRAY_OBJ_DECLARE(
  block_11_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#171 */
AI_ARRAY_OBJ_DECLARE(
  block_11_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#172 */
AI_ARRAY_OBJ_DECLARE(
  block_11_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#173 */
AI_ARRAY_OBJ_DECLARE(
  block_12_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#174 */
AI_ARRAY_OBJ_DECLARE(
  block_12_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#175 */
AI_ARRAY_OBJ_DECLARE(
  block_12_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1728, AI_STATIC)

/* Array#176 */
AI_ARRAY_OBJ_DECLARE(
  block_12_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#177 */
AI_ARRAY_OBJ_DECLARE(
  block_12_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#178 */
AI_ARRAY_OBJ_DECLARE(
  block_12_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#179 */
AI_ARRAY_OBJ_DECLARE(
  block_13_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6144, AI_STATIC)

/* Array#180 */
AI_ARRAY_OBJ_DECLARE(
  block_13_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#181 */
AI_ARRAY_OBJ_DECLARE(
  block_13_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1728, AI_STATIC)

/* Array#182 */
AI_ARRAY_OBJ_DECLARE(
  block_13_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#183 */
AI_ARRAY_OBJ_DECLARE(
  block_13_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 10752, AI_STATIC)

/* Array#184 */
AI_ARRAY_OBJ_DECLARE(
  block_13_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#185 */
AI_ARRAY_OBJ_DECLARE(
  block_14_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18816, AI_STATIC)

/* Array#186 */
AI_ARRAY_OBJ_DECLARE(
  block_14_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#187 */
AI_ARRAY_OBJ_DECLARE(
  block_14_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3024, AI_STATIC)

/* Array#188 */
AI_ARRAY_OBJ_DECLARE(
  block_14_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#189 */
AI_ARRAY_OBJ_DECLARE(
  block_14_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18816, AI_STATIC)

/* Array#190 */
AI_ARRAY_OBJ_DECLARE(
  block_14_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#191 */
AI_ARRAY_OBJ_DECLARE(
  block_15_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18816, AI_STATIC)

/* Array#192 */
AI_ARRAY_OBJ_DECLARE(
  block_15_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#193 */
AI_ARRAY_OBJ_DECLARE(
  block_15_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3024, AI_STATIC)

/* Array#194 */
AI_ARRAY_OBJ_DECLARE(
  block_15_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#195 */
AI_ARRAY_OBJ_DECLARE(
  block_15_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18816, AI_STATIC)

/* Array#196 */
AI_ARRAY_OBJ_DECLARE(
  block_15_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#197 */
AI_ARRAY_OBJ_DECLARE(
  block_16_expand_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18816, AI_STATIC)

/* Array#198 */
AI_ARRAY_OBJ_DECLARE(
  block_16_expand_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#199 */
AI_ARRAY_OBJ_DECLARE(
  block_16_depthwise_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3024, AI_STATIC)

/* Array#200 */
AI_ARRAY_OBJ_DECLARE(
  block_16_depthwise_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#201 */
AI_ARRAY_OBJ_DECLARE(
  block_16_project_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 37632, AI_STATIC)

/* Array#202 */
AI_ARRAY_OBJ_DECLARE(
  block_16_project_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#203 */
AI_ARRAY_OBJ_DECLARE(
  Conv_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 143360, AI_STATIC)

/* Array#204 */
AI_ARRAY_OBJ_DECLARE(
  Conv_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1280, AI_STATIC)

/* Array#205 */
AI_ARRAY_OBJ_DECLARE(
  predictions_dense_weights_array, AI_ARRAY_FORMAT_LUT4_FLOAT,
  NULL, NULL, 1280000, AI_STATIC)

/* Array#206 */
AI_ARRAY_OBJ_DECLARE(
  predictions_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1000, AI_STATIC)

/* Array#207 */
AI_ARRAY_OBJ_DECLARE(
  Conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 27, AI_STATIC)

/* Array#208 */
AI_ARRAY_OBJ_DECLARE(
  expanded_conv_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#209 */
AI_ARRAY_OBJ_DECLARE(
  block_1_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#210 */
AI_ARRAY_OBJ_DECLARE(
  block_1_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#211 */
AI_ARRAY_OBJ_DECLARE(
  block_2_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#212 */
AI_ARRAY_OBJ_DECLARE(
  block_2_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#213 */
AI_ARRAY_OBJ_DECLARE(
  block_3_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#214 */
AI_ARRAY_OBJ_DECLARE(
  block_3_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#215 */
AI_ARRAY_OBJ_DECLARE(
  block_4_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#216 */
AI_ARRAY_OBJ_DECLARE(
  block_4_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#217 */
AI_ARRAY_OBJ_DECLARE(
  block_5_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#218 */
AI_ARRAY_OBJ_DECLARE(
  block_5_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#219 */
AI_ARRAY_OBJ_DECLARE(
  block_6_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#220 */
AI_ARRAY_OBJ_DECLARE(
  block_6_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#221 */
AI_ARRAY_OBJ_DECLARE(
  block_7_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#222 */
AI_ARRAY_OBJ_DECLARE(
  block_7_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#223 */
AI_ARRAY_OBJ_DECLARE(
  block_8_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#224 */
AI_ARRAY_OBJ_DECLARE(
  block_8_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#225 */
AI_ARRAY_OBJ_DECLARE(
  block_9_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#226 */
AI_ARRAY_OBJ_DECLARE(
  block_9_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#227 */
AI_ARRAY_OBJ_DECLARE(
  block_10_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#228 */
AI_ARRAY_OBJ_DECLARE(
  block_10_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#229 */
AI_ARRAY_OBJ_DECLARE(
  block_11_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#230 */
AI_ARRAY_OBJ_DECLARE(
  block_11_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#231 */
AI_ARRAY_OBJ_DECLARE(
  block_12_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#232 */
AI_ARRAY_OBJ_DECLARE(
  block_12_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#233 */
AI_ARRAY_OBJ_DECLARE(
  block_13_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#234 */
AI_ARRAY_OBJ_DECLARE(
  block_13_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#235 */
AI_ARRAY_OBJ_DECLARE(
  block_14_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#236 */
AI_ARRAY_OBJ_DECLARE(
  block_14_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#237 */
AI_ARRAY_OBJ_DECLARE(
  block_15_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#238 */
AI_ARRAY_OBJ_DECLARE(
  block_15_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#239 */
AI_ARRAY_OBJ_DECLARE(
  block_16_expand_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#240 */
AI_ARRAY_OBJ_DECLARE(
  block_16_project_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 336, AI_STATIC)

/* Array#241 */
AI_ARRAY_OBJ_DECLARE(
  Conv_1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  Conv1_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &Conv1_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  Conv1_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 64, 64), AI_STRIDE_INIT(4, 4, 4, 64, 4096),
  1, &Conv1_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  Conv1_relu_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 64, 64), AI_STRIDE_INIT(4, 4, 4, 64, 4096),
  1, &Conv1_relu_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  Conv1_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 3), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &Conv1_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  Conv1_weights, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 3, 3, 3, 16), AI_STRIDE_INIT(4, 4, 12, 192, 576),
  1, &Conv1_weights_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  Conv_1_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 1280, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5120, 5120),
  1, &Conv_1_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  Conv_1_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 1280, 4, 4), AI_STRIDE_INIT(4, 4, 4, 5120, 20480),
  1, &Conv_1_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  Conv_1_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 4, 4, 448, 448),
  1, &Conv_1_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  Conv_1_weights, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 112, 1, 1, 1280), AI_STRIDE_INIT(4, 4, 448, 573440, 573440),
  1, &Conv_1_weights_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  block_10_depthwise_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_10_depthwise_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  block_10_depthwise_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_10_depthwise_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  block_10_depthwise_relu_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_10_depthwise_relu_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  block_10_depthwise_weights, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 144), AI_STRIDE_INIT(4, 1, 144, 144, 144),
  1, &block_10_depthwise_weights_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  block_10_expand_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_10_expand_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  block_10_expand_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_10_expand_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  block_10_expand_relu_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_10_expand_relu_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  block_10_expand_scratch0, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_10_expand_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  block_10_expand_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 144), AI_STRIDE_INIT(4, 4, 96, 13824, 13824),
  1, &block_10_expand_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  block_10_project_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_10_project_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  block_10_project_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_10_project_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  block_10_project_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_10_project_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  block_10_project_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 144, 1, 1, 32), AI_STRIDE_INIT(4, 4, 576, 18432, 18432),
  1, &block_10_project_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  block_11_add_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_11_add_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  block_11_depthwise_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_11_depthwise_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  block_11_depthwise_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_11_depthwise_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  block_11_depthwise_relu_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_11_depthwise_relu_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  block_11_depthwise_weights, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 192), AI_STRIDE_INIT(4, 1, 192, 192, 192),
  1, &block_11_depthwise_weights_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  block_11_expand_bias, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_11_expand_bias_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  block_11_expand_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_11_expand_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  block_11_expand_relu_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_11_expand_relu_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  block_11_expand_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_11_expand_scratch0_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  block_11_expand_weights, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 192), AI_STRIDE_INIT(4, 4, 128, 24576, 24576),
  1, &block_11_expand_weights_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  block_11_project_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_11_project_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  block_11_project_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_11_project_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  block_11_project_scratch0, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_11_project_scratch0_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  block_11_project_weights, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 32), AI_STRIDE_INIT(4, 4, 768, 24576, 24576),
  1, &block_11_project_weights_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  block_12_add_output, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_12_add_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  block_12_depthwise_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_12_depthwise_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  block_12_depthwise_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_12_depthwise_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  block_12_depthwise_relu_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_12_depthwise_relu_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  block_12_depthwise_weights, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 192), AI_STRIDE_INIT(4, 1, 192, 192, 192),
  1, &block_12_depthwise_weights_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  block_12_expand_bias, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_12_expand_bias_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  block_12_expand_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_12_expand_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  block_12_expand_relu_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_12_expand_relu_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  block_12_expand_scratch0, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_12_expand_scratch0_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  block_12_expand_weights, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 192), AI_STRIDE_INIT(4, 4, 128, 24576, 24576),
  1, &block_12_expand_weights_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  block_12_project_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_12_project_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  block_12_project_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 4, 4, 128, 1024),
  1, &block_12_project_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  block_12_project_scratch0, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_12_project_scratch0_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  block_12_project_weights, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 32), AI_STRIDE_INIT(4, 4, 768, 24576, 24576),
  1, &block_12_project_weights_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  block_13_depthwise_bias, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_13_depthwise_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  block_13_depthwise_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 4, 4), AI_STRIDE_INIT(4, 4, 4, 768, 3072),
  1, &block_13_depthwise_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  block_13_depthwise_relu_output, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 4, 4), AI_STRIDE_INIT(4, 4, 4, 768, 3072),
  1, &block_13_depthwise_relu_output_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  block_13_depthwise_weights, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 192), AI_STRIDE_INIT(4, 1, 192, 192, 192),
  1, &block_13_depthwise_weights_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  block_13_expand_bias, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_13_expand_bias_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  block_13_expand_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_13_expand_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  block_13_expand_relu_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 8, 8), AI_STRIDE_INIT(4, 4, 4, 768, 6144),
  1, &block_13_expand_relu_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  block_13_expand_scratch0, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &block_13_expand_scratch0_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  block_13_expand_weights, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 192), AI_STRIDE_INIT(4, 4, 128, 24576, 24576),
  1, &block_13_expand_weights_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  block_13_project_bias, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &block_13_project_bias_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  block_13_project_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 4, 4, 224, 896),
  1, &block_13_project_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  block_13_project_scratch0, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &block_13_project_scratch0_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  block_13_project_weights, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 192, 1, 1, 56), AI_STRIDE_INIT(4, 4, 768, 43008, 43008),
  1, &block_13_project_weights_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  block_14_add_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 4, 4, 224, 896),
  1, &block_14_add_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  block_14_depthwise_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_14_depthwise_bias_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  block_14_depthwise_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_14_depthwise_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  block_14_depthwise_relu_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_14_depthwise_relu_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  block_14_depthwise_weights, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 336), AI_STRIDE_INIT(4, 1, 336, 336, 336),
  1, &block_14_depthwise_weights_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  block_14_expand_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_14_expand_bias_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  block_14_expand_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_14_expand_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  block_14_expand_relu_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_14_expand_relu_output_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  block_14_expand_scratch0, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &block_14_expand_scratch0_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  block_14_expand_weights, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 56, 1, 1, 336), AI_STRIDE_INIT(4, 4, 224, 75264, 75264),
  1, &block_14_expand_weights_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  block_14_project_bias, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &block_14_project_bias_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  block_14_project_output, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 4, 4, 224, 896),
  1, &block_14_project_output_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  block_14_project_scratch0, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_14_project_scratch0_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  block_14_project_weights, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 336, 1, 1, 56), AI_STRIDE_INIT(4, 4, 1344, 75264, 75264),
  1, &block_14_project_weights_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  block_15_add_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 4, 4, 224, 896),
  1, &block_15_add_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  block_15_depthwise_bias, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_15_depthwise_bias_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  block_15_depthwise_output, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_15_depthwise_output_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  block_15_depthwise_relu_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_15_depthwise_relu_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  block_15_depthwise_weights, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 336), AI_STRIDE_INIT(4, 1, 336, 336, 336),
  1, &block_15_depthwise_weights_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  block_15_expand_bias, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_15_expand_bias_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  block_15_expand_output, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_15_expand_output_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  block_15_expand_relu_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_15_expand_relu_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  block_15_expand_scratch0, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &block_15_expand_scratch0_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  block_15_expand_weights, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 56, 1, 1, 336), AI_STRIDE_INIT(4, 4, 224, 75264, 75264),
  1, &block_15_expand_weights_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  block_15_project_bias, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &block_15_project_bias_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  block_15_project_output, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 4, 4), AI_STRIDE_INIT(4, 4, 4, 224, 896),
  1, &block_15_project_output_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  block_15_project_scratch0, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_15_project_scratch0_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  block_15_project_weights, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 336, 1, 1, 56), AI_STRIDE_INIT(4, 4, 1344, 75264, 75264),
  1, &block_15_project_weights_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  block_16_depthwise_bias, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_16_depthwise_bias_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  block_16_depthwise_output, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_16_depthwise_output_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  block_16_depthwise_relu_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_16_depthwise_relu_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  block_16_depthwise_weights, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 336), AI_STRIDE_INIT(4, 1, 336, 336, 336),
  1, &block_16_depthwise_weights_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  block_16_expand_bias, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_16_expand_bias_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  block_16_expand_output, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_16_expand_output_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  block_16_expand_relu_output, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 4, 4), AI_STRIDE_INIT(4, 4, 4, 1344, 5376),
  1, &block_16_expand_relu_output_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  block_16_expand_scratch0, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 56, 1, 1), AI_STRIDE_INIT(4, 4, 4, 224, 224),
  1, &block_16_expand_scratch0_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  block_16_expand_weights, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 56, 1, 1, 336), AI_STRIDE_INIT(4, 4, 224, 75264, 75264),
  1, &block_16_expand_weights_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  block_16_project_bias, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 1, 1), AI_STRIDE_INIT(4, 4, 4, 448, 448),
  1, &block_16_project_bias_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  block_16_project_output, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 112, 4, 4), AI_STRIDE_INIT(4, 4, 4, 448, 1792),
  1, &block_16_project_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  block_16_project_scratch0, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 336, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1344, 1344),
  1, &block_16_project_scratch0_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  block_16_project_weights, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 336, 1, 1, 112), AI_STRIDE_INIT(4, 4, 1344, 150528, 150528),
  1, &block_16_project_weights_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  block_1_depthwise_bias, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_1_depthwise_bias_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  block_1_depthwise_output, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_1_depthwise_output_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  block_1_depthwise_relu_output, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_1_depthwise_relu_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  block_1_depthwise_weights, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 48), AI_STRIDE_INIT(4, 1, 48, 48, 48),
  1, &block_1_depthwise_weights_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  block_1_expand_bias, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_1_expand_bias_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  block_1_expand_output, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 64, 64), AI_STRIDE_INIT(4, 4, 4, 192, 12288),
  1, &block_1_expand_output_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  block_1_expand_relu_output, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 64, 64), AI_STRIDE_INIT(4, 4, 4, 192, 12288),
  1, &block_1_expand_relu_output_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  block_1_expand_scratch0, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &block_1_expand_scratch0_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  block_1_expand_weights, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 48), AI_STRIDE_INIT(4, 4, 32, 1536, 1536),
  1, &block_1_expand_weights_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  block_1_project_bias, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &block_1_project_bias_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  block_1_project_output, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 4, 4, 32, 1024),
  1, &block_1_project_output_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  block_1_project_scratch0, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_1_project_scratch0_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  block_1_project_weights, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 48, 1, 1, 8), AI_STRIDE_INIT(4, 4, 192, 1536, 1536),
  1, &block_1_project_weights_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  block_2_add_output, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 4, 4, 32, 1024),
  1, &block_2_add_output_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  block_2_depthwise_bias, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_2_depthwise_bias_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  block_2_depthwise_output, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_2_depthwise_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  block_2_depthwise_relu_output, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_2_depthwise_relu_output_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  block_2_depthwise_weights, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 48), AI_STRIDE_INIT(4, 1, 48, 48, 48),
  1, &block_2_depthwise_weights_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  block_2_expand_bias, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_2_expand_bias_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  block_2_expand_output, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_2_expand_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  block_2_expand_relu_output, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_2_expand_relu_output_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  block_2_expand_scratch0, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &block_2_expand_scratch0_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  block_2_expand_weights, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 48), AI_STRIDE_INIT(4, 4, 32, 1536, 1536),
  1, &block_2_expand_weights_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  block_2_project_bias, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &block_2_project_bias_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  block_2_project_output, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 4, 4, 32, 1024),
  1, &block_2_project_output_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  block_2_project_scratch0, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_2_project_scratch0_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  block_2_project_weights, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 48, 1, 1, 8), AI_STRIDE_INIT(4, 4, 192, 1536, 1536),
  1, &block_2_project_weights_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  block_3_depthwise_bias, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_3_depthwise_bias_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  block_3_depthwise_output, AI_STATIC,
  132, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 16, 16), AI_STRIDE_INIT(4, 4, 4, 192, 3072),
  1, &block_3_depthwise_output_array, NULL)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  block_3_depthwise_relu_output, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 16, 16), AI_STRIDE_INIT(4, 4, 4, 192, 3072),
  1, &block_3_depthwise_relu_output_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  block_3_depthwise_weights, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 48), AI_STRIDE_INIT(4, 1, 48, 48, 48),
  1, &block_3_depthwise_weights_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  block_3_expand_bias, AI_STATIC,
  135, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_3_expand_bias_array, NULL)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  block_3_expand_output, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_3_expand_output_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  block_3_expand_relu_output, AI_STATIC,
  137, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 32, 32), AI_STRIDE_INIT(4, 4, 4, 192, 6144),
  1, &block_3_expand_relu_output_array, NULL)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  block_3_expand_scratch0, AI_STATIC,
  138, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &block_3_expand_scratch0_array, NULL)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  block_3_expand_weights, AI_STATIC,
  139, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 48), AI_STRIDE_INIT(4, 4, 32, 1536, 1536),
  1, &block_3_expand_weights_array, NULL)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  block_3_project_bias, AI_STATIC,
  140, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_3_project_bias_array, NULL)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  block_3_project_output, AI_STATIC,
  141, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_3_project_output_array, NULL)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  block_3_project_scratch0, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &block_3_project_scratch0_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  block_3_project_weights, AI_STATIC,
  143, 0x0,
  AI_SHAPE_INIT(4, 48, 1, 1, 16), AI_STRIDE_INIT(4, 4, 192, 3072, 3072),
  1, &block_3_project_weights_array, NULL)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  block_4_add_output, AI_STATIC,
  144, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_4_add_output_array, NULL)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  block_4_depthwise_bias, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_4_depthwise_bias_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  block_4_depthwise_output, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_4_depthwise_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  block_4_depthwise_relu_output, AI_STATIC,
  147, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_4_depthwise_relu_output_array, NULL)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  block_4_depthwise_weights, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 96), AI_STRIDE_INIT(4, 1, 96, 96, 96),
  1, &block_4_depthwise_weights_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  block_4_expand_bias, AI_STATIC,
  149, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_4_expand_bias_array, NULL)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  block_4_expand_output, AI_STATIC,
  150, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_4_expand_output_array, NULL)

/* Tensor #151 */
AI_TENSOR_OBJ_DECLARE(
  block_4_expand_relu_output, AI_STATIC,
  151, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_4_expand_relu_output_array, NULL)

/* Tensor #152 */
AI_TENSOR_OBJ_DECLARE(
  block_4_expand_scratch0, AI_STATIC,
  152, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_4_expand_scratch0_array, NULL)

/* Tensor #153 */
AI_TENSOR_OBJ_DECLARE(
  block_4_expand_weights, AI_STATIC,
  153, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 96), AI_STRIDE_INIT(4, 4, 64, 6144, 6144),
  1, &block_4_expand_weights_array, NULL)

/* Tensor #154 */
AI_TENSOR_OBJ_DECLARE(
  block_4_project_bias, AI_STATIC,
  154, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_4_project_bias_array, NULL)

/* Tensor #155 */
AI_TENSOR_OBJ_DECLARE(
  block_4_project_output, AI_STATIC,
  155, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_4_project_output_array, NULL)

/* Tensor #156 */
AI_TENSOR_OBJ_DECLARE(
  block_4_project_scratch0, AI_STATIC,
  156, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_4_project_scratch0_array, NULL)

/* Tensor #157 */
AI_TENSOR_OBJ_DECLARE(
  block_4_project_weights, AI_STATIC,
  157, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 16), AI_STRIDE_INIT(4, 4, 384, 6144, 6144),
  1, &block_4_project_weights_array, NULL)

/* Tensor #158 */
AI_TENSOR_OBJ_DECLARE(
  block_5_add_output, AI_STATIC,
  158, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_5_add_output_array, NULL)

/* Tensor #159 */
AI_TENSOR_OBJ_DECLARE(
  block_5_depthwise_bias, AI_STATIC,
  159, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_5_depthwise_bias_array, NULL)

/* Tensor #160 */
AI_TENSOR_OBJ_DECLARE(
  block_5_depthwise_output, AI_STATIC,
  160, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_5_depthwise_output_array, NULL)

/* Tensor #161 */
AI_TENSOR_OBJ_DECLARE(
  block_5_depthwise_relu_output, AI_STATIC,
  161, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_5_depthwise_relu_output_array, NULL)

/* Tensor #162 */
AI_TENSOR_OBJ_DECLARE(
  block_5_depthwise_weights, AI_STATIC,
  162, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 96), AI_STRIDE_INIT(4, 1, 96, 96, 96),
  1, &block_5_depthwise_weights_array, NULL)

/* Tensor #163 */
AI_TENSOR_OBJ_DECLARE(
  block_5_expand_bias, AI_STATIC,
  163, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_5_expand_bias_array, NULL)

/* Tensor #164 */
AI_TENSOR_OBJ_DECLARE(
  block_5_expand_output, AI_STATIC,
  164, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_5_expand_output_array, NULL)

/* Tensor #165 */
AI_TENSOR_OBJ_DECLARE(
  block_5_expand_relu_output, AI_STATIC,
  165, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_5_expand_relu_output_array, NULL)

/* Tensor #166 */
AI_TENSOR_OBJ_DECLARE(
  block_5_expand_scratch0, AI_STATIC,
  166, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_5_expand_scratch0_array, NULL)

/* Tensor #167 */
AI_TENSOR_OBJ_DECLARE(
  block_5_expand_weights, AI_STATIC,
  167, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 96), AI_STRIDE_INIT(4, 4, 64, 6144, 6144),
  1, &block_5_expand_weights_array, NULL)

/* Tensor #168 */
AI_TENSOR_OBJ_DECLARE(
  block_5_project_bias, AI_STATIC,
  168, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_5_project_bias_array, NULL)

/* Tensor #169 */
AI_TENSOR_OBJ_DECLARE(
  block_5_project_output, AI_STATIC,
  169, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 4, 4, 64, 1024),
  1, &block_5_project_output_array, NULL)

/* Tensor #170 */
AI_TENSOR_OBJ_DECLARE(
  block_5_project_scratch0, AI_STATIC,
  170, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_5_project_scratch0_array, NULL)

/* Tensor #171 */
AI_TENSOR_OBJ_DECLARE(
  block_5_project_weights, AI_STATIC,
  171, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 16), AI_STRIDE_INIT(4, 4, 384, 6144, 6144),
  1, &block_5_project_weights_array, NULL)

/* Tensor #172 */
AI_TENSOR_OBJ_DECLARE(
  block_6_depthwise_bias, AI_STATIC,
  172, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_6_depthwise_bias_array, NULL)

/* Tensor #173 */
AI_TENSOR_OBJ_DECLARE(
  block_6_depthwise_output, AI_STATIC,
  173, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 8, 8), AI_STRIDE_INIT(4, 4, 4, 384, 3072),
  1, &block_6_depthwise_output_array, NULL)

/* Tensor #174 */
AI_TENSOR_OBJ_DECLARE(
  block_6_depthwise_relu_output, AI_STATIC,
  174, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 8, 8), AI_STRIDE_INIT(4, 4, 4, 384, 3072),
  1, &block_6_depthwise_relu_output_array, NULL)

/* Tensor #175 */
AI_TENSOR_OBJ_DECLARE(
  block_6_depthwise_weights, AI_STATIC,
  175, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 96), AI_STRIDE_INIT(4, 1, 96, 96, 96),
  1, &block_6_depthwise_weights_array, NULL)

/* Tensor #176 */
AI_TENSOR_OBJ_DECLARE(
  block_6_expand_bias, AI_STATIC,
  176, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_6_expand_bias_array, NULL)

/* Tensor #177 */
AI_TENSOR_OBJ_DECLARE(
  block_6_expand_output, AI_STATIC,
  177, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_6_expand_output_array, NULL)

/* Tensor #178 */
AI_TENSOR_OBJ_DECLARE(
  block_6_expand_relu_output, AI_STATIC,
  178, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 16, 16), AI_STRIDE_INIT(4, 4, 4, 384, 6144),
  1, &block_6_expand_relu_output_array, NULL)

/* Tensor #179 */
AI_TENSOR_OBJ_DECLARE(
  block_6_expand_scratch0, AI_STATIC,
  179, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &block_6_expand_scratch0_array, NULL)

/* Tensor #180 */
AI_TENSOR_OBJ_DECLARE(
  block_6_expand_weights, AI_STATIC,
  180, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 96), AI_STRIDE_INIT(4, 4, 64, 6144, 6144),
  1, &block_6_expand_weights_array, NULL)

/* Tensor #181 */
AI_TENSOR_OBJ_DECLARE(
  block_6_project_bias, AI_STATIC,
  181, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_6_project_bias_array, NULL)

/* Tensor #182 */
AI_TENSOR_OBJ_DECLARE(
  block_6_project_output, AI_STATIC,
  182, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_6_project_output_array, NULL)

/* Tensor #183 */
AI_TENSOR_OBJ_DECLARE(
  block_6_project_scratch0, AI_STATIC,
  183, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &block_6_project_scratch0_array, NULL)

/* Tensor #184 */
AI_TENSOR_OBJ_DECLARE(
  block_6_project_weights, AI_STATIC,
  184, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 1, 24), AI_STRIDE_INIT(4, 4, 384, 9216, 9216),
  1, &block_6_project_weights_array, NULL)

/* Tensor #185 */
AI_TENSOR_OBJ_DECLARE(
  block_7_add_output, AI_STATIC,
  185, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_7_add_output_array, NULL)

/* Tensor #186 */
AI_TENSOR_OBJ_DECLARE(
  block_7_depthwise_bias, AI_STATIC,
  186, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_7_depthwise_bias_array, NULL)

/* Tensor #187 */
AI_TENSOR_OBJ_DECLARE(
  block_7_depthwise_output, AI_STATIC,
  187, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_7_depthwise_output_array, NULL)

/* Tensor #188 */
AI_TENSOR_OBJ_DECLARE(
  block_7_depthwise_relu_output, AI_STATIC,
  188, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_7_depthwise_relu_output_array, NULL)

/* Tensor #189 */
AI_TENSOR_OBJ_DECLARE(
  block_7_depthwise_weights, AI_STATIC,
  189, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 144), AI_STRIDE_INIT(4, 1, 144, 144, 144),
  1, &block_7_depthwise_weights_array, NULL)

/* Tensor #190 */
AI_TENSOR_OBJ_DECLARE(
  block_7_expand_bias, AI_STATIC,
  190, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_7_expand_bias_array, NULL)

/* Tensor #191 */
AI_TENSOR_OBJ_DECLARE(
  block_7_expand_output, AI_STATIC,
  191, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_7_expand_output_array, NULL)

/* Tensor #192 */
AI_TENSOR_OBJ_DECLARE(
  block_7_expand_relu_output, AI_STATIC,
  192, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_7_expand_relu_output_array, NULL)

/* Tensor #193 */
AI_TENSOR_OBJ_DECLARE(
  block_7_expand_scratch0, AI_STATIC,
  193, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_7_expand_scratch0_array, NULL)

/* Tensor #194 */
AI_TENSOR_OBJ_DECLARE(
  block_7_expand_weights, AI_STATIC,
  194, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 144), AI_STRIDE_INIT(4, 4, 96, 13824, 13824),
  1, &block_7_expand_weights_array, NULL)

/* Tensor #195 */
AI_TENSOR_OBJ_DECLARE(
  block_7_project_bias, AI_STATIC,
  195, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_7_project_bias_array, NULL)

/* Tensor #196 */
AI_TENSOR_OBJ_DECLARE(
  block_7_project_output, AI_STATIC,
  196, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_7_project_output_array, NULL)

/* Tensor #197 */
AI_TENSOR_OBJ_DECLARE(
  block_7_project_scratch0, AI_STATIC,
  197, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_7_project_scratch0_array, NULL)

/* Tensor #198 */
AI_TENSOR_OBJ_DECLARE(
  block_7_project_weights, AI_STATIC,
  198, 0x0,
  AI_SHAPE_INIT(4, 144, 1, 1, 24), AI_STRIDE_INIT(4, 4, 576, 13824, 13824),
  1, &block_7_project_weights_array, NULL)

/* Tensor #199 */
AI_TENSOR_OBJ_DECLARE(
  block_8_add_output, AI_STATIC,
  199, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_8_add_output_array, NULL)

/* Tensor #200 */
AI_TENSOR_OBJ_DECLARE(
  block_8_depthwise_bias, AI_STATIC,
  200, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_8_depthwise_bias_array, NULL)

/* Tensor #201 */
AI_TENSOR_OBJ_DECLARE(
  block_8_depthwise_output, AI_STATIC,
  201, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_8_depthwise_output_array, NULL)

/* Tensor #202 */
AI_TENSOR_OBJ_DECLARE(
  block_8_depthwise_relu_output, AI_STATIC,
  202, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_8_depthwise_relu_output_array, NULL)

/* Tensor #203 */
AI_TENSOR_OBJ_DECLARE(
  block_8_depthwise_weights, AI_STATIC,
  203, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 144), AI_STRIDE_INIT(4, 1, 144, 144, 144),
  1, &block_8_depthwise_weights_array, NULL)

/* Tensor #204 */
AI_TENSOR_OBJ_DECLARE(
  block_8_expand_bias, AI_STATIC,
  204, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_8_expand_bias_array, NULL)

/* Tensor #205 */
AI_TENSOR_OBJ_DECLARE(
  block_8_expand_output, AI_STATIC,
  205, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_8_expand_output_array, NULL)

/* Tensor #206 */
AI_TENSOR_OBJ_DECLARE(
  block_8_expand_relu_output, AI_STATIC,
  206, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_8_expand_relu_output_array, NULL)

/* Tensor #207 */
AI_TENSOR_OBJ_DECLARE(
  block_8_expand_scratch0, AI_STATIC,
  207, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_8_expand_scratch0_array, NULL)

/* Tensor #208 */
AI_TENSOR_OBJ_DECLARE(
  block_8_expand_weights, AI_STATIC,
  208, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 144), AI_STRIDE_INIT(4, 4, 96, 13824, 13824),
  1, &block_8_expand_weights_array, NULL)

/* Tensor #209 */
AI_TENSOR_OBJ_DECLARE(
  block_8_project_bias, AI_STATIC,
  209, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_8_project_bias_array, NULL)

/* Tensor #210 */
AI_TENSOR_OBJ_DECLARE(
  block_8_project_output, AI_STATIC,
  210, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_8_project_output_array, NULL)

/* Tensor #211 */
AI_TENSOR_OBJ_DECLARE(
  block_8_project_scratch0, AI_STATIC,
  211, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_8_project_scratch0_array, NULL)

/* Tensor #212 */
AI_TENSOR_OBJ_DECLARE(
  block_8_project_weights, AI_STATIC,
  212, 0x0,
  AI_SHAPE_INIT(4, 144, 1, 1, 24), AI_STRIDE_INIT(4, 4, 576, 13824, 13824),
  1, &block_8_project_weights_array, NULL)

/* Tensor #213 */
AI_TENSOR_OBJ_DECLARE(
  block_9_add_output, AI_STATIC,
  213, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_9_add_output_array, NULL)

/* Tensor #214 */
AI_TENSOR_OBJ_DECLARE(
  block_9_depthwise_bias, AI_STATIC,
  214, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_9_depthwise_bias_array, NULL)

/* Tensor #215 */
AI_TENSOR_OBJ_DECLARE(
  block_9_depthwise_output, AI_STATIC,
  215, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_9_depthwise_output_array, NULL)

/* Tensor #216 */
AI_TENSOR_OBJ_DECLARE(
  block_9_depthwise_relu_output, AI_STATIC,
  216, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_9_depthwise_relu_output_array, NULL)

/* Tensor #217 */
AI_TENSOR_OBJ_DECLARE(
  block_9_depthwise_weights, AI_STATIC,
  217, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 144), AI_STRIDE_INIT(4, 1, 144, 144, 144),
  1, &block_9_depthwise_weights_array, NULL)

/* Tensor #218 */
AI_TENSOR_OBJ_DECLARE(
  block_9_expand_bias, AI_STATIC,
  218, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_9_expand_bias_array, NULL)

/* Tensor #219 */
AI_TENSOR_OBJ_DECLARE(
  block_9_expand_output, AI_STATIC,
  219, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_9_expand_output_array, NULL)

/* Tensor #220 */
AI_TENSOR_OBJ_DECLARE(
  block_9_expand_relu_output, AI_STATIC,
  220, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 8, 8), AI_STRIDE_INIT(4, 4, 4, 576, 4608),
  1, &block_9_expand_relu_output_array, NULL)

/* Tensor #221 */
AI_TENSOR_OBJ_DECLARE(
  block_9_expand_scratch0, AI_STATIC,
  221, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_9_expand_scratch0_array, NULL)

/* Tensor #222 */
AI_TENSOR_OBJ_DECLARE(
  block_9_expand_weights, AI_STATIC,
  222, 0x0,
  AI_SHAPE_INIT(4, 24, 1, 1, 144), AI_STRIDE_INIT(4, 4, 96, 13824, 13824),
  1, &block_9_expand_weights_array, NULL)

/* Tensor #223 */
AI_TENSOR_OBJ_DECLARE(
  block_9_project_bias, AI_STATIC,
  223, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 1, 1), AI_STRIDE_INIT(4, 4, 4, 96, 96),
  1, &block_9_project_bias_array, NULL)

/* Tensor #224 */
AI_TENSOR_OBJ_DECLARE(
  block_9_project_output, AI_STATIC,
  224, 0x0,
  AI_SHAPE_INIT(4, 1, 24, 8, 8), AI_STRIDE_INIT(4, 4, 4, 96, 768),
  1, &block_9_project_output_array, NULL)

/* Tensor #225 */
AI_TENSOR_OBJ_DECLARE(
  block_9_project_scratch0, AI_STATIC,
  225, 0x0,
  AI_SHAPE_INIT(4, 1, 144, 1, 1), AI_STRIDE_INIT(4, 4, 4, 576, 576),
  1, &block_9_project_scratch0_array, NULL)

/* Tensor #226 */
AI_TENSOR_OBJ_DECLARE(
  block_9_project_weights, AI_STATIC,
  226, 0x0,
  AI_SHAPE_INIT(4, 144, 1, 1, 24), AI_STRIDE_INIT(4, 4, 576, 13824, 13824),
  1, &block_9_project_weights_array, NULL)

/* Tensor #227 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_depthwise_bias, AI_STATIC,
  227, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expanded_conv_depthwise_bias_array, NULL)

/* Tensor #228 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_depthwise_output, AI_STATIC,
  228, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 64, 64), AI_STRIDE_INIT(4, 4, 4, 64, 4096),
  1, &expanded_conv_depthwise_output_array, NULL)

/* Tensor #229 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_depthwise_relu_output, AI_STATIC,
  229, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 64, 64), AI_STRIDE_INIT(4, 4, 4, 64, 4096),
  1, &expanded_conv_depthwise_relu_output_array, NULL)

/* Tensor #230 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_depthwise_weights, AI_STATIC,
  230, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1, 16, 16, 16),
  1, &expanded_conv_depthwise_weights_array, NULL)

/* Tensor #231 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_project_bias, AI_STATIC,
  231, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expanded_conv_project_bias_array, NULL)

/* Tensor #232 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_project_output, AI_STATIC,
  232, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 64, 64), AI_STRIDE_INIT(4, 4, 4, 32, 2048),
  1, &expanded_conv_project_output_array, NULL)

/* Tensor #233 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_project_scratch0, AI_STATIC,
  233, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expanded_conv_project_scratch0_array, NULL)

/* Tensor #234 */
AI_TENSOR_OBJ_DECLARE(
  expanded_conv_project_weights, AI_STATIC,
  234, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 8), AI_STRIDE_INIT(4, 4, 64, 512, 512),
  1, &expanded_conv_project_weights_array, NULL)

/* Tensor #235 */
AI_TENSOR_OBJ_DECLARE(
  global_average_pooling2d_pool_output, AI_STATIC,
  235, 0x0,
  AI_SHAPE_INIT(4, 1, 1280, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5120, 5120),
  1, &global_average_pooling2d_pool_output_array, NULL)

/* Tensor #236 */
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  236, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 128, 128), AI_STRIDE_INIT(4, 4, 4, 12, 1536),
  1, &input_1_output_array, NULL)

/* Tensor #237 */
AI_TENSOR_OBJ_DECLARE(
  out_relu_output, AI_STATIC,
  237, 0x0,
  AI_SHAPE_INIT(4, 1, 1280, 4, 4), AI_STRIDE_INIT(4, 4, 4, 5120, 20480),
  1, &out_relu_output_array, NULL)

/* Tensor #238 */
AI_TENSOR_OBJ_DECLARE(
  predictions_dense_bias, AI_STATIC,
  238, 0x0,
  AI_SHAPE_INIT(4, 1, 1000, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4000, 4000),
  1, &predictions_dense_bias_array, NULL)

/* Tensor #239 */
AI_TENSOR_OBJ_DECLARE(
  predictions_dense_output, AI_STATIC,
  239, 0x0,
  AI_SHAPE_INIT(4, 1, 1000, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4000, 4000),
  1, &predictions_dense_output_array, NULL)

/* Tensor #240 */
AI_TENSOR_OBJ_DECLARE(
  predictions_dense_weights, AI_STATIC,
  240, 0x0,
  AI_SHAPE_INIT(4, 1280, 1000, 1, 1), AI_STRIDE_INIT(4, 1, 640, 640000, 640000),
  1, &predictions_dense_weights_array, NULL)

/* Tensor #241 */
AI_TENSOR_OBJ_DECLARE(
  predictions_output, AI_STATIC,
  241, 0x0,
  AI_SHAPE_INIT(4, 1, 1000, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4000, 4000),
  1, &predictions_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  predictions_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &predictions_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &predictions_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  predictions_layer, 155,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &predictions_chain,
  NULL, &predictions_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  predictions_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling2d_pool_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &predictions_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &predictions_dense_weights, &predictions_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  predictions_dense_layer, 155,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &predictions_dense_chain,
  NULL, &predictions_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  global_average_pooling2d_pool_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &out_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_average_pooling2d_pool_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  global_average_pooling2d_pool_layer, 154,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &global_average_pooling2d_pool_chain,
  NULL, &predictions_dense_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(4, 4), 
  .pool_stride = AI_SHAPE_2D_INIT(4, 4), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_float out_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    out_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    out_relu_nl_params_data, out_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  out_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &out_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  out_relu_layer, 153,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &out_relu_chain,
  NULL, &global_average_pooling2d_pool_layer, AI_STATIC, 
  .nl_params = &out_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv_1_weights, &Conv_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &Conv_1_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  Conv_1_layer, 152,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &Conv_1_chain,
  NULL, &out_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_16_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_16_project_weights, &block_16_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_16_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_16_project_layer, 150,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_16_project_chain,
  NULL, &Conv_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_16_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_16_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_16_depthwise_relu_nl_params_data, block_16_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_16_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_16_depthwise_relu_layer, 148,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_16_depthwise_relu_chain,
  NULL, &block_16_project_layer, AI_STATIC, 
  .nl_params = &block_16_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_16_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_16_depthwise_weights, &block_16_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_16_depthwise_layer, 147,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_16_depthwise_chain,
  NULL, &block_16_depthwise_relu_layer, AI_STATIC, 
  .groups = 336, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_16_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_16_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_16_expand_relu_nl_params_data, block_16_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_16_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_16_expand_relu_layer, 145,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_16_expand_relu_chain,
  NULL, &block_16_depthwise_layer, AI_STATIC, 
  .nl_params = &block_16_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_16_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_16_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_16_expand_weights, &block_16_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_16_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_16_expand_layer, 144,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_16_expand_chain,
  NULL, &block_16_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_15_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_14_add_output, &block_15_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_15_add_layer, 142,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_15_add_chain,
  NULL, &block_16_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_15_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_15_project_weights, &block_15_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_15_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_15_project_layer, 141,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_15_project_chain,
  NULL, &block_15_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_15_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_15_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_15_depthwise_relu_nl_params_data, block_15_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_15_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_15_depthwise_relu_layer, 139,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_15_depthwise_relu_chain,
  NULL, &block_15_project_layer, AI_STATIC, 
  .nl_params = &block_15_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_15_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_15_depthwise_weights, &block_15_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_15_depthwise_layer, 138,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_15_depthwise_chain,
  NULL, &block_15_depthwise_relu_layer, AI_STATIC, 
  .groups = 336, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_15_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_15_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_15_expand_relu_nl_params_data, block_15_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_15_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_15_expand_relu_layer, 136,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_15_expand_relu_chain,
  NULL, &block_15_depthwise_layer, AI_STATIC, 
  .nl_params = &block_15_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_15_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_15_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_15_expand_weights, &block_15_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_15_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_15_expand_layer, 135,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_15_expand_chain,
  NULL, &block_15_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_14_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_13_project_output, &block_14_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_14_add_layer, 133,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_14_add_chain,
  NULL, &block_15_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_14_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_14_project_weights, &block_14_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_14_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_14_project_layer, 132,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_14_project_chain,
  NULL, &block_14_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_14_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_14_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_14_depthwise_relu_nl_params_data, block_14_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_14_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_14_depthwise_relu_layer, 130,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_14_depthwise_relu_chain,
  NULL, &block_14_project_layer, AI_STATIC, 
  .nl_params = &block_14_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_14_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_14_depthwise_weights, &block_14_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_14_depthwise_layer, 129,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_14_depthwise_chain,
  NULL, &block_14_depthwise_relu_layer, AI_STATIC, 
  .groups = 336, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_14_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_14_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_14_expand_relu_nl_params_data, block_14_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_14_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_14_expand_relu_layer, 127,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_14_expand_relu_chain,
  NULL, &block_14_depthwise_layer, AI_STATIC, 
  .nl_params = &block_14_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_14_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_14_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_14_expand_weights, &block_14_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_14_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_14_expand_layer, 126,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_14_expand_chain,
  NULL, &block_14_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_13_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_13_project_weights, &block_13_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_13_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_13_project_layer, 124,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_13_project_chain,
  NULL, &block_14_expand_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_13_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_13_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_13_depthwise_relu_nl_params_data, block_13_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_13_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_13_depthwise_relu_layer, 122,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_13_depthwise_relu_chain,
  NULL, &block_13_project_layer, AI_STATIC, 
  .nl_params = &block_13_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_13_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_13_depthwise_weights, &block_13_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_13_depthwise_layer, 119,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_13_depthwise_chain,
  NULL, &block_13_depthwise_relu_layer, AI_STATIC, 
  .groups = 192, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_13_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_13_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_13_expand_relu_nl_params_data, block_13_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_13_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_13_expand_relu_layer, 118,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_13_expand_relu_chain,
  NULL, &block_13_depthwise_layer, AI_STATIC, 
  .nl_params = &block_13_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_13_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_13_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_13_expand_weights, &block_13_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_13_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_13_expand_layer, 117,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_13_expand_chain,
  NULL, &block_13_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_12_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_11_add_output, &block_12_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_12_add_layer, 115,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_12_add_chain,
  NULL, &block_13_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_12_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_12_project_weights, &block_12_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_12_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_12_project_layer, 114,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_12_project_chain,
  NULL, &block_12_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_12_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_12_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_12_depthwise_relu_nl_params_data, block_12_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_12_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_12_depthwise_relu_layer, 112,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_12_depthwise_relu_chain,
  NULL, &block_12_project_layer, AI_STATIC, 
  .nl_params = &block_12_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_12_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_12_depthwise_weights, &block_12_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_12_depthwise_layer, 111,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_12_depthwise_chain,
  NULL, &block_12_depthwise_relu_layer, AI_STATIC, 
  .groups = 192, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_12_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_12_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_12_expand_relu_nl_params_data, block_12_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_12_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_12_expand_relu_layer, 109,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_12_expand_relu_chain,
  NULL, &block_12_depthwise_layer, AI_STATIC, 
  .nl_params = &block_12_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_12_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_12_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_12_expand_weights, &block_12_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_12_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_12_expand_layer, 108,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_12_expand_chain,
  NULL, &block_12_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_11_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_10_project_output, &block_11_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_11_add_layer, 106,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_11_add_chain,
  NULL, &block_12_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_11_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_11_project_weights, &block_11_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_11_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_11_project_layer, 105,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_11_project_chain,
  NULL, &block_11_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_11_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_11_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_11_depthwise_relu_nl_params_data, block_11_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_11_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_11_depthwise_relu_layer, 103,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_11_depthwise_relu_chain,
  NULL, &block_11_project_layer, AI_STATIC, 
  .nl_params = &block_11_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_11_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_11_depthwise_weights, &block_11_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_11_depthwise_layer, 102,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_11_depthwise_chain,
  NULL, &block_11_depthwise_relu_layer, AI_STATIC, 
  .groups = 192, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_11_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_11_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_11_expand_relu_nl_params_data, block_11_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_11_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_11_expand_relu_layer, 100,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_11_expand_relu_chain,
  NULL, &block_11_depthwise_layer, AI_STATIC, 
  .nl_params = &block_11_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_11_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_11_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_11_expand_weights, &block_11_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_11_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_11_expand_layer, 99,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_11_expand_chain,
  NULL, &block_11_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_10_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_10_project_weights, &block_10_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_10_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_10_project_layer, 97,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_10_project_chain,
  NULL, &block_11_expand_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_10_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_10_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_10_depthwise_relu_nl_params_data, block_10_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_10_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_10_depthwise_relu_layer, 95,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_10_depthwise_relu_chain,
  NULL, &block_10_project_layer, AI_STATIC, 
  .nl_params = &block_10_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_10_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_10_depthwise_weights, &block_10_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_10_depthwise_layer, 94,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_10_depthwise_chain,
  NULL, &block_10_depthwise_relu_layer, AI_STATIC, 
  .groups = 144, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_10_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_10_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_10_expand_relu_nl_params_data, block_10_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_10_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_10_expand_relu_layer, 92,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_10_expand_relu_chain,
  NULL, &block_10_depthwise_layer, AI_STATIC, 
  .nl_params = &block_10_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_10_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_10_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_10_expand_weights, &block_10_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_10_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_10_expand_layer, 91,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_10_expand_chain,
  NULL, &block_10_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_9_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_8_add_output, &block_9_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_9_add_layer, 89,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_9_add_chain,
  NULL, &block_10_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_9_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_9_project_weights, &block_9_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_9_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_9_project_layer, 88,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_9_project_chain,
  NULL, &block_9_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_9_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_9_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_9_depthwise_relu_nl_params_data, block_9_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_9_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_9_depthwise_relu_layer, 86,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_9_depthwise_relu_chain,
  NULL, &block_9_project_layer, AI_STATIC, 
  .nl_params = &block_9_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_9_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_9_depthwise_weights, &block_9_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_9_depthwise_layer, 85,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_9_depthwise_chain,
  NULL, &block_9_depthwise_relu_layer, AI_STATIC, 
  .groups = 144, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_9_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_9_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_9_expand_relu_nl_params_data, block_9_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_9_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_9_expand_relu_layer, 83,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_9_expand_relu_chain,
  NULL, &block_9_depthwise_layer, AI_STATIC, 
  .nl_params = &block_9_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_9_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_9_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_9_expand_weights, &block_9_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_9_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_9_expand_layer, 82,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_9_expand_chain,
  NULL, &block_9_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_8_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_7_add_output, &block_8_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_8_add_layer, 80,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_8_add_chain,
  NULL, &block_9_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_8_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_8_project_weights, &block_8_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_8_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_8_project_layer, 79,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_8_project_chain,
  NULL, &block_8_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_8_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_8_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_8_depthwise_relu_nl_params_data, block_8_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_8_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_8_depthwise_relu_layer, 77,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_8_depthwise_relu_chain,
  NULL, &block_8_project_layer, AI_STATIC, 
  .nl_params = &block_8_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_8_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_8_depthwise_weights, &block_8_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_8_depthwise_layer, 76,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_8_depthwise_chain,
  NULL, &block_8_depthwise_relu_layer, AI_STATIC, 
  .groups = 144, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_8_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_8_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_8_expand_relu_nl_params_data, block_8_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_8_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_8_expand_relu_layer, 74,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_8_expand_relu_chain,
  NULL, &block_8_depthwise_layer, AI_STATIC, 
  .nl_params = &block_8_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_8_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_8_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_8_expand_weights, &block_8_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_8_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_8_expand_layer, 73,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_8_expand_chain,
  NULL, &block_8_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_7_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_6_project_output, &block_7_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_7_add_layer, 71,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_7_add_chain,
  NULL, &block_8_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_7_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_7_project_weights, &block_7_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_7_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_7_project_layer, 70,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_7_project_chain,
  NULL, &block_7_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_7_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_7_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_7_depthwise_relu_nl_params_data, block_7_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_7_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_7_depthwise_relu_layer, 68,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_7_depthwise_relu_chain,
  NULL, &block_7_project_layer, AI_STATIC, 
  .nl_params = &block_7_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_7_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_7_depthwise_weights, &block_7_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_7_depthwise_layer, 67,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_7_depthwise_chain,
  NULL, &block_7_depthwise_relu_layer, AI_STATIC, 
  .groups = 144, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_7_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_7_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_7_expand_relu_nl_params_data, block_7_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_7_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_7_expand_relu_layer, 65,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_7_expand_relu_chain,
  NULL, &block_7_depthwise_layer, AI_STATIC, 
  .nl_params = &block_7_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_7_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_7_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_7_expand_weights, &block_7_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_7_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_7_expand_layer, 64,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_7_expand_chain,
  NULL, &block_7_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_6_project_weights, &block_6_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_6_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_6_project_layer, 62,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_6_project_chain,
  NULL, &block_7_expand_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_6_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_6_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_6_depthwise_relu_nl_params_data, block_6_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_6_depthwise_relu_layer, 60,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_6_depthwise_relu_chain,
  NULL, &block_6_project_layer, AI_STATIC, 
  .nl_params = &block_6_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_6_depthwise_weights, &block_6_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_6_depthwise_layer, 57,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_6_depthwise_chain,
  NULL, &block_6_depthwise_relu_layer, AI_STATIC, 
  .groups = 96, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_6_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_6_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_6_expand_relu_nl_params_data, block_6_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_6_expand_relu_layer, 56,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_6_expand_relu_chain,
  NULL, &block_6_depthwise_layer, AI_STATIC, 
  .nl_params = &block_6_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_6_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_6_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_6_expand_weights, &block_6_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_6_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_6_expand_layer, 55,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_6_expand_chain,
  NULL, &block_6_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_4_add_output, &block_5_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_5_add_layer, 53,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_5_add_chain,
  NULL, &block_6_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_5_project_weights, &block_5_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_5_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_5_project_layer, 52,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_5_project_chain,
  NULL, &block_5_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_5_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_5_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_5_depthwise_relu_nl_params_data, block_5_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_5_depthwise_relu_layer, 50,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_5_depthwise_relu_chain,
  NULL, &block_5_project_layer, AI_STATIC, 
  .nl_params = &block_5_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_5_depthwise_weights, &block_5_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_5_depthwise_layer, 49,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_5_depthwise_chain,
  NULL, &block_5_depthwise_relu_layer, AI_STATIC, 
  .groups = 96, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_5_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_5_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_5_expand_relu_nl_params_data, block_5_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_5_expand_relu_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_5_expand_relu_chain,
  NULL, &block_5_depthwise_layer, AI_STATIC, 
  .nl_params = &block_5_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_5_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_5_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_5_expand_weights, &block_5_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_5_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_5_expand_layer, 46,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_5_expand_chain,
  NULL, &block_5_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_3_project_output, &block_4_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_add_layer, 44,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_4_add_chain,
  NULL, &block_5_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_4_project_weights, &block_4_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_4_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_4_project_layer, 43,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_4_project_chain,
  NULL, &block_4_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_4_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_4_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_4_depthwise_relu_nl_params_data, block_4_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_depthwise_relu_layer, 41,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_4_depthwise_relu_chain,
  NULL, &block_4_project_layer, AI_STATIC, 
  .nl_params = &block_4_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_4_depthwise_weights, &block_4_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_depthwise_layer, 40,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_4_depthwise_chain,
  NULL, &block_4_depthwise_relu_layer, AI_STATIC, 
  .groups = 96, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_4_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_4_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_4_expand_relu_nl_params_data, block_4_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_4_expand_relu_layer, 38,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_4_expand_relu_chain,
  NULL, &block_4_depthwise_layer, AI_STATIC, 
  .nl_params = &block_4_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_4_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_4_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_4_expand_weights, &block_4_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_4_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_4_expand_layer, 37,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_4_expand_chain,
  NULL, &block_4_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_3_project_weights, &block_3_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_3_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_3_project_layer, 35,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_3_project_chain,
  NULL, &block_4_expand_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_3_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_3_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_3_depthwise_relu_nl_params_data, block_3_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_3_depthwise_relu_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_3_depthwise_relu_chain,
  NULL, &block_3_project_layer, AI_STATIC, 
  .nl_params = &block_3_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_3_depthwise_weights, &block_3_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_3_depthwise_layer, 30,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_3_depthwise_chain,
  NULL, &block_3_depthwise_relu_layer, AI_STATIC, 
  .groups = 48, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_3_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_3_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_3_expand_relu_nl_params_data, block_3_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_3_expand_relu_layer, 29,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_3_expand_relu_chain,
  NULL, &block_3_depthwise_layer, AI_STATIC, 
  .nl_params = &block_3_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_3_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_add_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_3_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_3_expand_weights, &block_3_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_3_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_3_expand_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_3_expand_chain,
  NULL, &block_3_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_add_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_1_project_output, &block_2_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_add_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_add_layer, 26,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &block_2_add_chain,
  NULL, &block_3_expand_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_2_project_weights, &block_2_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_2_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_2_project_layer, 25,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_2_project_chain,
  NULL, &block_2_add_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_2_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_2_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_2_depthwise_relu_nl_params_data, block_2_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_depthwise_relu_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_2_depthwise_relu_chain,
  NULL, &block_2_project_layer, AI_STATIC, 
  .nl_params = &block_2_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_2_depthwise_weights, &block_2_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_depthwise_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_2_depthwise_chain,
  NULL, &block_2_depthwise_relu_layer, AI_STATIC, 
  .groups = 48, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_2_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_2_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_2_expand_relu_nl_params_data, block_2_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_2_expand_relu_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_2_expand_relu_chain,
  NULL, &block_2_depthwise_layer, AI_STATIC, 
  .nl_params = &block_2_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_2_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_2_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_2_expand_weights, &block_2_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_2_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_2_expand_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_2_expand_chain,
  NULL, &block_2_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_1_project_weights, &block_1_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_1_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_1_project_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_1_project_chain,
  NULL, &block_2_expand_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_1_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_1_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_1_depthwise_relu_nl_params_data, block_1_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_1_depthwise_relu_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_1_depthwise_relu_chain,
  NULL, &block_1_project_layer, AI_STATIC, 
  .nl_params = &block_1_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_expand_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_1_depthwise_weights, &block_1_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_1_depthwise_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &block_1_depthwise_chain,
  NULL, &block_1_depthwise_relu_layer, AI_STATIC, 
  .groups = 48, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float block_1_expand_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    block_1_expand_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    block_1_expand_relu_nl_params_data, block_1_expand_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_expand_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_expand_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  block_1_expand_relu_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &block_1_expand_relu_chain,
  NULL, &block_1_depthwise_layer, AI_STATIC, 
  .nl_params = &block_1_expand_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  block_1_expand_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expanded_conv_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &block_1_expand_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &block_1_expand_weights, &block_1_expand_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &block_1_expand_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  block_1_expand_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &block_1_expand_chain,
  NULL, &block_1_expand_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expanded_conv_project_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expanded_conv_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expanded_conv_project_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expanded_conv_project_weights, &expanded_conv_project_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expanded_conv_project_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  expanded_conv_project_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &expanded_conv_project_chain,
  NULL, &block_1_expand_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float expanded_conv_depthwise_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    expanded_conv_depthwise_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    expanded_conv_depthwise_relu_nl_params_data, expanded_conv_depthwise_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  expanded_conv_depthwise_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expanded_conv_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expanded_conv_depthwise_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expanded_conv_depthwise_relu_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &expanded_conv_depthwise_relu_chain,
  NULL, &expanded_conv_project_layer, AI_STATIC, 
  .nl_params = &expanded_conv_depthwise_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expanded_conv_depthwise_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv1_relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expanded_conv_depthwise_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expanded_conv_depthwise_weights, &expanded_conv_depthwise_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expanded_conv_depthwise_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &expanded_conv_depthwise_chain,
  NULL, &expanded_conv_depthwise_relu_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float Conv1_relu_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    Conv1_relu_nl_params, AI_ARRAY_FORMAT_FLOAT,
    Conv1_relu_nl_params_data, Conv1_relu_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv1_relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv1_relu_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv1_relu_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &Conv1_relu_chain,
  NULL, &expanded_conv_depthwise_layer, AI_STATIC, 
  .nl_params = &Conv1_relu_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv1_weights, &Conv1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &Conv1_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  Conv1_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &Conv1_chain,
  NULL, &Conv1_relu_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2200416, 1, 1),
    2200416, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 815328, 1, 1),
    815328, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &predictions_output),
  &Conv1_layer, 0xec27fc2f, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2200416, 1, 1),
      2200416, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 815328, 1, 1),
      815328, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &predictions_output),
  &Conv1_layer, 0xec27fc2f, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_1_output_array.data = AI_PTR(g_network_activations_map[0] + 356436);
    input_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 356436);
    Conv1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 553044);
    Conv1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 553044);
    Conv1_output_array.data = AI_PTR(g_network_activations_map[0] + 553152);
    Conv1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 553152);
    Conv1_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 553152);
    Conv1_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 553152);
    expanded_conv_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 291008);
    expanded_conv_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 291008);
    expanded_conv_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 291008);
    expanded_conv_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 291008);
    expanded_conv_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 553152);
    expanded_conv_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 553152);
    expanded_conv_project_output_array.data = AI_PTR(g_network_activations_map[0] + 684224);
    expanded_conv_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 684224);
    block_1_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 815296);
    block_1_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 815296);
    block_1_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 6336);
    block_1_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6336);
    block_1_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 6336);
    block_1_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6336);
    block_1_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_1_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_1_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 196608);
    block_1_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 196608);
    block_1_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_1_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_1_project_output_array.data = AI_PTR(g_network_activations_map[0] + 192);
    block_1_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 192);
    block_2_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_2_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_2_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 32960);
    block_2_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32960);
    block_2_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 229568);
    block_2_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 229568);
    block_2_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 32960);
    block_2_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32960);
    block_2_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 229568);
    block_2_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 229568);
    block_2_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_2_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_2_project_output_array.data = AI_PTR(g_network_activations_map[0] + 32960);
    block_2_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32960);
    block_2_add_output_array.data = AI_PTR(g_network_activations_map[0] + 65728);
    block_2_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 65728);
    block_3_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_3_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_3_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 98496);
    block_3_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 98496);
    block_3_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 295104);
    block_3_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 295104);
    block_3_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_3_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_3_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 49152);
    block_3_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 49152);
    block_3_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_3_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_3_project_output_array.data = AI_PTR(g_network_activations_map[0] + 192);
    block_3_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 192);
    block_4_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_4_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_4_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 16576);
    block_4_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16576);
    block_4_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 114880);
    block_4_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 114880);
    block_4_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 16576);
    block_4_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16576);
    block_4_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 114880);
    block_4_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 114880);
    block_4_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 16576);
    block_4_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 16576);
    block_4_project_output_array.data = AI_PTR(g_network_activations_map[0] + 16960);
    block_4_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16960);
    block_4_add_output_array.data = AI_PTR(g_network_activations_map[0] + 33344);
    block_4_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 33344);
    block_5_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_5_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_5_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 49728);
    block_5_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 49728);
    block_5_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 148032);
    block_5_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 148032);
    block_5_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 49728);
    block_5_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 49728);
    block_5_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 148032);
    block_5_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 148032);
    block_5_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_5_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_5_project_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    block_5_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    block_5_add_output_array.data = AI_PTR(g_network_activations_map[0] + 16768);
    block_5_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16768);
    block_6_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_6_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_6_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 33152);
    block_6_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 33152);
    block_6_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 131456);
    block_6_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 131456);
    block_6_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_6_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_6_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 24576);
    block_6_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24576);
    block_6_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_6_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_6_project_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    block_6_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    block_7_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_7_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_7_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 6528);
    block_7_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6528);
    block_7_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 43392);
    block_7_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 43392);
    block_7_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 6528);
    block_7_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6528);
    block_7_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 43392);
    block_7_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 43392);
    block_7_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 6528);
    block_7_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 6528);
    block_7_project_output_array.data = AI_PTR(g_network_activations_map[0] + 7104);
    block_7_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7104);
    block_7_add_output_array.data = AI_PTR(g_network_activations_map[0] + 13248);
    block_7_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 13248);
    block_8_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_8_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_8_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 19392);
    block_8_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 19392);
    block_8_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 56256);
    block_8_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56256);
    block_8_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 19392);
    block_8_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 19392);
    block_8_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 56256);
    block_8_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56256);
    block_8_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_8_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_8_project_output_array.data = AI_PTR(g_network_activations_map[0] + 576);
    block_8_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 576);
    block_8_add_output_array.data = AI_PTR(g_network_activations_map[0] + 6720);
    block_8_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6720);
    block_9_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_9_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_9_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 12864);
    block_9_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12864);
    block_9_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 49728);
    block_9_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 49728);
    block_9_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 12864);
    block_9_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12864);
    block_9_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 49728);
    block_9_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 49728);
    block_9_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_9_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_9_project_output_array.data = AI_PTR(g_network_activations_map[0] + 576);
    block_9_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 576);
    block_9_add_output_array.data = AI_PTR(g_network_activations_map[0] + 12864);
    block_9_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12864);
    block_10_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_10_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_10_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 19008);
    block_10_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 19008);
    block_10_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 55872);
    block_10_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 55872);
    block_10_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_10_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_10_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 36864);
    block_10_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 36864);
    block_10_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_10_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_10_project_output_array.data = AI_PTR(g_network_activations_map[0] + 576);
    block_10_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 576);
    block_11_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_11_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_11_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 8768);
    block_11_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8768);
    block_11_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 57920);
    block_11_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 57920);
    block_11_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 8768);
    block_11_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8768);
    block_11_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 57920);
    block_11_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 57920);
    block_11_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 8768);
    block_11_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 8768);
    block_11_project_output_array.data = AI_PTR(g_network_activations_map[0] + 9536);
    block_11_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9536);
    block_11_add_output_array.data = AI_PTR(g_network_activations_map[0] + 17728);
    block_11_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 17728);
    block_12_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_12_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_12_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 25920);
    block_12_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25920);
    block_12_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 75072);
    block_12_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 75072);
    block_12_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 25920);
    block_12_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25920);
    block_12_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 75072);
    block_12_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 75072);
    block_12_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_12_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_12_project_output_array.data = AI_PTR(g_network_activations_map[0] + 768);
    block_12_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 768);
    block_12_add_output_array.data = AI_PTR(g_network_activations_map[0] + 8960);
    block_12_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8960);
    block_13_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_13_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_13_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 17152);
    block_13_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 17152);
    block_13_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 66304);
    block_13_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 66304);
    block_13_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_13_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_13_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 12288);
    block_13_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12288);
    block_13_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_13_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_13_project_output_array.data = AI_PTR(g_network_activations_map[0] + 768);
    block_13_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 768);
    block_14_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_14_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_14_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 4352);
    block_14_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4352);
    block_14_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 25856);
    block_14_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25856);
    block_14_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 4352);
    block_14_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4352);
    block_14_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 25856);
    block_14_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25856);
    block_14_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 4352);
    block_14_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 4352);
    block_14_project_output_array.data = AI_PTR(g_network_activations_map[0] + 5696);
    block_14_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5696);
    block_14_add_output_array.data = AI_PTR(g_network_activations_map[0] + 9280);
    block_14_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9280);
    block_15_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_15_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_15_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 12864);
    block_15_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12864);
    block_15_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 34368);
    block_15_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34368);
    block_15_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 12864);
    block_15_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12864);
    block_15_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 34368);
    block_15_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 34368);
    block_15_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_15_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_15_project_output_array.data = AI_PTR(g_network_activations_map[0] + 1344);
    block_15_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1344);
    block_15_add_output_array.data = AI_PTR(g_network_activations_map[0] + 4928);
    block_15_add_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4928);
    block_16_expand_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_16_expand_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_16_expand_output_array.data = AI_PTR(g_network_activations_map[0] + 8512);
    block_16_expand_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8512);
    block_16_expand_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 30016);
    block_16_expand_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 30016);
    block_16_depthwise_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_16_depthwise_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_16_depthwise_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 21504);
    block_16_depthwise_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 21504);
    block_16_project_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    block_16_project_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    block_16_project_output_array.data = AI_PTR(g_network_activations_map[0] + 1344);
    block_16_project_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1344);
    Conv_1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    Conv_1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    Conv_1_output_array.data = AI_PTR(g_network_activations_map[0] + 8512);
    Conv_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8512);
    out_relu_output_array.data = AI_PTR(g_network_activations_map[0] + 90432);
    out_relu_output_array.data_start = AI_PTR(g_network_activations_map[0] + 90432);
    global_average_pooling2d_pool_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    global_average_pooling2d_pool_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    predictions_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 5120);
    predictions_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5120);
    predictions_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    predictions_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    Conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv1_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    Conv1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    Conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv1_bias_array.data = AI_PTR(g_network_weights_map[0] + 1728);
    Conv1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1728);
    expanded_conv_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    expanded_conv_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 1792);
    expanded_conv_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1792);
    expanded_conv_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    expanded_conv_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 2368);
    expanded_conv_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2368);
    expanded_conv_project_weights_array.format |= AI_FMT_FLAG_CONST;
    expanded_conv_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 2432);
    expanded_conv_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2432);
    expanded_conv_project_bias_array.format |= AI_FMT_FLAG_CONST;
    expanded_conv_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 2944);
    expanded_conv_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2944);
    block_1_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_1_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 2976);
    block_1_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2976);
    block_1_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_1_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 4512);
    block_1_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4512);
    block_1_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_1_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 4704);
    block_1_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4704);
    block_1_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_1_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 6432);
    block_1_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6432);
    block_1_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_1_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 6624);
    block_1_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 6624);
    block_1_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_1_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 8160);
    block_1_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 8160);
    block_2_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_2_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 8192);
    block_2_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 8192);
    block_2_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_2_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 9728);
    block_2_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 9728);
    block_2_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_2_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 9920);
    block_2_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 9920);
    block_2_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_2_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 11648);
    block_2_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11648);
    block_2_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_2_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 11840);
    block_2_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11840);
    block_2_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_2_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 13376);
    block_2_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 13376);
    block_3_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_3_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 13408);
    block_3_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 13408);
    block_3_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_3_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 14944);
    block_3_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 14944);
    block_3_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_3_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 15136);
    block_3_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 15136);
    block_3_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_3_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 16864);
    block_3_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 16864);
    block_3_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_3_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 17056);
    block_3_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 17056);
    block_3_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_3_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 20128);
    block_3_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 20128);
    block_4_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_4_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 20192);
    block_4_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 20192);
    block_4_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_4_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 26336);
    block_4_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 26336);
    block_4_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_4_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 26720);
    block_4_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 26720);
    block_4_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_4_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 30176);
    block_4_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 30176);
    block_4_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_4_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 30560);
    block_4_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 30560);
    block_4_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_4_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 36704);
    block_4_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 36704);
    block_5_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_5_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 36768);
    block_5_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 36768);
    block_5_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_5_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 42912);
    block_5_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 42912);
    block_5_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_5_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 43296);
    block_5_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 43296);
    block_5_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_5_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 46752);
    block_5_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 46752);
    block_5_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_5_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 47136);
    block_5_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 47136);
    block_5_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_5_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 53280);
    block_5_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 53280);
    block_6_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_6_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 53344);
    block_6_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 53344);
    block_6_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_6_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 59488);
    block_6_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 59488);
    block_6_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_6_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 59872);
    block_6_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 59872);
    block_6_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_6_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 63328);
    block_6_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 63328);
    block_6_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_6_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 63712);
    block_6_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 63712);
    block_6_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_6_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 72928);
    block_6_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 72928);
    block_7_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_7_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 73024);
    block_7_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 73024);
    block_7_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_7_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 86848);
    block_7_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 86848);
    block_7_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_7_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 87424);
    block_7_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 87424);
    block_7_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_7_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 92608);
    block_7_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 92608);
    block_7_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_7_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 93184);
    block_7_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 93184);
    block_7_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_7_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 107008);
    block_7_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 107008);
    block_8_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_8_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 107104);
    block_8_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 107104);
    block_8_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_8_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 120928);
    block_8_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 120928);
    block_8_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_8_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 121504);
    block_8_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 121504);
    block_8_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_8_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 126688);
    block_8_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 126688);
    block_8_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_8_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 127264);
    block_8_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 127264);
    block_8_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_8_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 141088);
    block_8_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 141088);
    block_9_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_9_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 141184);
    block_9_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 141184);
    block_9_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_9_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 155008);
    block_9_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 155008);
    block_9_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_9_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 155584);
    block_9_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 155584);
    block_9_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_9_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 160768);
    block_9_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 160768);
    block_9_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_9_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 161344);
    block_9_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 161344);
    block_9_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_9_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 175168);
    block_9_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 175168);
    block_10_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_10_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 175264);
    block_10_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 175264);
    block_10_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_10_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 189088);
    block_10_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 189088);
    block_10_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_10_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 189664);
    block_10_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 189664);
    block_10_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_10_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 194848);
    block_10_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 194848);
    block_10_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_10_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 195424);
    block_10_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 195424);
    block_10_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_10_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 213856);
    block_10_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 213856);
    block_11_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_11_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 213984);
    block_11_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 213984);
    block_11_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_11_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 238560);
    block_11_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 238560);
    block_11_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_11_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 239328);
    block_11_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 239328);
    block_11_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_11_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 246240);
    block_11_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 246240);
    block_11_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_11_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 247008);
    block_11_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 247008);
    block_11_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_11_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 271584);
    block_11_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 271584);
    block_12_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_12_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 271712);
    block_12_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 271712);
    block_12_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_12_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 296288);
    block_12_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 296288);
    block_12_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_12_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 297056);
    block_12_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 297056);
    block_12_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_12_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 303968);
    block_12_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 303968);
    block_12_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_12_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 304736);
    block_12_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 304736);
    block_12_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_12_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 329312);
    block_12_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 329312);
    block_13_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_13_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 329440);
    block_13_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 329440);
    block_13_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_13_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 354016);
    block_13_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 354016);
    block_13_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_13_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 354784);
    block_13_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 354784);
    block_13_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_13_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 361696);
    block_13_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 361696);
    block_13_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_13_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 362464);
    block_13_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 362464);
    block_13_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_13_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 405472);
    block_13_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 405472);
    block_14_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_14_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 405696);
    block_14_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 405696);
    block_14_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_14_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 480960);
    block_14_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 480960);
    block_14_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_14_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 482304);
    block_14_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 482304);
    block_14_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_14_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 494400);
    block_14_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 494400);
    block_14_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_14_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 495744);
    block_14_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 495744);
    block_14_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_14_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 571008);
    block_14_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 571008);
    block_15_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_15_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 571232);
    block_15_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 571232);
    block_15_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_15_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 646496);
    block_15_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 646496);
    block_15_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_15_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 647840);
    block_15_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 647840);
    block_15_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_15_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 659936);
    block_15_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 659936);
    block_15_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_15_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 661280);
    block_15_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 661280);
    block_15_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_15_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 736544);
    block_15_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 736544);
    block_16_expand_weights_array.format |= AI_FMT_FLAG_CONST;
    block_16_expand_weights_array.data = AI_PTR(g_network_weights_map[0] + 736768);
    block_16_expand_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 736768);
    block_16_expand_bias_array.format |= AI_FMT_FLAG_CONST;
    block_16_expand_bias_array.data = AI_PTR(g_network_weights_map[0] + 812032);
    block_16_expand_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 812032);
    block_16_depthwise_weights_array.format |= AI_FMT_FLAG_CONST;
    block_16_depthwise_weights_array.data = AI_PTR(g_network_weights_map[0] + 813376);
    block_16_depthwise_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 813376);
    block_16_depthwise_bias_array.format |= AI_FMT_FLAG_CONST;
    block_16_depthwise_bias_array.data = AI_PTR(g_network_weights_map[0] + 825472);
    block_16_depthwise_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 825472);
    block_16_project_weights_array.format |= AI_FMT_FLAG_CONST;
    block_16_project_weights_array.data = AI_PTR(g_network_weights_map[0] + 826816);
    block_16_project_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 826816);
    block_16_project_bias_array.format |= AI_FMT_FLAG_CONST;
    block_16_project_bias_array.data = AI_PTR(g_network_weights_map[0] + 977344);
    block_16_project_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 977344);
    Conv_1_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv_1_weights_array.data = AI_PTR(g_network_weights_map[0] + 977792);
    Conv_1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 977792);
    Conv_1_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 1551232);
    Conv_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1551232);
    predictions_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    predictions_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 1556416);
    predictions_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1556352);
    predictions_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    predictions_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 2196416);
    predictions_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2196416);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 22004224,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xec27fc2f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 22004224,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xec27fc2f,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

