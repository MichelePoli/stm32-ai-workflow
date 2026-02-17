/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-02-17T11:10:18+0000
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
#define AI_NETWORK_MODEL_SIGNATURE     "0x9810b854c903cb8be4ebf8e0e3c451e4"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-02-17T11:10:18+0000"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_10_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 150528, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  nl_0_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  nl_1_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 200704, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  nl_2_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 200704, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  nl_3_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  nl_4_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  nl_5_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  nl_6_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100352, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  nl_7_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  nl_8_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  nl_9_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  nl_10_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50176, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  nl_11_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  nl_13_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  nl_14_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  nl_15_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  nl_16_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  nl_17_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  nl_18_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  nl_19_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  nl_20_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  nl_21_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  nl_22_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25088, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6272, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  nl_23_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6272, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  nl_24_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  nl_25_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1000, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 132, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  nl_34_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 132, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 216, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32768, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2304, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 65536, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256000, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1000, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_weights_array, AI_ARRAY_FORMAT_LUT4_FLOAT,
  NULL, NULL, 132000, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  gemm_33_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 132, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 27, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12544, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 112, 112), AI_STRIDE_INIT(4, 4, 4, 32, 3584),
  1, &conv2d_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 3), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &conv2d_0_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 3, 3, 3, 8), AI_STRIDE_INIT(4, 4, 12, 96, 288),
  1, &conv2d_0_weights_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_10_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 28, 28), AI_STRIDE_INIT(4, 4, 4, 256, 7168),
  1, &conv2d_10_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_10_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_weights, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &conv2d_10_weights_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_11_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 14, 14), AI_STRIDE_INIT(4, 4, 4, 256, 3584),
  1, &conv2d_11_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_11_weights_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_12_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_12_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_12_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_weights, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 128), AI_STRIDE_INIT(4, 4, 256, 32768, 32768),
  1, &conv2d_12_weights_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_13_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_13_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_13_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_14_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_14_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_14_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &conv2d_14_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_15_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_15_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_weights, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_15_weights_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_bias, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_16_bias_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_16_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_16_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_weights, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &conv2d_16_weights_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_17_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_17_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_weights, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_17_weights_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_18_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_18_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_scratch0, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_18_scratch0_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_weights, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &conv2d_18_weights_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_19_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_19_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_weights, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_19_weights_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 112, 112), AI_STRIDE_INIT(4, 4, 4, 32, 3584),
  1, &conv2d_1_output_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &conv2d_1_weights_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_bias, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_20_bias_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_20_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_scratch0, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_20_scratch0_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_weights, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &conv2d_20_weights_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_21_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_21_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_weights, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_21_weights_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_bias, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_22_bias_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &conv2d_22_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_scratch0, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_22_scratch0_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_weights, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &conv2d_22_weights_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_bias, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_23_bias_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 7, 7), AI_STRIDE_INIT(4, 4, 4, 512, 3584),
  1, &conv2d_23_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_weights, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_23_weights_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_24_bias_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 7, 7), AI_STRIDE_INIT(4, 4, 4, 1024, 7168),
  1, &conv2d_24_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_24_scratch0_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_weights, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 256), AI_STRIDE_INIT(4, 4, 512, 131072, 131072),
  1, &conv2d_24_weights_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_bias, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_25_bias_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_output, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 7, 7), AI_STRIDE_INIT(4, 4, 4, 1024, 7168),
  1, &conv2d_25_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_weights, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 256), AI_STRIDE_INIT(4, 1, 256, 256, 256),
  1, &conv2d_25_weights_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_bias, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_26_bias_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_26_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_scratch0, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_26_scratch0_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_scratch1, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 7, 7), AI_STRIDE_INIT(4, 4, 4, 1024, 7168),
  1, &conv2d_26_scratch1_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_weights, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 4, 1024, 262144, 262144),
  1, &conv2d_26_weights_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 1000, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4000, 4000),
  1, &conv2d_28_bias_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_output, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 1000, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4000, 4000),
  1, &conv2d_28_output_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_scratch0, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_28_scratch0_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_weights, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 256, 1, 1, 1000), AI_STRIDE_INIT(4, 4, 1024, 1024000, 1024000),
  1, &conv2d_28_weights_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 112, 112), AI_STRIDE_INIT(4, 4, 4, 64, 7168),
  1, &conv2d_2_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &conv2d_2_weights_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 56, 56), AI_STRIDE_INIT(4, 4, 4, 64, 3584),
  1, &conv2d_3_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1, 16, 16, 16),
  1, &conv2d_3_weights_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 56, 56), AI_STRIDE_INIT(4, 4, 4, 128, 7168),
  1, &conv2d_4_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &conv2d_4_weights_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_5_bias_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 56, 56), AI_STRIDE_INIT(4, 4, 4, 128, 7168),
  1, &conv2d_5_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &conv2d_5_weights_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_bias, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_6_bias_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 56, 56), AI_STRIDE_INIT(4, 4, 4, 128, 7168),
  1, &conv2d_6_output_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_scratch0, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_6_scratch0_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_weights, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 4096),
  1, &conv2d_6_weights_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_7_bias_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 28, 28), AI_STRIDE_INIT(4, 4, 4, 128, 3584),
  1, &conv2d_7_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &conv2d_7_weights_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_8_bias_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 28, 28), AI_STRIDE_INIT(4, 4, 4, 256, 7168),
  1, &conv2d_8_output_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_8_scratch0_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 8192),
  1, &conv2d_8_weights_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_bias, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_9_bias_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 28, 28), AI_STRIDE_INIT(4, 4, 4, 256, 7168),
  1, &conv2d_9_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_weights, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_9_weights_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_bias, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 132, 1, 1), AI_STRIDE_INIT(4, 4, 4, 528, 528),
  1, &gemm_33_bias_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_output, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 132, 1, 1), AI_STRIDE_INIT(4, 4, 4, 528, 528),
  1, &gemm_33_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  gemm_33_weights, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1000, 132, 1, 1), AI_STRIDE_INIT(4, 1, 500, 66000, 66000),
  1, &gemm_33_weights_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  nl_0_nl_output, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 112, 112), AI_STRIDE_INIT(4, 4, 4, 32, 3584),
  1, &nl_0_nl_output_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  nl_10_nl_output, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 28, 28), AI_STRIDE_INIT(4, 4, 4, 256, 7168),
  1, &nl_10_nl_output_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  nl_11_nl_output, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 14, 14), AI_STRIDE_INIT(4, 4, 4, 256, 3584),
  1, &nl_11_nl_output_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_nl_output, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_12_nl_output_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  nl_13_nl_output, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_13_nl_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  nl_14_nl_output, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_14_nl_output_array, NULL)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  nl_15_nl_output, AI_STATIC,
  109, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_15_nl_output_array, NULL)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  nl_16_nl_output, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_16_nl_output_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  nl_17_nl_output, AI_STATIC,
  111, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_17_nl_output_array, NULL)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  nl_18_nl_output, AI_STATIC,
  112, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_18_nl_output_array, NULL)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  nl_19_nl_output, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_19_nl_output_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  nl_1_nl_output, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 112, 112), AI_STRIDE_INIT(4, 4, 4, 32, 3584),
  1, &nl_1_nl_output_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  nl_20_nl_output, AI_STATIC,
  115, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_20_nl_output_array, NULL)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  nl_21_nl_output, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_21_nl_output_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  nl_22_nl_output, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 14, 14), AI_STRIDE_INIT(4, 4, 4, 512, 7168),
  1, &nl_22_nl_output_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  nl_23_nl_output, AI_STATIC,
  118, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 7, 7), AI_STRIDE_INIT(4, 4, 4, 512, 3584),
  1, &nl_23_nl_output_array, NULL)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  nl_24_nl_output, AI_STATIC,
  119, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 7, 7), AI_STRIDE_INIT(4, 4, 4, 1024, 7168),
  1, &nl_24_nl_output_array, NULL)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  nl_25_nl_output, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 7, 7), AI_STRIDE_INIT(4, 4, 4, 1024, 7168),
  1, &nl_25_nl_output_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  nl_2_nl_output, AI_STATIC,
  121, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 112, 112), AI_STRIDE_INIT(4, 4, 4, 64, 7168),
  1, &nl_2_nl_output_array, NULL)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  nl_34_output, AI_STATIC,
  122, 0x0,
  AI_SHAPE_INIT(4, 1, 132, 1, 1), AI_STRIDE_INIT(4, 4, 4, 528, 528),
  1, &nl_34_output_array, NULL)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  nl_3_nl_output, AI_STATIC,
  123, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 56, 56), AI_STRIDE_INIT(4, 4, 4, 64, 3584),
  1, &nl_3_nl_output_array, NULL)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  nl_4_nl_output, AI_STATIC,
  124, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 56, 56), AI_STRIDE_INIT(4, 4, 4, 128, 7168),
  1, &nl_4_nl_output_array, NULL)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  nl_5_nl_output, AI_STATIC,
  125, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 56, 56), AI_STRIDE_INIT(4, 4, 4, 128, 7168),
  1, &nl_5_nl_output_array, NULL)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  nl_6_nl_output, AI_STATIC,
  126, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 56, 56), AI_STRIDE_INIT(4, 4, 4, 128, 7168),
  1, &nl_6_nl_output_array, NULL)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  nl_7_nl_output, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 28, 28), AI_STRIDE_INIT(4, 4, 4, 128, 3584),
  1, &nl_7_nl_output_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  nl_8_nl_output, AI_STATIC,
  128, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 28, 28), AI_STRIDE_INIT(4, 4, 4, 256, 7168),
  1, &nl_8_nl_output_array, NULL)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  nl_9_nl_output, AI_STATIC,
  129, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 28, 28), AI_STRIDE_INIT(4, 4, 4, 256, 7168),
  1, &nl_9_nl_output_array, NULL)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_10_output, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 224, 224), AI_STRIDE_INIT(4, 4, 4, 12, 2688),
  1, &serving_default_input_10_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_34_layer, 34,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &nl_34_chain,
  NULL, &nl_34_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_33_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_33_weights, &gemm_33_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gemm_33_layer, 33,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gemm_33_chain,
  NULL, &nl_34_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_28_weights, &conv2d_28_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_28_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_28_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_28_chain,
  NULL, &gemm_33_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float conv2d_26_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    conv2d_26_nl_params, AI_ARRAY_FORMAT_FLOAT,
    conv2d_26_nl_params_data, conv2d_26_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_26_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_26_weights, &conv2d_26_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_26_scratch0, &conv2d_26_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_26_layer, 27,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &conv2d_26_chain,
  NULL, &conv2d_28_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = &conv2d_26_nl_params, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_clip_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(7, 7), 
  .pool_stride = AI_SHAPE_2D_INIT(7, 7), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_ap_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_25_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_25_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_25_nl_nl_params_data, nl_25_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_25_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_25_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_25_nl_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_25_nl_chain,
  NULL, &conv2d_26_layer, AI_STATIC, 
  .nl_params = &nl_25_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_25_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_25_weights, &conv2d_25_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_25_layer, 25,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_25_chain,
  NULL, &nl_25_nl_layer, AI_STATIC, 
  .groups = 256, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_24_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_24_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_24_nl_nl_params_data, nl_24_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_24_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_24_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_24_nl_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_24_nl_chain,
  NULL, &conv2d_25_layer, AI_STATIC, 
  .nl_params = &nl_24_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_23_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_24_weights, &conv2d_24_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_24_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_24_layer, 24,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_24_chain,
  NULL, &nl_24_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_23_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_23_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_23_nl_nl_params_data, nl_23_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_23_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_23_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_23_nl_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_23_nl_chain,
  NULL, &conv2d_24_layer, AI_STATIC, 
  .nl_params = &nl_23_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_23_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_23_weights, &conv2d_23_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_23_layer, 23,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_23_chain,
  NULL, &nl_23_nl_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_22_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_22_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_22_nl_nl_params_data, nl_22_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_22_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_22_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_22_nl_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_22_nl_chain,
  NULL, &conv2d_23_layer, AI_STATIC, 
  .nl_params = &nl_22_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_21_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_22_weights, &conv2d_22_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_22_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_22_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_22_chain,
  NULL, &nl_22_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_21_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_21_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_21_nl_nl_params_data, nl_21_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_21_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_21_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_21_nl_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_21_nl_chain,
  NULL, &conv2d_22_layer, AI_STATIC, 
  .nl_params = &nl_21_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_20_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_21_weights, &conv2d_21_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_21_layer, 21,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_21_chain,
  NULL, &nl_21_nl_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_20_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_20_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_20_nl_nl_params_data, nl_20_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_20_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_20_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_20_nl_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_20_nl_chain,
  NULL, &conv2d_21_layer, AI_STATIC, 
  .nl_params = &nl_20_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_19_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_20_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_20_weights, &conv2d_20_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_20_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_20_layer, 20,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_20_chain,
  NULL, &nl_20_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_19_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_19_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_19_nl_nl_params_data, nl_19_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_19_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_19_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_19_nl_layer, 19,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_19_nl_chain,
  NULL, &conv2d_20_layer, AI_STATIC, 
  .nl_params = &nl_19_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_19_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_19_weights, &conv2d_19_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_19_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_19_chain,
  NULL, &nl_19_nl_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_18_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_18_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_18_nl_nl_params_data, nl_18_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_18_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_18_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_18_nl_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_18_nl_chain,
  NULL, &conv2d_19_layer, AI_STATIC, 
  .nl_params = &nl_18_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_18_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_18_weights, &conv2d_18_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_18_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_18_layer, 18,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_18_chain,
  NULL, &nl_18_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_17_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_17_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_17_nl_nl_params_data, nl_17_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_17_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_17_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_17_nl_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_17_nl_chain,
  NULL, &conv2d_18_layer, AI_STATIC, 
  .nl_params = &nl_17_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_16_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_17_weights, &conv2d_17_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_17_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_17_chain,
  NULL, &nl_17_nl_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_16_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_16_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_16_nl_nl_params_data, nl_16_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_16_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_16_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_16_nl_layer, 16,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_16_nl_chain,
  NULL, &conv2d_17_layer, AI_STATIC, 
  .nl_params = &nl_16_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_16_weights, &conv2d_16_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_16_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_16_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_16_chain,
  NULL, &nl_16_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_15_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_15_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_15_nl_nl_params_data, nl_15_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_15_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_15_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_15_nl_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_15_nl_chain,
  NULL, &conv2d_16_layer, AI_STATIC, 
  .nl_params = &nl_15_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_15_weights, &conv2d_15_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_15_layer, 15,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_15_chain,
  NULL, &nl_15_nl_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_14_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_14_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_14_nl_nl_params_data, nl_14_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_14_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_14_nl_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_14_nl_chain,
  NULL, &conv2d_15_layer, AI_STATIC, 
  .nl_params = &nl_14_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_13_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_14_weights, &conv2d_14_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_14_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_layer, 14,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_14_chain,
  NULL, &nl_14_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_13_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_13_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_13_nl_nl_params_data, nl_13_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_13_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_13_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_13_nl_layer, 13,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_13_nl_chain,
  NULL, &conv2d_14_layer, AI_STATIC, 
  .nl_params = &nl_13_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_13_weights, &conv2d_13_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_13_chain,
  NULL, &nl_13_nl_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_12_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_12_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_12_nl_nl_params_data, nl_12_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_12_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_12_nl_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_12_nl_chain,
  NULL, &conv2d_13_layer, AI_STATIC, 
  .nl_params = &nl_12_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_11_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_12_weights, &conv2d_12_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_12_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_12_chain,
  NULL, &nl_12_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_11_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_11_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_11_nl_nl_params_data, nl_11_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_11_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_11_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_11_nl_layer, 11,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_11_nl_chain,
  NULL, &conv2d_12_layer, AI_STATIC, 
  .nl_params = &nl_11_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_11_chain,
  NULL, &nl_11_nl_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_10_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_10_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_10_nl_nl_params_data, nl_10_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_10_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_10_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_10_nl_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_10_nl_chain,
  NULL, &conv2d_11_layer, AI_STATIC, 
  .nl_params = &nl_10_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_10_weights, &conv2d_10_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_10_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_10_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_10_chain,
  NULL, &nl_10_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_9_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_9_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_9_nl_nl_params_data, nl_9_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_9_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_9_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_9_nl_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_9_nl_chain,
  NULL, &conv2d_10_layer, AI_STATIC, 
  .nl_params = &nl_9_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_9_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_9_weights, &conv2d_9_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_9_layer, 9,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_9_chain,
  NULL, &nl_9_nl_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_8_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_8_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_8_nl_nl_params_data, nl_8_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_8_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_8_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_8_nl_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_8_nl_chain,
  NULL, &conv2d_9_layer, AI_STATIC, 
  .nl_params = &nl_8_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_7_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_8_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_8_chain,
  NULL, &nl_8_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_7_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_7_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_7_nl_nl_params_data, nl_7_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_7_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_7_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_7_nl_layer, 7,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_7_nl_chain,
  NULL, &conv2d_8_layer, AI_STATIC, 
  .nl_params = &nl_7_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_7_chain,
  NULL, &nl_7_nl_layer, AI_STATIC, 
  .groups = 32, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_6_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_6_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_6_nl_nl_params_data, nl_6_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_6_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_6_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_6_nl_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_6_nl_chain,
  NULL, &conv2d_7_layer, AI_STATIC, 
  .nl_params = &nl_6_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_6_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_6_weights, &conv2d_6_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_6_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_6_layer, 6,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_6_chain,
  NULL, &nl_6_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_5_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_5_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_5_nl_nl_params_data, nl_5_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_5_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_5_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_5_nl_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_5_nl_chain,
  NULL, &conv2d_6_layer, AI_STATIC, 
  .nl_params = &nl_5_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_5_chain,
  NULL, &nl_5_nl_layer, AI_STATIC, 
  .groups = 32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_4_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_4_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_4_nl_nl_params_data, nl_4_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_4_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_4_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_4_nl_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_4_nl_chain,
  NULL, &conv2d_5_layer, AI_STATIC, 
  .nl_params = &nl_4_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_4_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_4_chain,
  NULL, &nl_4_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_3_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_3_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_3_nl_nl_params_data, nl_3_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_3_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_3_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_3_nl_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_3_nl_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .nl_params = &nl_3_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_2_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_3_chain,
  NULL, &nl_3_nl_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_2_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_2_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_2_nl_nl_params_data, nl_2_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_2_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_2_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_2_nl_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_2_nl_chain,
  NULL, &conv2d_3_layer, AI_STATIC, 
  .nl_params = &nl_2_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_1_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_2_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_2_chain,
  NULL, &nl_2_nl_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_1_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_1_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_1_nl_nl_params_data, nl_1_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_1_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_1_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_1_nl_layer, 1,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_1_nl_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .nl_params = &nl_1_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_0_nl_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &conv2d_1_chain,
  NULL, &nl_1_nl_layer, AI_STATIC, 
  .groups = 8, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float nl_0_nl_nl_params_data[] = { 0.0, 6.0 };
AI_ARRAY_OBJ_DECLARE(
    nl_0_nl_nl_params, AI_ARRAY_FORMAT_FLOAT,
    nl_0_nl_nl_params_data, nl_0_nl_nl_params_data, 2, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_0_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_0_nl_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_0_nl_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, forward_clip,
  &nl_0_nl_chain,
  NULL, &conv2d_1_layer, AI_STATIC, 
  .nl_params = &nl_0_nl_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 0,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &conv2d_0_chain,
  NULL, &nl_0_nl_layer, AI_STATIC, 
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
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1935936, 1, 1),
    1935936, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 817248, 1, 1),
    817248, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_34_output),
  &conv2d_0_layer, 0xe676ee29, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1935936, 1, 1),
      1935936, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 817248, 1, 1),
      817248, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_10_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_34_output),
  &conv2d_0_layer, 0xe676ee29, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input_10_output_array.data = AI_PTR(g_network_activations_map[0] + 18912);
    serving_default_input_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18912);
    conv2d_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 621024);
    conv2d_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 621024);
    conv2d_0_output_array.data = AI_PTR(g_network_activations_map[0] + 14400);
    conv2d_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 14400);
    nl_0_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 415808);
    nl_0_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 415808);
    conv2d_1_output_array.data = AI_PTR(g_network_activations_map[0] + 14400);
    conv2d_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 14400);
    nl_1_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 415808);
    nl_1_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 415808);
    conv2d_2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 817216);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 817216);
    conv2d_2_output_array.data = AI_PTR(g_network_activations_map[0] + 3648);
    conv2d_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3648);
    nl_2_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 3648);
    nl_2_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3648);
    conv2d_3_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_3_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 200704);
    nl_3_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200704);
    conv2d_4_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_4_output_array.data = AI_PTR(g_network_activations_map[0] + 401408);
    conv2d_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 401408);
    nl_4_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_4_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_5_output_array.data = AI_PTR(g_network_activations_map[0] + 401408);
    conv2d_5_output_array.data_start = AI_PTR(g_network_activations_map[0] + 401408);
    nl_5_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_5_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_6_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 401408);
    conv2d_6_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 401408);
    conv2d_6_output_array.data = AI_PTR(g_network_activations_map[0] + 401536);
    conv2d_6_output_array.data_start = AI_PTR(g_network_activations_map[0] + 401536);
    nl_6_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_6_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_7_output_array.data = AI_PTR(g_network_activations_map[0] + 401408);
    conv2d_7_output_array.data_start = AI_PTR(g_network_activations_map[0] + 401408);
    nl_7_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_7_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_8_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_8_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_8_output_array.data = AI_PTR(g_network_activations_map[0] + 100480);
    conv2d_8_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100480);
    nl_8_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 301184);
    nl_8_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 301184);
    conv2d_9_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_9_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_9_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 200704);
    nl_9_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200704);
    conv2d_10_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_10_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_10_output_array.data = AI_PTR(g_network_activations_map[0] + 401408);
    conv2d_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 401408);
    nl_10_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_10_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_11_output_array.data = AI_PTR(g_network_activations_map[0] + 200704);
    conv2d_11_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200704);
    nl_11_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_11_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_12_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 50176);
    conv2d_12_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 50176);
    conv2d_12_output_array.data = AI_PTR(g_network_activations_map[0] + 50432);
    conv2d_12_output_array.data_start = AI_PTR(g_network_activations_map[0] + 50432);
    nl_12_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 150784);
    nl_12_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 150784);
    conv2d_13_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_13_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_13_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    nl_13_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_14_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_14_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_14_output_array.data = AI_PTR(g_network_activations_map[0] + 200704);
    conv2d_14_output_array.data_start = AI_PTR(g_network_activations_map[0] + 200704);
    nl_14_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_14_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_15_output_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    nl_15_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_15_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_16_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_16_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_16_output_array.data = AI_PTR(g_network_activations_map[0] + 100864);
    conv2d_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100864);
    nl_16_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_16_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_17_output_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_17_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    nl_17_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_17_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_18_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_18_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_18_output_array.data = AI_PTR(g_network_activations_map[0] + 100864);
    conv2d_18_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100864);
    nl_18_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_18_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_19_output_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_19_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    nl_19_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_19_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_20_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_20_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_20_output_array.data = AI_PTR(g_network_activations_map[0] + 100864);
    conv2d_20_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100864);
    nl_20_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_20_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_21_output_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_21_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    nl_21_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_21_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_22_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_22_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_22_output_array.data = AI_PTR(g_network_activations_map[0] + 100864);
    conv2d_22_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100864);
    nl_22_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_22_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_23_output_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_23_output_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    nl_23_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_23_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_24_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 25088);
    conv2d_24_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 25088);
    conv2d_24_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    conv2d_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    nl_24_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 75776);
    nl_24_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 75776);
    conv2d_25_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_25_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_25_nl_output_array.data = AI_PTR(g_network_activations_map[0] + 50176);
    nl_25_nl_output_array.data_start = AI_PTR(g_network_activations_map[0] + 50176);
    conv2d_26_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_26_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_26_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_26_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 100352);
    conv2d_26_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    conv2d_26_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    conv2d_28_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_28_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_28_output_array.data = AI_PTR(g_network_activations_map[0] + 2048);
    conv2d_28_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2048);
    gemm_33_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_33_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_34_output_array.data = AI_PTR(g_network_activations_map[0] + 528);
    nl_34_output_array.data_start = AI_PTR(g_network_activations_map[0] + 528);
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
    
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 864);
    conv2d_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 864);
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_network_weights_map[0] + 896);
    conv2d_1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 896);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 1184);
    conv2d_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1184);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_network_weights_map[0] + 1216);
    conv2d_2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1216);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_network_weights_map[0] + 1728);
    conv2d_2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1728);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_network_weights_map[0] + 1792);
    conv2d_3_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1792);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_network_weights_map[0] + 2368);
    conv2d_3_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2368);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_network_weights_map[0] + 2432);
    conv2d_4_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2432);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_network_weights_map[0] + 4480);
    conv2d_4_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4480);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(g_network_weights_map[0] + 4608);
    conv2d_5_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4608);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(g_network_weights_map[0] + 5760);
    conv2d_5_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5760);
    conv2d_6_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_weights_array.data = AI_PTR(g_network_weights_map[0] + 5888);
    conv2d_6_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5888);
    conv2d_6_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_bias_array.data = AI_PTR(g_network_weights_map[0] + 9984);
    conv2d_6_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 9984);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(g_network_weights_map[0] + 10112);
    conv2d_7_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10112);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(g_network_weights_map[0] + 11264);
    conv2d_7_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11264);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(g_network_weights_map[0] + 11392);
    conv2d_8_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11392);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(g_network_weights_map[0] + 19584);
    conv2d_8_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 19584);
    conv2d_9_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_weights_array.data = AI_PTR(g_network_weights_map[0] + 19840);
    conv2d_9_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 19840);
    conv2d_9_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_bias_array.data = AI_PTR(g_network_weights_map[0] + 22144);
    conv2d_9_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 22144);
    conv2d_10_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_weights_array.data = AI_PTR(g_network_weights_map[0] + 22400);
    conv2d_10_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 22400);
    conv2d_10_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_bias_array.data = AI_PTR(g_network_weights_map[0] + 38784);
    conv2d_10_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 38784);
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(g_network_weights_map[0] + 39040);
    conv2d_11_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 39040);
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(g_network_weights_map[0] + 41344);
    conv2d_11_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 41344);
    conv2d_12_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_weights_array.data = AI_PTR(g_network_weights_map[0] + 41600);
    conv2d_12_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 41600);
    conv2d_12_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_bias_array.data = AI_PTR(g_network_weights_map[0] + 74368);
    conv2d_12_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 74368);
    conv2d_13_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_weights_array.data = AI_PTR(g_network_weights_map[0] + 74880);
    conv2d_13_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 74880);
    conv2d_13_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_bias_array.data = AI_PTR(g_network_weights_map[0] + 79488);
    conv2d_13_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 79488);
    conv2d_14_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_weights_array.data = AI_PTR(g_network_weights_map[0] + 80000);
    conv2d_14_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 80000);
    conv2d_14_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_bias_array.data = AI_PTR(g_network_weights_map[0] + 145536);
    conv2d_14_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 145536);
    conv2d_15_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_weights_array.data = AI_PTR(g_network_weights_map[0] + 146048);
    conv2d_15_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 146048);
    conv2d_15_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_bias_array.data = AI_PTR(g_network_weights_map[0] + 150656);
    conv2d_15_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 150656);
    conv2d_16_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_weights_array.data = AI_PTR(g_network_weights_map[0] + 151168);
    conv2d_16_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 151168);
    conv2d_16_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_bias_array.data = AI_PTR(g_network_weights_map[0] + 216704);
    conv2d_16_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 216704);
    conv2d_17_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_weights_array.data = AI_PTR(g_network_weights_map[0] + 217216);
    conv2d_17_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 217216);
    conv2d_17_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_bias_array.data = AI_PTR(g_network_weights_map[0] + 221824);
    conv2d_17_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 221824);
    conv2d_18_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_18_weights_array.data = AI_PTR(g_network_weights_map[0] + 222336);
    conv2d_18_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 222336);
    conv2d_18_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_18_bias_array.data = AI_PTR(g_network_weights_map[0] + 287872);
    conv2d_18_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 287872);
    conv2d_19_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_weights_array.data = AI_PTR(g_network_weights_map[0] + 288384);
    conv2d_19_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 288384);
    conv2d_19_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_bias_array.data = AI_PTR(g_network_weights_map[0] + 292992);
    conv2d_19_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 292992);
    conv2d_20_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_weights_array.data = AI_PTR(g_network_weights_map[0] + 293504);
    conv2d_20_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 293504);
    conv2d_20_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_bias_array.data = AI_PTR(g_network_weights_map[0] + 359040);
    conv2d_20_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 359040);
    conv2d_21_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_weights_array.data = AI_PTR(g_network_weights_map[0] + 359552);
    conv2d_21_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 359552);
    conv2d_21_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_bias_array.data = AI_PTR(g_network_weights_map[0] + 364160);
    conv2d_21_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 364160);
    conv2d_22_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_weights_array.data = AI_PTR(g_network_weights_map[0] + 364672);
    conv2d_22_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 364672);
    conv2d_22_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_bias_array.data = AI_PTR(g_network_weights_map[0] + 430208);
    conv2d_22_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 430208);
    conv2d_23_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_23_weights_array.data = AI_PTR(g_network_weights_map[0] + 430720);
    conv2d_23_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 430720);
    conv2d_23_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_23_bias_array.data = AI_PTR(g_network_weights_map[0] + 435328);
    conv2d_23_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 435328);
    conv2d_24_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_weights_array.data = AI_PTR(g_network_weights_map[0] + 435840);
    conv2d_24_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 435840);
    conv2d_24_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_bias_array.data = AI_PTR(g_network_weights_map[0] + 566912);
    conv2d_24_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 566912);
    conv2d_25_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_25_weights_array.data = AI_PTR(g_network_weights_map[0] + 567936);
    conv2d_25_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 567936);
    conv2d_25_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_25_bias_array.data = AI_PTR(g_network_weights_map[0] + 577152);
    conv2d_25_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 577152);
    conv2d_26_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_weights_array.data = AI_PTR(g_network_weights_map[0] + 578176);
    conv2d_26_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 578176);
    conv2d_26_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_bias_array.data = AI_PTR(g_network_weights_map[0] + 840320);
    conv2d_26_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 840320);
    conv2d_28_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_weights_array.data = AI_PTR(g_network_weights_map[0] + 841344);
    conv2d_28_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 841344);
    conv2d_28_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_bias_array.data = AI_PTR(g_network_weights_map[0] + 1865344);
    conv2d_28_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1865344);
    gemm_33_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_33_weights_array.data = AI_PTR(g_network_weights_map[0] + 1869408);
    gemm_33_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1869344);
    gemm_33_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_33_bias_array.data = AI_PTR(g_network_weights_map[0] + 1935408);
    gemm_33_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1935408);
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
      
      .n_macc            = 43702008,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xe676ee29,
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
      
      .n_macc            = 43702008,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xe676ee29,
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

