
import os
import re
import shutil
from dataclasses import dataclass

@dataclass
class MasterState:
    modify_main: bool = True
    main_c_path: str = ""
    network_name: str = "network"
    timestamp: str = "test"
    main_modification_success: bool = False

def mock_modify_main_c(state: MasterState):
    print(f"Modifying {state.main_c_path}...")
    with open(state.main_c_path, 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    net_name = state.network_name if state.network_name else "network"
    net_upper = net_name.upper()

    # 1. Includes
    includes_pattern = r'(\/\* USER CODE BEGIN Includes \*\/)'
    ai_includes = f'\n/* AI includes */\n#include "{net_name}.h"\n#include "{net_name}_data.h"\n'
    if re.search(includes_pattern, main_content) and f'#include "{net_name}.h"' not in main_content:
        main_content = re.sub(includes_pattern, r'\1' + ai_includes, main_content)

    # 2. Private Variables (PV)
    pv_pattern = r'(\/\* USER CODE BEGIN PV \*\/)'
    ai_pv = f'''
/* AI Variables */
static ai_handle {net_name} = AI_HANDLE_NULL;
static ai_u8 activations[AI_{net_upper}_DATA_ACTIVATIONS_SIZE];
static ai_float in_data[AI_{net_upper}_IN_1_SIZE];
static ai_float out_data[AI_{net_upper}_OUT_1_SIZE];

static ai_buffer ai_input[AI_{net_upper}_IN_NUM];
static ai_buffer ai_output[AI_{net_upper}_OUT_NUM];
'''
    if re.search(pv_pattern, main_content) and f'static ai_handle {net_name}' not in main_content:
         main_content = re.sub(pv_pattern, r'\1' + ai_pv, main_content)

    # 3. Initialization (BEGIN 2)
    init_pattern = r'(\/\* USER CODE BEGIN 2 \*\/)'
    ai_init = f'''
  /* AI Initialization */
  ai_error err;
  ai_network_params params;
  
  /* Get the weights/params from the data module */
  if (!ai_{net_name}_data_params_get(&params)) {{
      /* TODO: Handle error */
  }}
  
  /* Set the activations buffer */
  params.activations = AI_HANDLE_PTR(activations);
  
  /* Create the network */
  err = ai_{net_name}_create(&{net_name}, AI_{net_upper}_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {{
      /* TODO: Handle error */
  }}
  
  /* Initialize the network */
  if (!ai_{net_name}_init({net_name}, &params)) {{
      /* TODO: Handle error */
  }}
  
  /* Initialize input/output buffers */
  ai_input[0] = ai_{net_name}_inputs_get({net_name}, NULL)[0];
  ai_output[0] = ai_{net_name}_outputs_get({net_name}, NULL)[0];
  
  ai_input[0].data = AI_HANDLE_PTR(in_data);
  ai_output[0].data = AI_HANDLE_PTR(out_data);
'''
    if re.search(init_pattern, main_content) and f'ai_{net_name}_create' not in main_content:
        main_content = re.sub(init_pattern, r'\1' + ai_init, main_content)

    # 4. Inference Loop (WHILE)
    while_pattern = r'(\/\* USER CODE BEGIN WHILE \*\/)'
    ai_while = f'''
    /* AI Inference */
    /* TODO: Fill in_data with sensor data */
    
    if (ai_{net_name}_run({net_name}, &ai_input[0], &ai_output[0]) != 1) {{
        /* TODO: Handle error */
    }}
    
    /* TODO: Use out_data results */
'''
    if re.search(while_pattern, main_content) and f'ai_{net_name}_run' not in main_content:
        main_content = re.sub(while_pattern, r'\1' + ai_while, main_content)

    with open(state.main_c_path + ".test", 'w', encoding='utf-8') as f:
        f.write(main_content)
    print("Test file created at " + state.main_c_path + ".test")

if __name__ == "__main__":
    state = MasterState(main_c_path="/mnt/shared-storage/mrusso/STM32CubeMX/UseCase_20260120_165402/UseCase/Src/main.c")
    mock_modify_main_c(state)
