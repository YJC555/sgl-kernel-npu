# find python binary
find_program(PYTHON_EXECUTABLE NAMES python3)

if (NOT EXISTS ${PYTHON_EXECUTABLE})
    message(FATAL_ERROR "python3 is not found, install python firstly")
endif ()

# get torch path, torch npu path, pybind11 path via python script
# 每条信息带唯一前缀输出, 避免 torch/torch_npu 自身的 warning 干扰解析
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "import torch, torch_npu, pybind11, os
print('TORCH_DIR=' + os.path.realpath(os.path.dirname(torch.__file__)))
print('TORCH_NPU_DIR=' + os.path.realpath(os.path.dirname(torch_npu.__file__)))
print('PYBIND11_DIR=' + os.path.realpath(os.path.dirname(pybind11.__file__)))
print('TORCH_API_ENABLED=' + str(torch.compiled_with_cxx11_abi()))"
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE OUTPUT_ENV_DEFINES
)

# if failed to run the python script
if (NOT "${EXEC_RESULT}" STREQUAL "0")
    message(FATAL_ERROR "failed to get run python script to get ENVS like TORCH_DIR etc")
else ()
    message(STATUS "run python script successfully, output string is [${OUTPUT_ENV_DEFINES}]")
endif ()

# 按前缀精确提取, 允许输出中夹杂其它无关行
string(REGEX MATCH "TORCH_DIR=([^\n\r]*)" _TORCH_MATCH "${OUTPUT_ENV_DEFINES}")
set(TORCH_DIR "${CMAKE_MATCH_1}")

string(REGEX MATCH "TORCH_NPU_DIR=([^\n\r]*)" _TORCH_NPU_MATCH "${OUTPUT_ENV_DEFINES}")
set(TORCH_NPU_DIR "${CMAKE_MATCH_1}")

string(REGEX MATCH "PYBIND11_DIR=([^\n\r]*)" _PYBIND11_MATCH "${OUTPUT_ENV_DEFINES}")
set(PYBIND11_DIR "${CMAKE_MATCH_1}")

string(REGEX MATCH "TORCH_API_ENABLED=([^\n\r]*)" _TORCH_ABI_MATCH "${OUTPUT_ENV_DEFINES}")
set(TORCH_API_ENABLED "${CMAKE_MATCH_1}")

if (NOT TORCH_DIR OR NOT TORCH_NPU_DIR OR NOT PYBIND11_DIR OR NOT TORCH_API_ENABLED)
    message(FATAL_ERROR "failed to parse ENVS from python script, output is [${OUTPUT_ENV_DEFINES}]")
endif ()

message(STATUS "SOC_VERSION=${SOC_VERSION}")
message(STATUS "TORCH_DIR=${TORCH_DIR}")
message(STATUS "TORCH_NPU_DIR=${TORCH_NPU_DIR}")
message(STATUS "PYBIND11_DIR=${PYBIND11_DIR}")
message(STATUS "TORCH_API_ENABLED=${TORCH_API_ENABLED}")

# set _GLIBCXX_USE_CXX11_ABI
if ("${TORCH_API_ENABLED}" STREQUAL "True")
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=1")
else ()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=0")
endif ()
