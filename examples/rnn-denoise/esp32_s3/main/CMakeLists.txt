set(NNOM_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../..)

file(GLOB NNOM_CORE_SRCS
    "${NNOM_ROOT}/src/core/*.c"
    "${NNOM_ROOT}/src/layers/*.c"
    "${NNOM_ROOT}/src/backends/*.c"
)

idf_component_register(
    SRCS
        "main.c"
        "${NNOM_ROOT}/examples/rnn-denoise/mfcc.c"
        ${NNOM_CORE_SRCS}
    INCLUDE_DIRS
        "."
        "${NNOM_ROOT}/examples/rnn-denoise"
        "${NNOM_ROOT}/inc"
        "${NNOM_ROOT}/port"
)
