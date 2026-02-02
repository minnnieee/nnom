set(NNOM_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../..)

file(GLOB_RECURSE NNOM_SRCS "${NNOM_ROOT}/src/*.c")

set(DENOISE_SRCS
    "main.c"
    "audio_i2s.c"
    "../../mfcc.c"
)

idf_component_register(
    SRCS ${DENOISE_SRCS} ${NNOM_SRCS}
    INCLUDE_DIRS "." "../../" "${NNOM_ROOT}/inc" "${NNOM_ROOT}/src"
)
