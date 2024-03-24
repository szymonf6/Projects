# Ustawienie folderu źródeł
set(SOURCES
    src/JetsonDriver.cpp
)

# Ustawienie folderu nagłówków
set(HEADERS
    include/JetsonDriver.hpp
    include/DisplayDriver.hpp
    include/CameraDriver.hpp
)

# Dodanie biblioteki
add_library(drv_library ${SOURCES} ${HEADERS})

# Ustawienie folderu include dla biblioteki
target_include_directories(drv_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
