# Ustawienie folderu źródeł
set(SOURCES
    #src/JetsonDriver.cpp
)

# Ustawienie folderu nagłówków
set(HEADERS
    #include/JetsonDriver.hpp
)

# Dodanie biblioteki
add_library(swc_library ${SOURCES} ${HEADERS})

# Ustawienie folderu include dla biblioteki
target_include_directories(swc_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
