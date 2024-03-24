# Ustawienie folderu źródeł
set(SOURCES
    src/ClassManager.cpp
    src/main.cpp
    src/CAN.cpp
    src/DBC.cpp
)

# Ustawienie folderu nagłówków
set(HEADERS
    include/ClassManager.hpp
    include/CAN.hpp
    include/DBC.hpp
    include/Enums.hpp
    include/includes.hpp
)

# Dodanie biblioteki
add_library(makieta_library ${SOURCES} ${HEADERS})

# Ustawienie folderu include dla biblioteki
target_include_directories(makieta_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
