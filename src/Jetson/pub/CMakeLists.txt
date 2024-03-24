# Ustawienie folderu źródeł
set(SOURCES
    src/DBC.cpp
    src/DriverManager.cpp
)

# Ustawienie folderu nagłówków
set(HEADERS
    include/DBC.hpp
    include/Constants.hpp
    include/DriverManager.hpp
    include/Enums.hpp
    include/CAN.hpp
    include/includes.hpp
)

# Dodanie biblioteki
add_library(pub_library ${SOURCES} ${HEADERS})

# Ustawienie folderu include dla biblioteki
target_include_directories(pub_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
