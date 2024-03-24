cmake_minimum_required(VERSION 3.10)

# Nazwa projektu
project(perla)

# Ustawienie ścieżki do folderu include
include_directories(src/Jetson/drv/include)
include_directories(src/Jetson/pub/include)
include_directories(src/Jetson/swc/include)

# Ustawienie ścieżki do źródeł plików .cpp
file(GLOB_RECURSE SOURCES src/Jetson/drv/src/*.cpp src/Jetson/main/*.cpp src/Jetson/pub/*.cpp src/Jetson/swc/*.cpp)

# Dodanie plików nagłówkowych
file(GLOB_RECURSE HEADERS include/*.hpp)

# Dodanie pliku main.cpp
list(APPEND SOURCES src/Jetson/main/main.cpp)

# Utworzenie folderu build w którym bedzie build projektu
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build)

# Dodanie celu wykonywalnego
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Dodanie dowiązań między plikami źródłowymi a nagłówkowymi
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES} ${HEADERS})

# Dodanie celu do kompilacji bibliotek
add_subdirectory(src/Jetson/drv)
add_subdirectory(src/Jetson/pub)
add_subdirectory(src/Jetson/swc)

# Dodanie dowiązań między biblioteką a plikami źródłowymi
target_link_libraries(${PROJECT_NAME} drv_library)
target_link_libraries(${PROJECT_NAME} pub_library)
target_link_libraries(${PROJECT_NAME} swc_library)

# Uprawnienia dla pliku wykonywalnego
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(PROGRAMS enable_can.sh DESTINATION bin PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
