cmake_minimum_required(VERSION 3.16)
project(game_chess_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(chess_engine
  main.cpp
  board.cpp
  movegen.cpp
  search.cpp
  eval.cpp
  tt.cpp
)

target_compile_options(chess_engine PRIVATE -Wall -Wextra -pedantic)
