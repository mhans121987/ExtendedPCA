# find gtest ...
if(NOT GTEST_FOUND)
 
IF(IS_DIRECTORY "/usr/src/gtest/")
  MESSAGE(STATUS "Found google test sources in /usr/src/gtest/") 
  
  ADD_SUBDIRECTORY(/usr/src/gtest/ gtest)
  
  # mimick the behaviour of find_package(GTest)
  SET(GTEST_FOUND TRUE)
  SET(GTEST_BOTH_LIBRARIES gtest gtest_main)
  SET(GTEST_LIBRARIES gtest)
  SET(GTEST_MAIN_LIBRARIES gtest_main)
ELSE()
  find_package(GTest)
ENDIF()

endif()

if(GTEST_FOUND)
include_directories(${GTEST_INCLUDE_DIRS} ../src)
add_executable(pcaTest testExtendedPCA.cpp)

target_link_libraries(pcaTest ${GTEST_MAIN_LIBRARIES} pcastar)
add_test(pcaTest pcaTest)

ELSE()
  MESSAGE(STATUS "GTEST not found. Not building Test") 
endif()
