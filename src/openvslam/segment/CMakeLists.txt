# Add sources
target_sources(${PROJECT_NAME}
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/base_model.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/deeplab.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/mask_rcnn.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/segmenter.cc
               )

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${OPENVSLAM_INCLUDE_INSTALL_DIR}/segment)
