#
# Extra add-on TPLs
#
# For a description of the fields, see:
#
#   Trilinos/cmake/TrilinosTPLs.cmake
#

SET(DataTransferKit_TPLS_FINDMODS_CLASSIFICATIONS
  HDF5      "cmake/TPLs/"      EX
  MOAB      "cmake/TPLs/"      EX
  )

# NOTE: Above, the paths to the FindTPL<TPLNAME> modules (with an implicit
# *.cmake extension) are relative to the Trilinos/cmake/TPLs directory.
