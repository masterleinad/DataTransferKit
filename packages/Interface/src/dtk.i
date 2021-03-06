%module DataTransferKit

%apply int { MPI_Comm };
%typemap(ftype) MPI_Comm
   "integer"
%typemap(fin, noblock=1) MPI_Comm {
    $1 = int($input, C_INT)
}
%typemap(fout, noblock=1) MPI_Comm {
    $result = int($1)
}

%typemap(in, noblock=1) MPI_Comm {
    $1 = MPI_Comm_f2c(%static_cast(*$input, MPI_Fint));
}
%typemap(out, noblock=1) MPI_Comm {
    $result = %static_cast(MPI_Comm_c2f($1), int);
}

%{
#include "DTK_C_API.h"
%}

%rename DTK_initializeCmd DTK_initialize_cmd;
%rename DTK_isInitialized DTK_is_initialized;

%rename DTK_isValidUserApplication DTK_is_valid_user_application;
%rename DTK_createUserApplication DTK_create_user_application;
%rename DTK_destroyUserApplication DTK_destroy_user_application;

%rename DTK_createMap DTK_create_map;
%rename DTK_isValidMap DTK_is_valid_map;
%rename DTK_applyMap DTK_apply_map;
%rename DTK_destroyMap DTK_destroy_map;

%rename DTK_setUserFunction DTK_set_user_function;

%include <std_string.i>

%rename DTK_string_version DTK_version;
%rename DTK_string_git_commit_hash DTK_git_commit_hash;
%inline %{
  std::string DTK_string_version() {
    return std::string(DTK_version());
  }
  std::string DTK_string_git_commit_hash() {
    return std::string(DTK_gitCommitHash());
  }
%}
%ignore DTK_version;
%ignore DTK_gitCommitHash;;

%include "DTK_CellTypes.h"
%include "DTK_C_API.h"
