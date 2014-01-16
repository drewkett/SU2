 /*!
 * \file linear_solvers_structure.cpp
 * \brief Main classes required for solving linear systems of equations
 * \author Current Development: Stanford University.
 * \version 2.0.10
 *
 * Stanford University Unstructured (SU2).
 * Copyright (C) 2012-2013 Aerospace Design Laboratory (ADL).
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#define NO_MUTATIONPP
#define NO_MPI

#include "../include/linear_solvers_structure.hpp"

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/jacobi_precond.hpp"
//#include "viennacl/linalg/ilu.hpp"

#include <sys/time.h>

#include "../include/matrix_structure.hpp"

unsigned long CSysSolve::BCGSTAB_CUDA(const CSysVector & b, CSysVector & x, CMatrixVectorProduct & mat_vec,
                                 CPreconditioner & precond, double tol, unsigned long m, bool monitoring) {
	
  int rank = 0;
#ifndef NO_MPI
#ifdef WINDOWS
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
	rank = MPI::COMM_WORLD.Get_rank();
#endif
#endif
  
  /*--- Check the subspace size ---*/
  if (m < 1) {
    if (rank == 0) cerr << "CSysSolve::BCGSTAB: illegal value for subspace size, m = " << m << endl;
#ifdef NO_MPI
    exit(1);
#else
#ifdef WINDOWS
	MPI_Abort(MPI_COMM_WORLD,1);
    MPI_Finalize();
#else
    MPI::COMM_WORLD.Abort(1);
    MPI::Finalize();
#endif
#endif
  }
	
  struct timeval current, last;
  double diff;
  gettimeofday(&last,NULL);
  CSysMatrix * matrix = dynamic_cast<CSysMatrixVectorProduct&>(mat_vec).sparse_matrix;
  vector< map < unsigned long, double > > cpu_sparse_matrix( matrix->nPoint * matrix->nEqn);
  for (int iPoint = 0; iPoint < matrix->nPoint; iPoint++) {
    for (int index = matrix->row_ptr[iPoint]; index < matrix->row_ptr[iPoint+1]; index++) {
    	for (int iVar = 0; iVar < matrix->nEqn; iVar++) {
    		for (int jVar = 0; jVar < matrix->nVar; jVar++) {
    			cpu_sparse_matrix[iPoint*matrix->nEqn+iVar][matrix->col_ind[index]*matrix->nVar+jVar] = matrix->matrix[index*matrix->nVar*matrix->nEqn+matrix->nEqn*iVar+jVar];
    		}
    	}
    }
  }
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy A -> cpu A took " << diff << endl;
  last = current;
  viennacl::compressed_matrix <double> vcl_sparse_matrix(matrix->nPoint*matrix->nEqn,matrix->nPoint*matrix->nVar);
  viennacl::vector<double> vcl_rhs(matrix->nPoint*matrix->nEqn);
  viennacl::vector<double> vcl_result(matrix->nPoint*matrix->nVar);
  vector<double> cpu_result(matrix->nPoint*matrix->nVar);

  vector<double> cpu_rhs(b.vec_val,b.vec_val+matrix->nPoint*matrix->nVar);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy b -> cpu b took " << diff << endl;
  last = current;

  copy(cpu_sparse_matrix, vcl_sparse_matrix);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy cpu A -> gpu A took " << diff << endl;
  last = current;
  copy(cpu_rhs,vcl_rhs);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy cpu b -> gpu b took " << diff << endl;
  last = current;
  viennacl::linalg::jacobi_precond< viennacl::compressed_matrix<double> > vcl_jacobi (vcl_sparse_matrix, viennacl::linalg::jacobi_tag());
  //viennacl::linalg::block_ilu_precond< viennacl::compressed_matrix<double>, viennacl::linalg::ilu0_tag> vcl_block_ilu (vcl_sparse_matrix, viennacl::linalg::ilu0_tag(true));

  vcl_result = viennacl::linalg::solve(vcl_sparse_matrix,vcl_rhs,viennacl::linalg::bicgstab_tag(1e-6,20),vcl_jacobi);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Solve took " << diff << endl;
  last = current;
  copy(vcl_result,cpu_rhs);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy gpu x -> cpu x took " << diff << endl;
  last = current;
  copy(cpu_rhs.begin(),cpu_rhs.end(),x.vec_val);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy cpu x -> x took " << diff << endl;
  last = current;
	return 1;
}
