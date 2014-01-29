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

//#include "viennacl/scalar.hpp"
//#include "viennacl/vector.hpp"
//#include "viennacl/compressed_matrix.hpp"
//#include "viennacl/linalg/bicgstab.hpp"
//#include "viennacl/linalg/jacobi_precond.hpp"
//#include "viennacl/linalg/ilu.hpp"
#include "cusparse_v2.h"
#include "cublas_v2.h"

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
  double *d_A,*d_b,*d_x;
  int *d_row_ptr, *d_col_ind;
  int *row_ptr = new int[matrix->nPoint+1];
  int *col_ind = new int[matrix->nnz];
  for (int i=0; i < matrix->nPoint+1; i++)
	  row_ptr[i] = matrix->row_ptr[i];

  for (int i=0; i < matrix->nnz; i++)
	  col_ind[i] = matrix->col_ind[i];

  cudaMalloc(&d_A,matrix->nnz*matrix->nVar*matrix->nEqn*sizeof(double));
  cudaMalloc(&d_row_ptr,(matrix->nPoint+1)*sizeof(int));
  cudaMalloc(&d_col_ind,matrix->nnz*sizeof(int));
  cudaMalloc(&d_b,matrix->nPoint*matrix->nEqn*sizeof(double));
  cudaMalloc(&d_x,matrix->nPoint*matrix->nVar*sizeof(double));
  cudaMemcpy(d_A,matrix->matrix,matrix->nnz*matrix->nVar*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr,row_ptr,(matrix->nPoint+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind,col_ind,matrix->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,b.vec_val,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_x,x.vec_val,matrix->nPoint*matrix->nVar*sizeof(double),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy -> gpu took " << diff << endl;
  last = current;
    
    
  double *d_r,*d_r0;
  cudaMalloc(&d_r,matrix->nPoint*matrix->nEqn*sizeof(double));
  cudaMalloc(&d_r0,matrix->nPoint*matrix->nEqn*sizeof(double));
  cudaMemcpy(d_r,b.vec_val,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice);
  CSysVector r(b);
  CSysVector r_0(b);
  CSysVector p(b);
	CSysVector v(b);
  CSysVector s(b);
	CSysVector t(b);
	CSysVector phat(b);
	CSysVector shat(b);
  CSysVector A_x(b);
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "CSysVector init" << diff << endl;
  last = current;
  
  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/
  cusparseMatDescr_t descr;
  cusparseHandle_t handle;
  cublasHandle_t blas_handle;
  cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
  cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseCreate(&handle);
  cublasCreate(&blas_handle);
  cusparseCreateMatDescr(&descr);
  double g_alpha = -1.;
  double g_beta = 1.;
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "cuda init  " << diff << endl;
  last = current;

  cusparseDbsrmv(handle,dir,trans,matrix->nPoint,matrix->nPoint,matrix->nnz,&g_alpha,descr,d_A,d_row_ptr,d_col_ind,matrix->nVar,d_x,&g_beta,d_r);
  cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "calc Ax gpu" << diff << endl;
  last = current;
  cublasDcopy(blas_handle,matrix->nPoint*matrix->nEqn,d_r,1.,d_r0,1.);
  cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "copy d_r to d_r0 " << diff << endl;
  last = current;
  cerr << "nPoint = " << matrix->nPoint << endl;
  cerr << "nnz = " << matrix->nnz << endl;
  cerr << "nVar = " << matrix->nVar << endl;
  cerr << "nEqn = " << matrix->nEqn << endl;
  //cudaMemcpy(r_0.vec_val,d_r,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyDeviceToHost);
  //gettimeofday(&current,NULL);
  //diff = current.tv_sec-last.tv_sec;
  //diff += 1.e-6*(current.tv_usec-last.tv_usec);
  //cerr << "r_0 copy" << diff << endl;
  //last = current;&
  //cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_DEVICE);

	//double *d_norm0, *d_norm_r;
	//cudaMalloc(&d_norm_r,sizeof(double));
	//cudaMalloc(&d_norm0,sizeof(double));
	//cublasDnrm2(blas_handle,matrix->nPoint*matrix->nEqn,d_r,1.,d_norm_r);
	//cublasDnrm2(blas_handle,matrix->nPoint*matrix->nEqn,d_b,1.,d_norm0);
	double norm0, norm_r;
	cublasDnrm2(blas_handle,matrix->nPoint*matrix->nEqn,d_r,1.,&norm_r);
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "calc/copy norm_r " << diff << endl;
  last = current;
	cublasDnrm2(blas_handle,matrix->nPoint*matrix->nEqn,d_b,1.,&norm0);
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "calc/copy norm_r " << diff << endl;
  last = current;

	double *d_rho, *d_rho_prime;
	cudaMalloc(&d_rho,sizeof(double));
	cudaMalloc(&d_rho_prime,sizeof(double));

	double alpha = 1.0, beta = 1.0, omega = 1.0;
	//*d_norm0 = *d_norm_r;
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "setup " << diff << endl;
  last = current;
    for (int i = 0; i < m; i++) {
		 //cublasDcopy(blas_handle,1,d_rho,1.,d_rho_prime,1.);
		/*--- Compute rho_i ---*/
		//cublasDdot(blas_handle,matrix->nPoint*matrix->nEqn,d_r,1.,d_r0,1.,d_rho);
		
		/*--- Compute beta ---*/
		//beta = (rho / rho_prime) * (alpha /omega);
	}
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "end loop " << diff << endl;
  last = current;

	mat_vec(x,A_x);
    r -= A_x; r_0 = r; // recall, r holds b initially
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "calc Ax cpu " << diff << endl;
  last = current;
    norm_r = r.norm();
	cerr << "norm_r = " << norm_r << endl;
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "calc norm_r cpu " << diff << endl;
//  last = current;
//  double norm0 = b.norm();
//  if ( (norm_r < tol*norm0) || (norm_r < eps) ) {
//    if (rank == 0) cout << "CSysSolve::BCGSTAB(): system solved by initial guess." << endl;
//    return 0;
//  }
//	
//	/*--- Initialization ---*/
//  double alpha = 1.0, beta = 1.0, omega = 1.0, rho = 1.0, rho_prime = 1.0;
//	
//  /*--- Set the norm to the initial initial residual value ---*/
//  norm0 = norm_r;
//  
//  /*--- Output header information including initial residual ---*/
//  int i = 0;
//  if ((monitoring) && (rank == 0)) {
//    writeHeader("BCGSTAB", tol, norm_r);
//    writeHistory(i, norm_r, norm0);
//  }
//	
//  /*---  Loop over all search directions ---*/
//  for (i = 0; i < m; i++) {
//		
//		/*--- Compute rho_prime ---*/
//		rho_prime = rho;
//		
//		/*--- Compute rho_i ---*/
//		rho = dotProd(r, r_0);
//		
//		/*--- Compute beta ---*/
//		beta = (rho / rho_prime) * (alpha /omega);
//		
//		/*--- p_{i} = r_{i-1} + beta * p_{i-1} - beta * omega * v_{i-1} ---*/
//		double beta_omega = -beta*omega;
//		p.Equals_AX_Plus_BY(beta, p, beta_omega, v);
//		p.Plus_AX(1.0, r);
//		
//		/*--- Preconditioning step ---*/
//		precond(p, phat);
//		mat_vec(phat, v);
//    
//		/*--- Calculate step-length alpha ---*/
//    double r_0_v = dotProd(r_0, v);
//    alpha = rho / r_0_v;
//    
//		/*--- s_{i} = r_{i-1} - alpha * v_{i} ---*/
//		s.Equals_AX_Plus_BY(1.0, r, -alpha, v);
//		
//		/*--- Preconditioning step ---*/
//		precond(s, shat);
//		mat_vec(shat, t);
//    
//		/*--- Calculate step-length omega ---*/
//    omega = dotProd(t, s) / dotProd(t, t);
//    
//		/*--- Update solution and residual: ---*/
//    x.Plus_AX(alpha, phat); x.Plus_AX(omega, shat);
//		r.Equals_AX_Plus_BY(1.0, s, -omega, t);
//    
//    /*--- Check if solution has converged, else output the relative residual if necessary ---*/
//    norm_r = r.norm();
//    if (norm_r < tol*norm0) break;
//    if (((monitoring) && (rank == 0)) && ((i+1) % 5 == 0) && (rank == 0)) writeHistory(i+1, norm_r, norm0);
//    
//  }
	return 1;
}
