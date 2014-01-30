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
//#include "cusparse_v2.h"
//#include "cublas_v2.h"

#define NSHARED 3
#include <sys/time.h>

#include "../include/matrix_structure.hpp"

__global__ void matmul(const double *A, unsigned long *row_ptr, unsigned long *col_ind, unsigned long nPoint, unsigned long nVar, unsigned long nEqn, double *x,double *r) {

  extern __shared__ double block[];
  int iBlock, jBlock, iVar, jVar, index, block_index;
  iBlock = 2*(blockIdx.x*gridDim.y + blockIdx.y);
  iVar = threadIdx.x;
  block[0] = row_ptr[iBlock];
  block[1] = row_ptr[iBlock+1];
  block[2] = row_ptr[iBlock+2];
  __syncthreads();
  iBlock = 2*(blockIdx.x*gridDim.y + blockIdx.y) + threadIdx.z;
  if (iBlock >= nPoint) return;
  block_index = NSHARED+threadIdx.z*blockDim.y*blockDim.x+threadIdx.x*blockDim.y+threadIdx.y;
  block[block_index] = 0.;
  //if (threadIdx.x >= row_ptr[iBlock+1]-row_ptr[iBlock]) return;
  //index = row_ptr[iBlock]+threadIdx.y;
  if (threadIdx.y >= block[1+threadIdx.z]-block[threadIdx.z]) return;
  index = block[threadIdx.z]+threadIdx.y;
  jBlock = col_ind[index];
  double sum = 0.;
  for (jVar = 0; jVar < nVar; jVar++) {
    sum -= A[index*nVar*nEqn+iVar*nVar+jVar]*x[jBlock*nVar+jVar];
  }
  block[block_index] = sum;
  if (threadIdx.y) return;
  __syncthreads();
  //for (index = 1; index < row_ptr[iBlock+1]-row_ptr[iBlock]; index++) {
  for (index = 1; index < block[threadIdx.z+1]-block[threadIdx.z]; index++) {
    block[block_index] += block[block_index+index];
  }
  r[iBlock*nEqn+iVar] += block[block_index];
}

//__device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//                             (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//old = atomicCAS(address_as_ull, assumed,
//                        __double_as_longlong(val +
//                               __longlong_as_double(assumed)));
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}
//
//__global__ void matmul_block(double *A, unsigned long *row_ptr, unsigned long *col_ind, unsigned long nPoint, unsigned long nVar, unsigned long nEqn, double alpha,double *x,double *r) {
//
//  unsigned long iBlock, jBlock, iVar, jVar, index, row_index;
//  index = blockIdx.x*blockDim.x+threadIdx.x;
//  if (index >= row_ptr[nPoint]) return;
//  jBlock = col_ind[index];
//  for (row_index = 0; row_index < nPoint; row_index++) {
//    if (index >= row_ptr[row_index]) {
//	  iBlock = row_index;
//	  break;
//	}
//  }
//  for (iVar = 0; iVar < nEqn; iVar++) {
//    for (jVar = 0; jVar < nVar; jVar++) {
//      r[iBlock*nEqn+iVar] += alpha*A[index*nVar*nEqn+iVar*nVar+jVar]*x[jBlock*nVar+jVar];
//     // atomicAdd(&r[iBlock*nEqn+iVar],alpha*A[index*nVar*nEqn+iVar*nVar+jVar]*x[jBlock*nVar+jVar]);
//  	}
//  }
//}

void cudasafe(char*message, cudaError_t error)
{
	if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %i\n",message,error); exit(-1); }
}

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
  double *d_A, *d_b, *d_x, *d_r, *d_r_0, *d_norm;
  unsigned long *d_row_ptr, *d_col_ind;
  cudasafe("Malloc A",cudaMalloc(&d_A,matrix->nnz*matrix->nVar*matrix->nEqn*sizeof(double)));
  cudasafe("Malloc row_ptr",cudaMalloc(&d_row_ptr,(matrix->nPoint+1)*sizeof(unsigned long)));
  cudasafe("Malloc col_ind",cudaMalloc(&d_col_ind,matrix->nnz*sizeof(unsigned long)));
  cudasafe("Malloc b",cudaMalloc(&d_b,matrix->nPoint*matrix->nEqn*sizeof(double)));
  cudasafe("Malloc x",cudaMalloc(&d_x,matrix->nPoint*matrix->nVar*sizeof(double)));
  cudasafe("Malloc r",cudaMalloc(&d_r,matrix->nPoint*matrix->nEqn*sizeof(double)));
  cudasafe("Malloc r_0",cudaMalloc(&d_r_0,matrix->nPoint*matrix->nEqn*sizeof(double)));
  cudasafe("Malloc d_norm",cudaMalloc(&d_norm,sizeof(double)));

  //last = current;
  cudasafe("Copy A",cudaMemcpy(d_A,matrix->matrix,matrix->nnz*matrix->nVar*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice));
  cudasafe("Copy row_ptr",cudaMemcpy(d_row_ptr,matrix->row_ptr,(matrix->nPoint+1)*sizeof(unsigned long),cudaMemcpyHostToDevice));
  cudasafe("Copy col_ind",cudaMemcpy(d_col_ind,matrix->col_ind,matrix->nnz*sizeof(unsigned long),cudaMemcpyHostToDevice));
  cudasafe("Copy b",cudaMemcpy(d_b,b.vec_val,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice));
  cudasafe("Copy r",cudaMemcpy(d_r,b.vec_val,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice));
  //cudasafe("Copy r_0",cudaMemcpy(d_r_0,b.vec_val,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "gpu init -> " << diff << endl;
  last = current;

  CSysVector b2(b);
  CSysVector r(b);
  CSysVector r_0(b);
  CSysVector g_r_0(b);
  CSysVector p(b);
	CSysVector v(b);
  CSysVector s(b);
	CSysVector t(b);
	CSysVector phat(b);
	CSysVector shat(b);
  CSysVector A_x(b);
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "cpu init " << diff << endl;
  last = current;
  
  /*--- Calculate the initial residual, compute norm, and check if system is already solved ---*/
	mat_vec(x,A_x);
  r -= A_x; r_0 = r; // recall, r holds b initially
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "cpu mat_vec " << diff << endl;
  last = current;
  double norm_r = r.norm();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "cpu r.norm() " << diff << endl;
  last = current;
  cerr << "cpu norm_r = " << norm_r << endl;
  double norm0 = b.norm();
  if ( (norm_r < tol*norm0) || (norm_r < eps) ) {
    if (rank == 0) cout << "CSysSolve::BCGSTAB(): system solved by initial guess." << endl;
    return 0;
  }
	
	/*--- Initialization ---*/
  double alpha = 1.0, beta = 1.0, omega = 1.0, rho = 1.0, rho_prime = 1.0;
	
  /*--- Set the norm to the initial initial residual value ---*/
  norm0 = norm_r;
  
  /*--- Output header information including initial residual ---*/
  int i = 0;
  if ((monitoring) && (rank == 0)) {
    writeHeader("BCGSTAB", tol, norm_r);
    writeHistory(i, norm_r, norm0);
  }
	
  /*---  Loop over all search directions ---*/
  for (i = 0; i < m; i++) {
		
		/*--- Compute rho_prime ---*/
		rho_prime = rho;
		
		/*--- Compute rho_i ---*/
		rho = dotProd(r, r_0);
		
		/*--- Compute beta ---*/
		beta = (rho / rho_prime) * (alpha /omega);
		
		/*--- p_{i} = r_{i-1} + beta * p_{i-1} - beta * omega * v_{i-1} ---*/
		double beta_omega = -beta*omega;
		p.Equals_AX_Plus_BY(beta, p, beta_omega, v);
		p.Plus_AX(1.0, r);
		
		/*--- Preconditioning step ---*/
		precond(p, phat);
		mat_vec(phat, v);
    
		/*--- Calculate step-length alpha ---*/
    double r_0_v = dotProd(r_0, v);
    alpha = rho / r_0_v;
    
		/*--- s_{i} = r_{i-1} - alpha * v_{i} ---*/
		s.Equals_AX_Plus_BY(1.0, r, -alpha, v);
		
		/*--- Preconditioning step ---*/
		precond(s, shat);
		mat_vec(shat, t);
    
		/*--- Calculate step-length omega ---*/
    omega = dotProd(t, s) / dotProd(t, t);
    
		/*--- Update solution and residual: ---*/
    x.Plus_AX(alpha, phat); x.Plus_AX(omega, shat);
		r.Equals_AX_Plus_BY(1.0, s, -omega, t);
    
    /*--- Check if solution has converged, else output the relative residual if necessary ---*/
    norm_r = r.norm();
    if (norm_r < tol*norm0) break;
    if (((monitoring) && (rank == 0)) && ((i+1) % 5 == 0) && (rank == 0)) writeHistory(i+1, norm_r, norm0);
    
  }
	  
  if ((monitoring) && (rank == 0)) {
    cout << "# BCGSTAB final (true) residual:" << endl;
    cout << "# Iteration = " << i << ": |res|/|res0| = "  << norm_r/norm0 << endl;
  }
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "cpu end " << diff << endl;
  last = current;
  mat_vec(x,A_x);
  b2 -= A_x;
  cerr << "cpu finalnorm " << b2.norm() << endl;
  cudasafe("Copy x",cudaMemcpy(d_x,x.vec_val,matrix->nPoint*matrix->nVar*sizeof(double),cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "gpu copy x " << diff << endl;
  last = current;

  dim3 griddim;
  griddim.x = ceil(sqrt(matrix->nPoint));
  griddim.y = ceil(sqrt(matrix->nPoint));
  //griddim.z = matrix->nVar;
  dim3 blockdim;
  blockdim.x = matrix->nVar;
  blockdim.y = 0;
  blockdim.z = 2;
  for (i=0; i < matrix->nPoint; i++) {
	  int d = matrix->row_ptr[i+1]-matrix->row_ptr[i];
	  if (d > blockdim.y) {
		  blockdim.y = d;
	  }
  }
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "calc blockdim " << diff << endl;
  last = current;

  int shared_mem = (NSHARED + blockdim.x*blockdim.y*blockdim.z)*sizeof(double);
  //griddim.z = matrix->nVar;
  //matmul<<<griddim,1>>>(d_A,d_row_ptr,d_col_ind,matrix->nPoint,matrix->nVar,matrix->nEqn,-1.,d_x,d_r);
  matmul<<<griddim,blockdim,shared_mem>>>(d_A,d_row_ptr,d_col_ind,matrix->nPoint,matrix->nVar,matrix->nEqn,d_x,d_r);
  //matmul_block<<<(matrix->nnz+1023)/1024,1024>>>(d_A,d_row_ptr,d_col_ind,matrix->nPoint,matrix->nVar,matrix->nEqn,-1.,d_x,d_r);
	cudaDeviceSynchronize();
  cudasafe("matmul_block",cudaGetLastError());
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "gpu mat_vec " << diff << endl;
  last = current;

  CSysVector r2(b);
  //cerr << "pregpu r.norm() = " << r2.norm() << endl;
  cudasafe("Copy r <",cudaMemcpy(r2.vec_val,d_r,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyDeviceToHost));
  gettimeofday(&current,NULL);
  diff = current.tv_sec-last.tv_sec;
  diff += 1.e-6*(current.tv_usec-last.tv_usec);
  cerr << "Copy r < " << diff << endl;
  //cerr << "gpu r[0] = " << r2.vec_val[0] << endl;
  //cerr << "gpu r[1] = " << r2.vec_val[1] << endl;
  //cerr << "gpu r[2] = " << r2.vec_val[2] << endl;
  //cerr << "gpu r[3] = " << r2.vec_val[3] << endl;
  //cerr << "gpu r[4] = " << r2.vec_val[4] << endl;
  //cerr << "gpu r[5] = " << r2.vec_val[5] << endl;
  //cerr << "gpu r[6] = " << r2.vec_val[6] << endl;
  //cerr << "nPoint = " << matrix->nPoint << endl;
  //cerr << "nnz = " << matrix->nnz << endl;
//  for (unsigned long iPoint = 0; iPoint < 31; iPoint++)
//	  cerr << "row_ptr[" << iPoint << "] = " << matrix->row_ptr[iPoint] << endl;
  cerr << "gpu r.norm() = " << r2.norm() << endl;
  //last = current;
//  cudasafe("Copy < r_0",cudaMemcpy(g_r_0.vec_val,d_r,matrix->nPoint*matrix->nEqn*sizeof(double),cudaMemcpyDeviceToHost));
//  gettimeofday(&current,NULL);
//  diff = current.tv_sec-last.tv_sec;
//  diff += 1.e-6*(current.tv_usec-last.tv_usec);
//  cerr << "gpu copy r_0 <- " << diff << endl;
//  last = current;
//	cerr << "gpu norm_r = " << g_r_0.norm() << endl;
  cudasafe("Free A",cudaFree(d_A));
  cudasafe("Free row_ptr",cudaFree(d_row_ptr));
  cudasafe("Free col_ind",cudaFree(d_col_ind));
  cudasafe("Free b",cudaFree(d_b));
  cudasafe("Free x",cudaFree(d_x));
  cudasafe("Free r",cudaFree(d_r));
  cudasafe("Free r_0",cudaFree(d_r_0));
  exit(0);
	return 1;
}
