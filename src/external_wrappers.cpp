/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//                              LICENSE NOTICES                                //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

//  This file is part of the program
//  Levenberg-Marquardt C++ non-linear minimization algorithm (henceforth LM++)
//  Copyright (C) 2021  Manuel Portilheiro
//
//  LM++ is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  LM++ is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with LM++.  If not, see <https://www.gnu.org/licenses/>.
//
//	LM++ is a modification of the Levenberg - Marquardt non-linear minimization
//  algorithm, Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr),
//  version 2.6. A copy of the original copyright notice follows.

//  ORIGINAL COPYRIGHT NOTICE:

	/* 
	////////////////////////////////////////////////////////////////////////////////////
	// 
	//  Prototypes and definitions for the Levenberg - Marquardt minimization algorithm
	//  Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr)
	//  Institute of Computer Science, Foundation for Research & Technology - Hellas
	//  Heraklion, Crete, Greece.
	//
	//  This program is free software; you can redistribute it and/or modify
	//  it under the terms of the GNU General Public License as published by
	//  the Free Software Foundation; either version 2 of the License, or
	//  (at your option) any later version.
	//
	//  This program is distributed in the hope that it will be useful,
	//  but WITHOUT ANY WARRANTY; without even the implied warranty of
	//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	//  GNU General Public License for more details.
	//
	////////////////////////////////////////////////////////////////////////////////////
	*/

#include <algorithm>
#include "external_wrappers.h"
#include "config_lmpp.h"

#ifdef LMPP_HAVE_MKL
#include <mkl.h>
#endif

#ifdef LMPP_HAVE_LAPACK

	#ifdef LMPP_SNGL_PREC
		#define LMPP_SNRM2  LMPP_MK_SBLAS_NAME(nrm2)
		#define LMPP_SGEMM  LMPP_MK_SBLAS_NAME(gemm)
		#define LMPP_SGEQP3 LMPP_MK_SLAPACK_NAME(geqp3)
		#define LMPP_STRTRI LMPP_MK_SLAPACK_NAME(trtri)
		#define LMPP_SGESVD LMPP_MK_SLAPACK_NAME(gesvd)
		#define LMPP_SGESDD LMPP_MK_SLAPACK_NAME(gesdd)
		#define LMPP_SPOTF2 LMPP_MK_SLAPACK_NAME(potf2)
		#define LMPP_SGEQRF LMPP_MK_SLAPACK_NAME(geqrf)
		#define LMPP_SORGQR LMPP_MK_SLAPACK_NAME(orgqr)
		#define LMPP_STRTRS LMPP_MK_SLAPACK_NAME(trtrs)
		#define LMPP_SPOTF2 LMPP_MK_SLAPACK_NAME(potf2)
		#define LMPP_SPOTRF LMPP_MK_SLAPACK_NAME(potrf)
		#define LMPP_SPOTRS LMPP_MK_SLAPACK_NAME(potrs)
		#define LMPP_SGETRF LMPP_MK_SLAPACK_NAME(getrf)
		#define LMPP_SGETRS LMPP_MK_SLAPACK_NAME(getrs)
		#define LMPP_SGESVD LMPP_MK_SLAPACK_NAME(gesvd)
		#define LMPP_SGESDD LMPP_MK_SLAPACK_NAME(gesdd)
		#define LMPP_SSYTRF LMPP_MK_SLAPACK_NAME(sytrf)
		#define LMPP_SSYTRS LMPP_MK_SLAPACK_NAME(sytrs)
	#endif // LMPP_SNGL_PREC

	#ifdef LMPP_DBL_PREC
		#define LMPP_DNRM2  LMPP_MK_DBLAS_NAME(nrm2)
		#define LMPP_DGEMM  LMPP_MK_DBLAS_NAME(gemm)
		#define LMPP_DGEQP3 LMPP_MK_DLAPACK_NAME(geqp3)
		#define LMPP_DTRTRI LMPP_MK_DLAPACK_NAME(trtri)
		#define LMPP_DGESVD LMPP_MK_DLAPACK_NAME(gesvd)
		#define LMPP_DGESDD LMPP_MK_DLAPACK_NAME(gesdd)
		#define LMPP_DPOTF2 LMPP_MK_DLAPACK_NAME(potf2)
		#define LMPP_DGEQRF LMPP_MK_DLAPACK_NAME(geqrf)
		#define LMPP_DORGQR LMPP_MK_DLAPACK_NAME(orgqr)
		#define LMPP_DTRTRS LMPP_MK_DLAPACK_NAME(trtrs)
		#define LMPP_DPOTF2 LMPP_MK_DLAPACK_NAME(potf2)
		#define LMPP_DPOTRF LMPP_MK_DLAPACK_NAME(potrf)
		#define LMPP_DPOTRS LMPP_MK_DLAPACK_NAME(potrs)
		#define LMPP_DGETRF LMPP_MK_DLAPACK_NAME(getrf)
		#define LMPP_DGETRS LMPP_MK_DLAPACK_NAME(getrs)
		#define LMPP_DGESVD LMPP_MK_DLAPACK_NAME(gesvd)
		#define LMPP_DGESDD LMPP_MK_DLAPACK_NAME(gesdd)
		#define LMPP_DSYTRF LMPP_MK_DLAPACK_NAME(sytrf)
		#define LMPP_DSYTRS LMPP_MK_DLAPACK_NAME(sytrs)
	#endif // LMPP_DBL_PREC

	extern "C" {
		#ifdef LMPP_SNGL_PREC
			/* BLAS matrix multiplication, vector norm, LAPACK SVD & Cholesky routines */

			/* Vector 2-norm, avoids overflows */
			extern float LMPP_SNRM2( const int* n, const float* dx, const int *incx );

			/* C := alpha*op( A )*op( B ) + beta*C */
			extern void LMPP_SGEMM( const char* transa, const char* transb, const int* m, const int* n, const int* k,
									const float* alpha, const float* a, const int* lda, const float* b, const int* ldb,
									const float* beta, float* c, const int* ldc );

			extern void LMPP_SGEQP3( const int* m, const int* n, float* a, const int* lda,
									 int* jpvt, float* tau, float* work, const int* lwork, int* info );

			extern void LMPP_STRTRI( const char* uplo, const char* diag, const int* n, float* a,
									 const int* lda, int* info );


			/* lapack 3.0 new SVD routine, xgesdd() is faster than xgesvd() */
			extern void LMPP_SGESVD( const char* jobu, const char* jobvt, const int* m, const int* n, float* a,
									 const int* lda, float* s, float* u, const int* ldu, float* vt, const int* ldvt,
									 float* work, const int* lwork, int* info );
			extern void LMPP_SGESDD( const char* jobz, const int* m, const int* n, float* a, const int* lda,
									 float* s, float* u, const int* ldu, float* vt, const int* ldvt, float* work,
									 const int* lwork, int* iwork, int* info );

			/* Cholesky decomposition and systems solution */
			extern void LMPP_SPOTF2( const char *uplo, const int *n, float *a, const int *lda, int *info );
			extern void LMPP_SPOTRF( const char *uplo, const int *n, float *a, const int *lda, int *info ); /* block version of dpotf2 */
			extern void LMPP_SPOTRS( const char *uplo, const int *n, const int *nrhs, const float *a, const int *lda,
									 float *b, const int *ldb, int *info );
			/* QR decomposition */
			extern void LMPP_SGEQRF( const int* m, const int* n, float* a, const int* lda, float* tau,
									 float* work, const int* lwork, int* info );
			extern void LMPP_SORGQR( const int* m, const int* n, const int* k, float* a, const int* lda,
									 const float* tau, float* work, const int* lwork, int* info );
			/* solution of triangular systems */
			extern void LMPP_STRTRS( const char* uplo, const char* trans, const char* diag,
									 const int* n, const int* nrhs, const float* a,
									 const int* lda, float* b, const int* ldb, int* info );
			/* LU decomposition and systems solution */
			extern void LMPP_SGETRF( const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info );
			extern void LMPP_SGETRS( const char* trans, const int* n, const int* nrhs,
									 const float* a, const int* lda, const int* ipiv, float* b,
									 const int* ldb, int* info );
			/* LDLt/UDUt factorization and systems solution */
			extern void LMPP_SSYTRF( const char* uplo, const int* n, float* a, const int* lda,
									 int* ipiv, float* work, const int* lwork, int* info );
			extern void LMPP_SSYTRS( const char* uplo, const int* n, const int* nrhs,
									 const float* a, const int* lda, const int* ipiv, float* b,
									 const int* ldb, int* info );
		#endif // LMPP_SNGL_PREC

		#ifdef LMPP_DBL_PREC
			/* BLAS matrix multiplication, vector norm, LAPACK SVD & Cholesky routines */

			/* Vector 2-norm, avoids overflows */
			extern double LMPP_DNRM2( const int* n, const double* dx, const int *incx );

			/* C := alpha*op( A )*op( B ) + beta*C */
			extern void LMPP_DGEMM( const char *transa, const char *transb, const int *m, const int *n, const int *k,
									const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
									const double *beta, double *c, const int *ldc );

			extern void LMPP_DGEQP3( const int* m, const int* n, double* a, const int* lda,
									 int* jpvt, double* tau, double* work, const int* lwork, int* info );

			extern void LMPP_DTRTRI( const char* uplo, const char* diag, const int* n, double* a,
									 const int* lda, int* info );

			/* lapack 3.0 new SVD routine, xgesdd() is faster than xgesvd() */
			extern void LMPP_DGESVD( const char* jobu, const char* jobvt, const int* m, const int* n, double* a, const int* lda,
									 double* s, double* u, const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork,
									 int* info );
			extern void LMPP_DGESDD( const char* jobz, const int* m, const int* n, double* a, const int* lda,
									 double* s, double* u, const int* ldu, double* vt, const int* ldvt, double* work,
									 const int* lwork, int* iwork, int* info );
			/* Cholesky decomposition and systems solution */
			extern void LMPP_DPOTF2( const char* uplo, const int* n, double* a, const int* lda, int* info );
			extern void LMPP_DPOTRF( const char *uplo, const int *n, double *a, const int *lda, int *info ); /* block version of dpotf2 */
			extern void LMPP_DPOTRS( const char *uplo, const int *n, const int *nrhs, const double *a, const int *lda,
									 double *b, const int *ldb, int *info );
			/* QR decomposition */
			extern void LMPP_DGEQRF( const int* m, const int* n, double* a, const int* lda, double* tau,
									 double* work, const int* lwork, int* info );
			extern void LMPP_DORGQR( const int* m, const int* n, const int* k, double* a, const int* lda,
									 const double* tau, double* work, const int* lwork, int* info );
			/* solution of triangular systems */
			extern void LMPP_DTRTRS( const char* uplo, const char* trans, const char* diag,
									 const int* n, const int* nrhs, const double* a,
									 const int* lda, double* b, const int* ldb, int* info );
			/* LU decomposition and systems solution */
			extern void LMPP_DGETRF( const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info );
			extern void LMPP_DGETRS( const char* trans, const int* n, const int* nrhs,
									 const double* a, const int* lda, const int* ipiv, double* b,
									 const int* ldb, int* info );
			/* LDLt/UDUt factorization and systems solution */
			extern void LMPP_DSYTRF( const char* uplo, const int* n, double* a, const int* lda,
									 int* ipiv, double* work, const int* lwork, int* info );
			extern void LMPP_DSYTRS( const char* uplo, const int* n, const int* nrhs,
									 const double* a, const int* lda, const int* ipiv, double* b,
									 const int* ldb, int* info );
		#endif // LMPP_DBL_PREC

	}

	#ifdef LMPP_SNGL_PREC
		/* Vector 2-norm, avoids overflows */
		template<>
		float lm_nrm2<float>( int n, float* x )
		{
			int one = 1;
			return LMPP_SNRM2( &n, x, &one );
		}

		/* C := alpha*op( A )*op( B ) + beta*C */
		template<>
		void lm_gemm<float>( const char *transa, const char *transb, const int *m, const int *n, const int *k,
							 const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
							 const float *beta, float *c, const int *ldc )
		{
			LMPP_SGEMM( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
		}

		template<>
		void lm_geqp3<float>( const int* m, const int* n, float* a, const int* lda,
							  int* jpvt, float* tau, float* work, const int* lwork, int* info )
		{
			LMPP_SGEQP3( m, n, a, lda, jpvt, tau,  work, lwork, info );
		}

		template<>
		void lm_trtri<float>( const char* uplo, const char* diag, const int* n, float* a,
							  const int* lda, int* info )
		{
			LMPP_STRTRI( uplo, diag, n, a, lda, info );
		}

		/* SVD routines */
		template<>
		void lm_gesvd<float>( const char* jobu, const char* jobvt, const int* m, const int* n, float* a, const int* lda,
							  float* s, float* u, const int* ldu, float* vt, const int* ldvt, float* work, const int* lwork,
							  int* info )
		{
			LMPP_SGESVD( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info );
		}
		template<>
		void lm_gesdd<float>( const char* jobz, const int* m, const int* n, float* a, const int* lda,
							  float* s, float* u, const int* ldu, float* vt, const int* ldvt, float* work,
							  const int* lwork, int* iwork, int* info )
		{
			return LMPP_SGESDD( jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info );
		}

		/* Cholesky decomposition and systems solution */
		template<>
		void lm_potf2<float>( const char *uplo, const int *n, float *a, const int *lda, int *info )
		{
			LMPP_SPOTF2( uplo, n, a, lda, info );
		}

		template<>
		void lm_potrf<float>( const char *uplo, const int *n, float *a, const int *lda, int *info ) /* block version of dpotf2 */
		{
			LMPP_SPOTRF( uplo, n, a, lda, info );
		}

		template<>
		void lm_potrs<float>( const char *uplo, const int *n, const int *nrhs, const float *a, const int *lda,
							  float *b, const int *ldb, int *info )
		{
			LMPP_SPOTRS( uplo, n, nrhs, a, lda, b, ldb, info );
		}

		/* QR decomposition */
		template<>
		void lm_geqrf<float>( const int* m, const int* n, float* a, const int* lda, float* tau,
							  float* work, const int* lwork, int* info )
		{
			LMPP_SGEQRF( m, n, a, lda, tau, work, lwork, info );
		}

		template<>
		void lm_orgqr<float>( const int* m, const int* n, const int* k, float* a, const int* lda,
							  const float* tau, float* work, const int* lwork, int* info )
		{
			LMPP_SORGQR( m, n, k, a, lda, tau, work, lwork, info );
		}

		/* solution of triangular systems */
		template<>
		void lm_trtrs<float>( const char* uplo, const char* trans, const char* diag,
							  const int* n, const int* nrhs, const float* a,
							  const int* lda, float* b, const int* ldb, int* info )
		{
			LMPP_STRTRS( uplo, trans, diag, n, nrhs, a, lda, b, ldb, info );
		}

		/* LU decomposition and systems solution */
		template<>
		void lm_getrf<float>( const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info )
		{
			LMPP_SGETRF( m, n, a, lda, ipiv, info );
		}

		template<>
		void lm_getrs<float>( const char* trans, const int* n, const int* nrhs,
							  const float* a, const int* lda, const int* ipiv, float* b,
							  const int* ldb, int* info )
		{
			LMPP_SGETRS( trans, n, nrhs, a, lda, ipiv, b, ldb, info );
		}

		/* LDLt/UDUt factorization and systems solution */
		template<>
		void lm_sytrf<float>( const char* uplo, const int* n, float* a, const int* lda,
							  int* ipiv, float* work, const int* lwork, int* info )
		{
			LMPP_SSYTRF( uplo, n, a, lda, ipiv, work, lwork, info );
		}

		template<>
		void lm_sytrs<float>( const char* uplo, const int* n, const int* nrhs,
							  const float* a, const int* lda, const int* ipiv, float* b,
							  const int* ldb, int* info )
		{
			LMPP_SSYTRS( uplo, n, nrhs, a, lda, ipiv, b, ldb, info );
		}
	#endif // LMPP_SNGL_PREC

	#ifdef LMPP_DBL_PREC
		/* Vector 2-norm, avoids overflows */
		template<>
		double lm_nrm2<double>( int n, double* x )
		{
			int one = 1;
			return LMPP_DNRM2( &n, x, &one );
		}

		/* C := alpha*op( A )*op( B ) + beta*C */
		template<>
		void lm_gemm<double>( const char *transa, const char *transb, const int *m, const int *n, const int *k,
							  const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
							  const double *beta, double *c, const int *ldc )
		{
			LMPP_DGEMM( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
		}

		template<>
		void lm_geqp3<double>( const int* m, const int* n, double* a, const int* lda,
							   int* jpvt, double* tau, double* work, const int* lwork, int* info )
		{
			LMPP_DGEQP3( m, n, a, lda, jpvt, tau,  work, lwork, info );
		}

		template<>
		void lm_trtri<double>( const char* uplo, const char* diag, const int* n, double* a,
							   const int* lda, int* info )
		{
			LMPP_DTRTRI( uplo, diag, n, a, lda, info );
		}

		/* SVD routines */
		template<>
		void lm_gesvd<double>( const char* jobu, const char* jobvt, const int* m, const int* n, double* a, const int* lda,
							   double* s, double* u, const int* ldu, double* vt, const int* ldvt, double* work, const int* lwork,
							   int* info )
		{
			LMPP_DGESVD( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info );
		}
		template<>
		void lm_gesdd<double>( const char* jobz, const int* m, const int* n, double* a, const int* lda,
							   double* s, double* u, const int* ldu, double* vt, const int* ldvt, double* work,
							   const int* lwork, int* iwork, int* info )
		{
			return LMPP_DGESDD( jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info );
		}

		/* Cholesky decomposition and systems solution */
		template<>
		void lm_potf2<double>( const char* uplo, const int* n, double* a, const int* lda, int* info )
		{
			LMPP_DPOTF2( uplo, n, a, lda, info );
		}
		template<>
		void lm_potrf<double>( const char *uplo, const int *n, double *a, const int *lda, int *info ) /* block version of dpotf2 */
		{
			LMPP_DPOTRF( uplo, n, a, lda, info );
		}

		template<>
		void lm_potrs<double>( const char *uplo, const int *n, const int *nrhs, const double *a, const int *lda,
							   double *b, const int *ldb, int *info )
		{
			LMPP_DPOTRS( uplo, n, nrhs, a, lda, b, ldb, info );
		}

		/* QR decomposition */
		template<>
		void lm_geqrf<double>( const int* m, const int* n, double* a, const int* lda, double* tau,
							   double* work, const int* lwork, int* info )
		{
			LMPP_DGEQRF( m, n, a, lda, tau, work, lwork, info );
		}

		template<>
		void lm_orgqr<double>( const int* m, const int* n, const int* k, double* a, const int* lda,
							   const double* tau, double* work, const int* lwork, int* info )
		{
			LMPP_DORGQR( m, n, k, a, lda, tau, work, lwork, info );
		}

		/* solution of triangular systems */
		template<>
		void lm_trtrs<double>( const char* uplo, const char* trans, const char* diag,
							   const int* n, const int* nrhs, const double* a,
							   const int* lda, double* b, const int* ldb, int* info )
		{
			LMPP_DTRTRS( uplo, trans, diag, n, nrhs, a, lda, b, ldb, info );
		}

		/* LU decomposition and systems solution */
		template<>
		void lm_getrf<double>( const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info )
		{
			LMPP_DGETRF( m, n, a, lda, ipiv, info );
		}

		template<>
		void lm_getrs<double>( const char* trans, const int* n, const int* nrhs,
							   const double* a, const int* lda, const int* ipiv, double* b,
							   const int* ldb, int* info )
		{
			LMPP_DGETRS( trans, n, nrhs, a, lda, ipiv, b, ldb, info );
		}

		/* LDLt/UDUt factorization and systems solution */
		template<>
		void lm_sytrf<double>( const char* uplo, const int* n, double* a, const int* lda,
							   int* ipiv, double* work, const int* lwork, int* info )
		{
			LMPP_DSYTRF( uplo, n, a, lda, ipiv, work, lwork, info );
		}

		template<>
		void lm_sytrs<double>( const char* uplo, const int* n, const int* nrhs,
							   const double* a, const int* lda, const int* ipiv, double* b,
							   const int* ldb, int* info )
		{
			LMPP_DSYTRS( uplo, n, nrhs, a, lda, ipiv, b, ldb, info );
		}
	#endif // LMPP_DBL_PREC

#else // LMPP_HAVE_LAPACK
	#ifdef LMPP_SNGL_PREC
		/* Vector 2-norm, avoids overflows */
		template<>
		float lm_nrm2<float>( int n, float* x )
		{
			float max_ = 0.0;
			for( int i = n-1; i >= 0; --i )
				max_ = std::max( max_, std::abs(x[i]) );

			float sum_ = 0.0;
			for( int i = n-1; i >= 0; --i )
			{
				float aux = x[i] / max_;
				sum_ += aux*aux;
			}

			return max_ * std::sqrt(sum_);
		}
	#endif // LMPP_SNGL_PREC

	#ifdef LMPP_DBL_PREC
		/* Vector 2-norm, avoids overflows */
		template<>
		double lm_nrm2<double>( int n, double* x )
		{
			double max_ = 0.0;
			for( int i = n-1; i >= 0; --i )
				max_ = std::max( max_, std::abs(x[i]) );

			double sum_ = 0.0;
			for( int i = n-1; i >= 0; --i )
			{
				double aux = x[i] / max_;
				sum_ += aux*aux;
			}

			return max_ * std::sqrt(sum_);
		}
	#endif // LMPP_DBL_PREC

#endif // LMPP_HAVE_LAPACK