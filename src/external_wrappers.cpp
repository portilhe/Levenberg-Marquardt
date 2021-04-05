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

#include "external_wrappers.h"

#ifdef LMPP_HAVE_MKL
#include <mkl.h>
#endif

#ifdef LMPP_HAVE_LAPACK

	#define SGEMM LMPP_MK_SBLAS_NAME(gemm)
	#define DGEMM LMPP_MK_DBLAS_NAME(gemm)
	#define SGESVD LMPP_MK_SLAPACK_NAME(gesvd)
	#define DGESVD LMPP_MK_DLAPACK_NAME(gesvd)
	#define SGESDD LMPP_MK_SLAPACK_NAME(gesdd)
	#define DGESDD LMPP_MK_DLAPACK_NAME(gesdd)
	#define SPOTF2 LMPP_MK_SLAPACK_NAME(potf2)
	#define DPOTF2 LMPP_MK_DLAPACK_NAME(potf2)

	extern "C" {
		/* BLAS matrix multiplication, LAPACK SVD & Cholesky routines */
		/* C := alpha*op( A )*op( B ) + beta*C */
		extern void SGEMM( char *transa, char *transb, int *m, int *n, int *k, float *alpha,
						   float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc );

		extern void DGEMM( char *transa, char *transb, int *m, int *n, int *k, double *alpha,
						   double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc );

		/* lapack 3.0 new SVD routine, faster than xgesvd() */
		extern int SGESVD( char *jobu, char *jobvt, int *m, int *n, float *a, int *lda, float *s,
						   float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info );

		extern int DGESVD( char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s,
						   double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info );

		extern int SGESDD( char *jobz, int *m, int *n, float *a, int *lda, float *s, float *u,
						   int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info );

		extern int DGESDD( char *jobz, int *m, int *n, double *a, int *lda, double *s, double *u,
						   int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info );

		/* Cholesky decomposition */
		extern void SPOTF2( char *uplo, int *n, float *a, int *lda, int *info );
		extern void DPOTF2( char *uplo, int *n, float *a, int *lda, int *info );
	}

	/* C := alpha*op( A )*op( B ) + beta*C */
	template<>
	void lm_gemm<float>( char *transa, char *transb, int *m, int *n, int *k, float *alpha,
						 float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc )
	{
		SGEMM( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
	}

	template<>
	void lm_gemm<double>( char *transa, char *transb, int *m, int *n, int *k, double *alpha,
						  double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc )
	{
		DGEMM( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
	}

	/* SVD routines */
	template<>
	int lm_gesvd<float>( char *jobu, char *jobvt, int *m, int *n, float *a, int *lda, float *s,
						 float *u, int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *info )
	{
		return SGESVD( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info );
	}

	template<>
	int lm_gesvd<double>( char *jobu, char *jobvt, int *m, int *n, double *a, int *lda, double *s,
						  double *u, int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *info )
	{
		return DGESVD( jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info );
	}

	template<>
	int lm_gesdd<float>( char *jobz, int *m, int *n, float *a, int *lda, float *s, float *u,
						 int *ldu, float *vt, int *ldvt, float *work, int *lwork, int *iwork, int *info )
	{
		return SGESDD( jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info );
	}

	template<>
	int lm_gesdd<double>( char *jobz, int *m, int *n, double *a, int *lda, double *s, double *u,
						  int *ldu, double *vt, int *ldvt, double *work, int *lwork, int *iwork, int *info )
	{
		return DGESDD( jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info );
	}

	/* Cholesky decomposition */
	template<>
	void lm_potf2<float>( char *uplo, int *n, float *a, int *lda, int *info )
	{
		SPOTF2( uplo, n, a, lda, info );
	}

	template<>
	void lm_potf2<double>( char *uplo, int *n, double *a, int *lda, int *info )
	{
		DPOTF2( uplo, n, a, lda, info );
	}

#endif // LMPP_HAVE_LAPACK