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

#ifndef _EXTERNAL_WRAPPERS_H_
#define _EXTERNAL_WRAPPERS_H_

#include "macro_defs.h"

#ifdef LMPP_HAVE_LAPACK

	/* Matrix multiplication, vector norm, SVD & Cholesky routines */

	/* Vector 2-norm, avoids overflows */
	template<typename FLOATTYPE>
	FLOATTYPE lm_nrm2( int n, FLOATTYPE* x );

	/* C := alpha*op( A )*op( B ) + beta*C */
	template<typename FLOATTYPE>
	void lm_gemm( const char *transa, const char *transb, const int *m, const int *n, const int *k, const FLOATTYPE *alpha,
				  const FLOATTYPE *a, const int *lda, const FLOATTYPE *b, const int *ldb, const FLOATTYPE *beta, FLOATTYPE *c,
				  const int *ldc );

	template<typename FLOATTYPE>
	void lm_geqp3( const int* m, const int* n, FLOATTYPE* a, const int* lda, int* jpvt,
				  FLOATTYPE* tau, FLOATTYPE* work, const int* lwork, int* info );

	template<typename FLOATTYPE>
	void lm_trtri( const char* uplo, const char* diag, const int* n, FLOATTYPE* a, const int* lda, int* info );

	/* SVD routines */
	template<typename FLOATTYPE>
	void lm_gesvd( const char *jobu, const char *jobvt, const int *m, const int *n, FLOATTYPE *a, const int *lda, FLOATTYPE *s,
				   FLOATTYPE *u, const int *ldu, FLOATTYPE *vt, const int *ldvt, FLOATTYPE *work, const int *lwork, int *info );

	template<typename FLOATTYPE>
	void lm_gesdd( const char *jobz, const int *m, const int *n, FLOATTYPE *a, const int *lda, FLOATTYPE *s, FLOATTYPE *u,
				   const int *ldu, FLOATTYPE *vt, const int *ldvt, FLOATTYPE *work, const int *lwork, int *iwork, int *info );

	/* Cholesky decomposition and systems solution */
	template<typename FLOATTYPE>
	void lm_potf2( const char *uplo, const int *n, FLOATTYPE *a, const int *lda, int *info );

	template<typename FLOATTYPE>
	void lm_potrf( const char *uplo, const int *n, FLOATTYPE *a, const int *lda, int *info ); /* block version of dpotf2 */

	template<typename FLOATTYPE>
	void lm_potrs( const char *uplo, const int *n, const int *nrhs, const FLOATTYPE *a, const int *lda,
				   FLOATTYPE *b, const int *ldb, int *info );

	/* QR decomposition */
	template<typename FLOATTYPE>
	void lm_geqrf( const int *m, const int *n, FLOATTYPE *a, const int *lda, FLOATTYPE *tau, FLOATTYPE *work,
				   const int *lwork, int *info );

	template<typename FLOATTYPE>
	void lm_orgqr( const int *m, const int *n, const int *k, FLOATTYPE *a, const int *lda,
				   const FLOATTYPE *tau, FLOATTYPE *work, const int *lwork, int *info );

	/* solution of triangular systems */
	template<typename FLOATTYPE>
	void lm_trtrs( const char* uplo, const char* trans, const char* diag,
				   const int* n, const int* nrhs, const FLOATTYPE* a,
				   const int* lda, FLOATTYPE* b, const int* ldb, int* info );

	/* LU decomposition and systems solution */
	template<typename FLOATTYPE>
	void lm_getrf( const int* m, const int* n, FLOATTYPE* a, const int* lda, int* ipiv, int* info );

	template<typename FLOATTYPE>
	void lm_getrs( const char* trans, const int* n, const int* nrhs,
				   const FLOATTYPE* a, const int* lda, const int* ipiv, FLOATTYPE* b,
				   const int* ldb, int* info );

	/* LDLt/UDUt factorization and systems solution */
	template<typename FLOATTYPE>
	void lm_sytrf( const char* uplo, const int* n, FLOATTYPE* a, const int* lda,
				   int* ipiv, FLOATTYPE* work, const int* lwork, int* info );

	template<typename FLOATTYPE>
	void lm_sytrs( const char* uplo, const int* n, const int* nrhs,
				   const FLOATTYPE* a, const int* lda, const int* ipiv, FLOATTYPE* b,
				   const int* ldb, int* info );
#endif // LMPP_HAVE_LAPACK

#endif // _EXTERNAL_WRAPPERS_H_
