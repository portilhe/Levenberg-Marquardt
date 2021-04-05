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
	/* Matrix multiplication, SVD & Cholesky routines */

	/* C := alpha*op( A )*op( B ) + beta*C */
	template<typename FLOATTYPE>
	void lm_gemm( char *transa, char *transb, int *m, int *n, int *k, FLOATTYPE *alpha,
				  FLOATTYPE *a, int *lda, FLOATTYPE *b, int *ldb, FLOATTYPE *beta, FLOATTYPE *c, int *ldc );

	/* SVD routines */
	template<typename FLOATTYPE>
	int lm_gesvd( char *jobu, char *jobvt, int *m, int *n, FLOATTYPE *a, int *lda, FLOATTYPE *s,
				  FLOATTYPE *u, int *ldu, FLOATTYPE *vt, int *ldvt, FLOATTYPE *work, int *lwork, int *info );

	template<typename FLOATTYPE>
	int lm_gesdd( char *jobz, int *m, int *n, FLOATTYPE *a, int *lda, FLOATTYPE *s, FLOATTYPE *u,
				  int *ldu, FLOATTYPE *vt, int *ldvt, FLOATTYPE *work, int *lwork, int *iwork, int *info);

	/* Cholesky decomposition */
	template<typename FLOATTYPE>
	void lm_potf2( char *uplo, int *n, FLOATTYPE *a, int *lda, int *info );

#endif // LMPP_HAVE_LAPACK

#endif // _EXTERNAL_WRAPPERS_H_
