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

#ifndef _MACRO_DEFS_H_
#define _MACRO_DEFS_H_

#include "config_lmpp.h"

#define LCAT_(a, b)    #a b
#define LCAT(a, b)    LCAT_(a, b) // force substitution
#define RCAT_(a, b)    a #b
#define RCAT(a, b)    RCAT_(a, b) // force substitution

#define LMPP_MK_SLAPACK_NAME(n)  LMPP_S_PREFIX(LMPP_CAT_(n, LMPP_LAPACK_SUFFIX))
#define LMPP_MK_DLAPACK_NAME(n)  LMPP_D_PREFIX(LMPP_CAT_(n, LMPP_LAPACK_SUFFIX))

#ifdef LMPP_BLAS_PREFIX
#define LMPP_MK_SBLAS_NAME(n) LMPP_CAT_(LMPP_BLAS_PREFIX, LMPP_S_PREFIX(LMPP_CAT_(n, LMPP_BLAS_SUFFIX)))
#define LMPP_MK_DBLAS_NAME(n) LMPP_CAT_(LMPP_BLAS_PREFIX, LMPP_D_PREFIX(LMPP_CAT_(n, LMPP_BLAS_SUFFIX)))
#else
#define LMPP_MK_SBLAS_NAME(n) LMPP_S_PREFIX(LMPP_CAT_(n, LMPP_BLAS_SUFFIX))
#define LMPP_MK_DBLAS_NAME(n) LMPP_D_PREFIX(LMPP_CAT_(n, LMPP_BLAS_SUFFIX))
#endif


#define __BLOCKSZ__       64 /* block size for cache-friendly matrix-matrix multiply. It should be
							  * such that __BLOCKSZ__^2*sizeof(LMPP_REAL) is smaller than the CPU (L1)
							  * data cache size. Notice that a value of 64 when LMPP_REAL=double assumes
							  * an 32Kb L1 data cache (64*64*8=32K).
							  */
#define __BLOCKSZ__SQ    (__BLOCKSZ__)*(__BLOCKSZ__)

/* add a prefix in front of a token */
#define LMPP_CAT__(a, b) a ## b
#define LMPP_CAT_(a, b) LMPP_CAT__(a, b) // force substitution
#define LMPP_S_PREFIX(n) LMPP_CAT_(s, n)
#define LMPP_D_PREFIX(n) LMPP_CAT_(d, n)

#endif // _MACRO_DEFS_H_
