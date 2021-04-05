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

#ifndef _CONFIG_H_
#define _CONFIG_H_

/************************************* Start of configuration options *************************************/
/* Note that when compiling with CMake, this configuration section is automatically generated
* based on the user's input, see levmar.h.in
*/

/* specifies whether to use LAPACK or not. Using LAPACK is strongly recommended */
#define LMPP_HAVE_LAPACK
/* specifies whether to use MKL or not. Using LAPACK is strongly recommended */
#define LMPP_HAVE_MKL

/* specifies whether the PLASMA parallel library for multicore CPUs is available */
/* #undef HAVE_PLASMA */

/* to avoid the overhead of repeated mallocs(), routines in Axb.c can be instructed to
* retain working memory between calls. Such a choice, however, renders these routines
* non-reentrant and is not safe in a shared memory multiprocessing environment.
* Bellow, an attempt is made to issue a warning if this option is turned on and OpenMP
* is being used (note that this will work only if omp.h is included before levmar.h)
*/
//#define LINSOLVERS_RETAIN_MEMORY
#if (defined(_OPENMP))
# ifdef LINSOLVERS_RETAIN_MEMORY
#  ifdef _MSC_VER
#  pragma message("LINSOLVERS_RETAIN_MEMORY is not safe in a multithreaded environment and should be turned off!")
#  else
#  warning LINSOLVERS_RETAIN_MEMORY is not safe in a multithreaded environment and should be turned off!
#  endif /* _MSC_VER */
# endif /* LINSOLVERS_RETAIN_MEMORY */
#endif /* _OPENMP */

/* specifies whether double precision routines will be compiled or not */
#define LMPP_DBL_PREC
/* specifies whether single precision routines will be compiled or not */
#define LMPP_SNGL_PREC

/* common suffix for LAPACK subroutines. Define empty in case of no prefix. */
//#define LMPP_LAPACK_SUFFIX _
#define LMPP_LAPACK_SUFFIX  // define empty

/* common prefix for BLAS subroutines. Leave undefined in case of no prefix.
* You might also need to modify LMPP_BLAS_PREFIX below
*/
/* f2c'd BLAS */
//#define LMPP_BLAS_PREFIX f2c_
/* C BLAS */
//#define LMPP_BLAS_PREFIX cblas_

/* common suffix for BLAS subroutines */
#define LMPP_BLAS_SUFFIX  // define empty if a f2c_ or cblas_ prefix was defined for LMPP_BLAS_PREFIX above
//#define LMPP_BLAS_SUFFIX _ // use this in case of no BLAS prefix

/****************** End of configuration options, no changes necessary beyond this point ******************/

/* work arrays size for levmar_der and levmar_dif functions.
* should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
*/
#define LMPP_DER_WORKSZ(npar, nmeas) (2*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))
#define LMPP_DIF_WORKSZ(npar, nmeas) (4*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))

/* work arrays size for levmar_bc_der and levmar_bc_dif functions.
* should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
*/
#define LMPP_BC_DER_WORKSZ(npar, nmeas) (2*(nmeas) + 4*(npar) + (nmeas)*(npar) + (npar)*(npar))
#define LMPP_BC_DIF_WORKSZ(npar, nmeas) LMPP_BC_DER_WORKSZ((npar), (nmeas)) /* LEVMAR_BC_DIF currently implemented using LEVMAR_BC_DER()! */

/* work arrays size for levmar_lec_der and levmar_lec_dif functions.
* should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
*/
#define LMPP_LEC_DER_WORKSZ(npar, nmeas, nconstr) LMPP_DER_WORKSZ((npar)-(nconstr), (nmeas))
#define LMPP_LEC_DIF_WORKSZ(npar, nmeas, nconstr) LMPP_DIF_WORKSZ((npar)-(nconstr), (nmeas))

/* work arrays size for levmar_blec_der and levmar_blec_dif functions.
* should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
*/
#define LMPP_BLEC_DER_WORKSZ(npar, nmeas, nconstr) LMPP_LEC_DER_WORKSZ((npar), (nmeas)+(npar), (nconstr))
#define LMPP_BLEC_DIF_WORKSZ(npar, nmeas, nconstr) LMPP_LEC_DIF_WORKSZ((npar), (nmeas)+(npar), (nconstr))

/* work arrays size for levmar_bleic_der and levmar_bleic_dif functions.
* should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
*/
#define LMPP_BLEIC_DER_WORKSZ(npar, nmeas, nconstr1, nconstr2) LMPP_BLEC_DER_WORKSZ((npar)+(nconstr2), (nmeas)+(nconstr2), (nconstr1)+(nconstr2))
#define LMPP_BLEIC_DIF_WORKSZ(npar, nmeas, nconstr1, nconstr2) LMPP_BLEC_DIF_WORKSZ((npar)+(nconstr2), (nmeas)+(nconstr2), (nconstr1)+(nconstr2))

#define LMPP_OPTS_SZ      5 /* max(4, 5) */
#define LMPP_INFO_SZ     10
#define LMPP_ERROR       -1
#define LMPP_INIT_MU     1E-03
#define LMPP_STOP_THRESH 1E-17
#define LMPP_DIFF_DELTA  1E-06
#define LMPP_VERSION     "0.1 (April 2021)"

#endif // _CONFIG_H_