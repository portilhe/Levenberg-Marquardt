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

	/////////////////////////////////////////////////////////////////////////////////
	// 
	//  Levenberg - Marquardt non-linear minimization algorithm
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
	/////////////////////////////////////////////////////////////////////////////////

#ifndef _MISC_H_
#define _MISC_H_

template<typename FLOATTYPE>
using FPTR = void (*)( FLOATTYPE* p, FLOATTYPE* hx, int m, int n, void* adata );

/* blocking-based matrix multiply */
template<typename FLOATTYPE>
void levmar_trans_mat_mat_mult( FLOATTYPE* a, FLOATTYPE* b, int n, int m );

/* forward finite differences */
template<typename FLOATTYPE>
void levmar_fdif_forw_jac_approx( FPTR<FLOATTYPE> func,
								  FLOATTYPE*      p,
								  FLOATTYPE*      hx,
								  FLOATTYPE*      hxx,
								  FLOATTYPE       delta,
								  FLOATTYPE*      jac,
								  int             m,
								  int             n,
								  void*           adata );

/* central finite differences */
template<typename FLOATTYPE>
void levmar_fdif_cent_jac_approx( FPTR<FLOATTYPE> func,
								  FLOATTYPE*      p,
								  FLOATTYPE*      hxm,
								  FLOATTYPE*      hxp,
								  FLOATTYPE       delta,
								  FLOATTYPE*      jac,
								  int             m,
								  int             n,
								  void*           adata );

/* e=x-y and ||e|| */
template<typename FLOATTYPE>
FLOATTYPE levmar_L2nrmxmy( FLOATTYPE* e, FLOATTYPE* x, FLOATTYPE* y, int n );

/* covariance of LS fit */
template<typename FLOATTYPE>
int levmar_covar( FLOATTYPE* JtJ, FLOATTYPE* C, FLOATTYPE sumsq, int m, int n );

/* box constraints consistency check */
template<typename FLOATTYPE>
int levmar_box_check( FLOATTYPE* lb, FLOATTYPE* ub, int m );

/* Cholesky */
template<typename FLOATTYPE>
int levmar_chol( FLOATTYPE* C, FLOATTYPE* W, int m );


#endif /* _MISC_H_ */
