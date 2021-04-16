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

#ifndef _AX_EQ_B_H_
#define _AX_EQ_B_H_

#include <vector>
#include "config_lmpp.h"

#ifdef LMPP_HAVE_LAPACK

template<typename FLOATTYPE>
int Ax_eq_b_QR( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer );

template<typename FLOATTYPE>
int Ax_eq_b_QRLS( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, int n, std::vector<char>& buffer );

template<typename FLOATTYPE>
int Ax_eq_b_Chol( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer );

template<typename FLOATTYPE>
int Ax_eq_b_LU( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer );

template<typename FLOATTYPE>
int Ax_eq_b_SVD( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer );

template<typename FLOATTYPE>
int Ax_eq_b_BK( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer );

#else // LMPP_HAVE_LAPACK -- No LAPACK !

template<typename FLOATTYPE>
int Ax_eq_b_LU_noLapack( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )

#endif

#endif // _AX_EQ_B_H_
