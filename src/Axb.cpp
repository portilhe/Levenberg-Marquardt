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

/******************************************************************************** 
 * LAPACK-based implementations for various linear system solvers.
 ********************************************************************************/

#include <vector>
#include <cstring>
#include <numeric>
#include <iostream>
#include <algorithm>

#include "Axb.h"
#include "misc.h"
#include "external_wrappers.h"

/* Solvers for the linear systems Ax=b. Solvers should NOT modify their A & B arguments! */

//#undef LMPP_HAVE_LAPACK
#ifdef LMPP_HAVE_LAPACK

/*
 * This function returns the solution of Ax = b
 *
 * The function is based on QR decomposition with explicit computation of Q:
 * If A=Q R with Q orthogonal and R upper triangular, the linear system becomes
 * Q R x = b or R x = Q^T b.
 * The last equation can be solved directly.
 *
 * A is mxm, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_QR( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )
{
	int info;
	static int nb = 0;
	if( nb == 0 )
	{
		int auxi= -1;
		FLOATTYPE auxf;
		// workspace query; optimal size is returned in auxf
		lm_geqrf<FLOATTYPE>( &m, &m, nullptr, &m, nullptr, &auxf, &auxi, &info );
		nb      = ((int)auxf)/m; // optimal worksize is m*nb
	}

	/* calculate required memory size */
	const int work_sz = nb*m;
	const int a_sz    = m*m;
	const int tau_sz  = m;
	const int r_sz    = m*m; /* only the upper triangular part really needed */
	const int buf_sz  = a_sz + tau_sz + r_sz + work_sz;

	if( buffer.size() < (std::size_t)buf_sz*sizeof(FLOATTYPE) )
	{
		buffer.resize( buf_sz*sizeof(FLOATTYPE) );
	}

	FLOATTYPE* a    = reinterpret_cast<FLOATTYPE*>( buffer.data() );
	FLOATTYPE* tau  = a   + a_sz;
	FLOATTYPE* r    = tau + tau_sz;
	FLOATTYPE* work = r   + r_sz;

	/* store A (column major!) into a */
	for( int i = 0; i < m; ++i )
	{
		for( int j = 0; j < m; ++j )
		{
			a[i+j*m] = A[i*m+j];
		}
	}

	/* QR decomposition of A */
	lm_geqrf( &m, &m, a, &m, tau, work, &work_sz, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_geqrf in Ax_eq_b_QR()\n";
			exit(1);
		}
		else
		{
			std::cerr << "Unknown LAPACK error " << info << " for lm_geqrf in Ax_eq_b_QR()\n";
			return 0;
		}
	}

	/* R is stored in the upper triangular part of a; copy it in r so that ORGQR() below won't destroy it */ 
	std::memcpy( r, a, r_sz * sizeof(FLOATTYPE) );

	/* compute Q using the elementary reflectors computed by the above decomposition */
	lm_orgqr( &m, &m, &m, a, &m, tau, work, &work_sz, &info );
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_orgqr in Ax_eq_b_QR()\n";
			exit(1);
		}
		else
		{
			std::cerr << "Unknown LAPACK error (" << info <<") in lm_orgqr called from Ax_eq_b_QR()\n";
			return 0;
		}
	}

	/* Q is now in a; compute Q^T b in x */
	for( int i = 0; i < m; ++i )
	{
		x[i] = std::accumulate( a + i*m, a + (i+1)*m,  zero<FLOATTYPE>() );
	}

	/* solve the linear system R x = Q^t b */
	int nrhs=1;
	lm_trtrs( "U", "N", "N", &m, &nrhs, r, &m, x, &m, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_trtrs in Ax_eq_b_QR()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: the " << info <<"-th diagonal element of A is zero (singular matrix) in lm_trtrs called from Ax_eq_b_QR()\n";
			return 0;
		}
	}
	return 1;
}

/*
 * This function returns the solution of min_x ||Ax - b||
 *
 * || . || is the second order (i.e. L2) norm. This is a least squares technique that
 * is based on QR decomposition:
 * If A=Q R with Q orthogonal and R upper triangular, the normal equations become
 * (A^T A) x = A^T b  or (R^T Q^T Q R) x = A^T b or (R^T R) x = A^T b.
 * This amounts to solving R^T y = A^T b for y and then R x = y for x
 * Note that Q does not need to be explicitly computed
 *
 * A is mxn, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_QRLS( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, int n, std::vector<char>& buffer )
{
	if( m < n )
	{
		std::cerr << "Normal equations require that the number of rows is greater than number of columns in Ax_eq_b_QRLS() [" << m << " x " << n << "]! -- try transposing\n";
		exit(1);
	}

	int info;
	static int nb = 0;
	if( nb == 0 )
	{
		int auxi = -1;
		FLOATTYPE auxf;
		// workspace query; optimal size is returned in auxf
		lm_geqrf<FLOATTYPE>( &m, &m, nullptr, &m, nullptr, &auxf, &auxi, &info );
		nb = ((int)auxf)/m; // optimal worksize is m*nb
	}

	/* calculate required memory size */
	const int work_sz = nb*m;
	const int a_sz    = m*n;
	const int tau_sz  = n;
	const int r_sz    = n*n;
	const int buf_sz  = a_sz + tau_sz + r_sz + work_sz;

	if( buffer.size() < (std::size_t)buf_sz*sizeof(FLOATTYPE) )
	{
		buffer.resize( buf_sz*sizeof(FLOATTYPE) );
	}

	FLOATTYPE* a    = reinterpret_cast<FLOATTYPE*>( buffer.data() );
	FLOATTYPE* tau  = a   + a_sz;
	FLOATTYPE* r    = tau + tau_sz;
	FLOATTYPE* work = r   + r_sz;

	/* store A (column major!) into a */
	for( int i = 0; i < m; ++i )
	{
		for( int j = 0; j < n; ++j )
		{
			a[i+j*m] = A[i*n+j];
		}
	}

	/* compute A^T b in x */
	for( int i = 0; i < n; ++i )
	{
		FLOATTYPE sum_ = zero<FLOATTYPE>();
		for( int j = 0; j < m; ++j )
		{
			sum_ += A[j*n+i] * B[j];
		}
		x[i] = sum_;
	}

	/* QR decomposition of A */
	lm_geqrf( &m, &n, a, &m, tau, work, &work_sz, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_geqrf in Ax_eq_b_QRLS()\n";
			exit(1);
		}
		else{
			std::cerr << "Unknown LAPACK error " << info << " for lm_geqrf in Ax_eq_b_QRLS()\n";
			return 0;
		}
	}

	/* R is stored in the upper triangular part of a. Note that a is mxn while r nxn */
	for(int j = 0; j < n; ++j )
	{
		for( int i = 0; i <= j; ++i )
		{
			r[i+j*n] = a[i+j*m];
		}

		/* lower part is zero */
		for( int i = j+1; i < n; ++i )
		{
			r[i+j*n] = zero<FLOATTYPE>();
		}
	}

	/* solve the linear system R^T y = A^t b */
	int nrhs = 1;
	lm_trtrs( "U", "T", "N", &n, &nrhs, r, &n, x, &n, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_trtrs in Ax_eq_b_QRLS()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: the " << info << "-th diagonal element of A is zero (singular matrix) in Ax_eq_b_QRLS()\n";
			return 0;
		}
	}

	/* solve the linear system R x = y */
	lm_trtrs( "U", "N", "N", &n, &nrhs, r, &n, x, &n, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_trtrs in Ax_eq_b_QRLS()\n";
			exit(1);
		}
		else{
			std::cerr << "LAPACK error: the " << info << "-th diagonal element of A is zero (singular matrix) in Ax_eq_b_QRLS()\n";
			return 0;
		}
	}
	return 1;
}

/*
 * This function returns the solution of Ax=b
 *
 * The function assumes that A is symmetric & postive definite and employs
 * the Cholesky decomposition:
 * If A=L L^T with L lower triangular, the system to be solved becomes
 * (L L^T) x = b
 * This amounts to solving L y = b for y and then L^T x = y for x
 *
 * A is mxm, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_Chol( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )
{
	/* calculate required memory size */
	int a_sz = m*m;

	if( buffer.size() < (std::size_t)a_sz*sizeof(FLOATTYPE) )
	{
		buffer.resize( a_sz*sizeof(FLOATTYPE) );
	}

	FLOATTYPE* a = reinterpret_cast<FLOATTYPE*>( buffer.data() );

	/* store A into a and B into x. A is assumed symmetric,
	 * hence no transposition is needed
	 */
	std::memcpy( a, A, a_sz * sizeof(FLOATTYPE) );
	std::memcpy( x, B, m    * sizeof(FLOATTYPE) );

	int info;
	/* Cholesky decomposition of A */
	//lm_potf2( "L", &m, a, &m, &info );
	lm_potrf( "L", &m, a, &m, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_potf2/lm_potrf in Ax_eq_b_Chol()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: the leading minor of order " << info << " is not positive definite,\nthe factorization could not be completed for lm_potf2/lm_potrf in Ax_eq_b_Chol()\n";
			return 0;
		}
	}

	int nrhs = 1;
	/* solve using the computed Cholesky in one lapack call */
	lm_potrs( "L", &m, &nrhs, a, &m, x, &m, &info );
	if( info < 0 )
	{
		std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_potrs in Ax_eq_b_Chol()\n";
		exit(1);
	}

#if 0
	/* alternative: solve the linear system L y = b ... */
	lm_trtrs( "L", "N", "N", &m, &nrhs, a, &m, x, &m, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_trtrs in Ax_eq_b_Chol()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: the " << info << "-th diagonal element of A is zero (singular matrix) in Ax_eq_b_Chol()\n";
			return 0;
		}
	}

	/* ... solve the linear system L^T x = y */
	lm_trtrs( "L", "T", "N", &m, &nrhs, a, &m, x, &m, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_trtrs in Ax_eq_b_Chol()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: the " << info << "-th diagonal element of A is zero (singular matrix) in Ax_eq_b_Chol()\n";
			return 0;
		}
	}
#endif /* 0 */

	return 1;
}

/*
 * This function returns the solution of Ax = b
 *
 * The function employs LU decomposition:
 * If A=L U with L lower and U upper triangular, then the original system
 * amounts to solving
 * L y = b, U x = y
 *
 * A is mxm, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_LU( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )
{
	/* calculate required memory size */
	const int a_sz    = m*m;
	const int ipiv_sz = m;
	const int tot_csz = a_sz * sizeof(FLOATTYPE) + ipiv_sz * sizeof(int);

	if( buffer.size() < (std::size_t)tot_csz )
	{
		buffer.resize( tot_csz );
	}

	FLOATTYPE* a    = reinterpret_cast<FLOATTYPE*>( buffer.data() );
	int*       ipiv = reinterpret_cast<int*>      ( a + a_sz );

	/* store A (column major!) into a and B into x */
	for( int i = 0; i < m; ++i )
	{
		for( int j = 0; j < m; ++j )
		{
			a[i+j*m] = A[i*m+j];
		}
		x[i] = B[i];
	}

	int info;
	/* LU decomposition for A */
	lm_getrf<FLOATTYPE>( &m, &m, a, &m, ipiv, &info );  
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "argument " << -info << " of lm_getrf illegal in Ax_eq_b_LU()\n";
			exit(1);
		}
		else
		{
			std::cerr << "singular matrix A for lm_getrf in Ax_eq_b_LU()\n";
			return 0;
		}
	}

	int nrhs = 1;
	/* solve the system with the computed LU */
	lm_getrs<FLOATTYPE>( "N", &m, &nrhs, a, &m, ipiv, x, &m, &info );
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "argument " << -info << " of lm_getrs illegal in Ax_eq_b_LU()\n";
			exit(1);
		}
		else
		{
			std::cerr << "unknown error for lm_getrs in Ax_eq_b_LU()\n";
			return 0;
		}
	}

	return 1;
}

/*
 * This function returns the solution of Ax = b
 *
 * The function is based on SVD decomposition:
 * If A=U D V^T with U, V orthogonal and D diagonal, the linear system becomes
 * (U D V^T) x = b or x=V D^{-1} U^T b
 * Note that V D^{-1} U^T is the pseudoinverse A^+
 *
 * A is mxm, b is mx1.
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_SVD( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )
{
	int info;

	/* calculate required memory size */
#if 1 /* use optimal size */
	int work_sz = -1; // workspace query. Keep in mind that GESDD requires more memory than GESVD
	{
			FLOATTYPE aux;
			/* note that optimal work size is returned in aux */
			//lm_gesvd( "A", "A", &m, &m, nullptr, &m, nullptr, nullptr, &m, nullptr, &m, &aux, &work_sz, &info );
			lm_gesdd<FLOATTYPE>( "A", &m, &m, nullptr, &m, nullptr, nullptr, &m, nullptr, &m, &aux, &work_sz, nullptr, &info );
			work_sz = (int)aux;
	}
#else /* use minimum size */
	//int work_sz = 5*m;       // min worksize for GESVD
	int work_sz = m*(7*m+4); // min worksize for GESDD
#endif

	const int a_sz     = m*m;
	const int u_sz     = m*m;
	const int s_sz     = m;
	const int vt_sz    = m*m;
	const int iwork_sz = 8*m;
	const int buf_sz   = a_sz + u_sz + s_sz + vt_sz + work_sz;
	const int tot_csz  = buf_sz * sizeof(FLOATTYPE) + iwork_sz * sizeof(int);

	if( buffer.size() < (std::size_t)tot_csz )
	{
		buffer.resize( tot_csz );
	}

	FLOATTYPE* a     = reinterpret_cast<FLOATTYPE*>( buffer.data() );
	FLOATTYPE* u     = a + a_sz;
	FLOATTYPE* s     = u + u_sz;
	FLOATTYPE* vt    = s + s_sz;
	FLOATTYPE* work  = vt + vt_sz;
	int*       iwork = reinterpret_cast<int*>( a + buf_sz );

	/* store A (column major!) into a */
	for( int i = 0; i < m; ++i )
	{
		for( int j = 0; j < m; ++j )
		{
			a[i+j*m] = A[i*m+j];
		}
	}

	/* SVD decomposition of A */
	//lm_gesvd( "A", "A", &m, &m, a, &m, s, u, &m, vt, &m, work, &work_sz, &info );
	lm_gesdd( "A", &m, &m, a, &m, s, u, &m, vt, &m, work, &work_sz, iwork, &info );

	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_gesvd/lm_gesdd in Ax_eq_b_SVD()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: dgesdd (dbdsdc)/dgesvd (dbdsqr) failed to converge in Ax_eq_b_SVD() [info=" << info << "]\n";
			return 0;
		}
	}

	/* compute the pseudoinverse in a */
	std::fill( a, a+a_sz, zero<FLOATTYPE>() );
	FLOATTYPE thresh = s[0] * std::numeric_limits<FLOATTYPE>::epsilon();
	for( int rank = 0; rank < m && s[rank] > thresh; ++rank )
	{
		FLOATTYPE one_over_denom = one<FLOATTYPE>() / s[rank];
		for( int j = 0; j < m; ++j )
		{
			for( int i = 0; i < m; ++i )
			{
				a[i*m+j] += vt[rank+i*m] * u[j+rank*m] * one_over_denom;
			}
		}
	}

	/* compute A^+ b in x */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE sum_ = zero<FLOATTYPE>();
		for( int j = 0; j < m; ++j )
		{
			sum_ += a[i*m+j] * B[j];
		}
		x[i] = sum_;
	}

	return 1;
}

/*
 * This function returns the solution of Ax = b for a real symmetric matrix A
 *
 * The function is based on LDLT factorization with the pivoting
 * strategy of Bunch and Kaufman:
 * A is factored as L*D*L^T where L is lower triangular and
 * D symmetric and block diagonal (aka spectral decomposition,
 * Banachiewicz factorization, modified Cholesky factorization)
 *
 * A is mxm, b is mx1.
 *
 * The function returns 0 in case of error, 1 if successfull
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_BK( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )
{
	int info;
	static int nb = 0;
	if( nb == 0 )
	{
		FLOATTYPE auxf;
		int auxi = -1; // workspace query; optimal size is returned in auxf
		lm_sytrf<FLOATTYPE>( "L", &m, nullptr, &m, nullptr, &auxf, &auxi, &info );
		nb = ((int)auxf)/m; // optimal worksize is m*nb
	}

	/* calculate required memory size */
	const int a_sz    = m*m;
	const int work_sz = nb*m;
	const int buf_sz  = a_sz + work_sz;
	const int ipiv_sz = m;
	const int tot_csz = buf_sz * sizeof(FLOATTYPE) + ipiv_sz * sizeof(int);

	if( buffer.size() < (std::size_t)tot_csz )
	{
		buffer.resize( tot_csz );
	}

	FLOATTYPE* a    = reinterpret_cast<FLOATTYPE*>( buffer.data() );
	FLOATTYPE* work = a + a_sz;
	int*       ipiv = reinterpret_cast<int*>      ( a + buf_sz );

	/* store A into a and B into x; A is assumed to be symmetric, hence
	 * the column and row major order representations are the same
	 */
	std::memcpy( a, A, a_sz * sizeof(FLOATTYPE) );
	std::memcpy( x, B, m    * sizeof(FLOATTYPE) );

	/* LDLt factorization for A */
	lm_sytrf( "L", &m, a, &m, ipiv, work, &work_sz, &info );
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_sytrf in Ax_eq_b_SVD()\n";
			exit(1);
		}
		else
		{
			std::cerr << "LAPACK error: singular block diagonal matrix D for lm_sytrf in Ax_eq_b_SVD() [D(" << info << ", " << info << ") is zero]\n";
			return 0;
		}
	}

	/* solve the system with the computed factorization */
	int nrhs = 1;
	lm_sytrs( "L", &m, &nrhs, a, &m, ipiv, x, &m, &info );
	if( info < 0 )
	{
		std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_sytrs in Ax_eq_b_SVD()\n";
		exit(1);
	}

	return 1;
}

#else // no LAPACK

/*
 * This function returns the solution of Ax = b
 *
 * The function employs LU decomposition followed by forward/back substitution (see 
 * also the LAPACK-based LU solver above)
 *
 * A is mxm, b is mx1
 *
 * The function returns 0 in case of error, 1 if successful
 *
 * This function is often called repetitively to solve problems of identical
 * dimensions. To avoid repetitive malloc's and free's, allocated memory is
 * retained between calls and free'd-malloc'ed when not of the appropriate size.
 * A call with NULL as the first argument forces this memory to be released.
 */
template<typename FLOATTYPE>
int Ax_eq_b_LU_noLapack( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, std::vector<char>& buffer )
{

	/* calculate required memory size */
	const int idx_sz  = m;
	const int a_sz    = m*m;
	const int work_sz = m;
	const int buf_sz  = a_sz + work_sz;
	const int tot_csz = buf_sz * sizeof(FLOATTYPE) + idx_sz * sizeof(int);

	if( buffer.size() < (std::size_t)tot_csz )
	{
		buffer.resize( tot_csz );
	}

	FLOATTYPE* a    = reinterpret_cast<FLOATTYPE*>( buffer.data() );
	FLOATTYPE* work = a + a_sz;
	int*       idx  = reinterpret_cast<int*>( a + buf_sz );

	/* avoid destroying A, B by copying them to a, x resp. */
	std::memcpy( a, A, a_sz * sizeof(FLOATTYPE) );
	std::memcpy( x, B, m    * sizeof(FLOATTYPE) );

	auto abs_compare = [](FLOATTYPE x, FLOATTYPE y) { return std::abs(x) < std::abs(y); };

	/* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE max_ = *std::max_element( a+(i*m), a+(i*m+m), abs_compare );
		if( max_ == zero<FLOATTYPE>() )
		{
			std::cerr << "Singular matrix A in Ax_eq_b_LU_noLapack()!\n";
			return 0;
		}
		work[i] = one<FLOATTYPE>() / max_;
	}

	for( int j = 0; j < m; ++j )
	{
		for( int i = 0; i < j; ++i )
		{
			FLOATTYPE sum_ = a[i*m+j];
			for( int k = 0; k < i; ++k )
			{
				sum_ -= a[i*m+k] * a[k*m+j];
			}
			a[i*m+j] = sum_;
		}
		FLOATTYPE max_ = zero<FLOATTYPE>();
		int       maxi = -1;
		for( int i = j; i < m; ++i )
		{
			FLOATTYPE sum_ = a[i*m+j];
			for( int k = 0; k < j; ++k )
			{
				sum_ -= a[i*m+k] * a[k*m+j];
			}
			a[i*m+j] = sum_;

			FLOATTYPE aux = work[i]*std::abs(sum_);
			if( aux >= max_ )
			{
				max_ = aux;
				maxi = i;
			}
		}

		if( j != maxi )
		{
			for( int k = 0; k < m; ++k )
			{
				std::swap( a[maxi*m+k], a[j*m+k] );
			}
			work[maxi] = work[j];
		}

		idx[j] = maxi;
		if( a[j*m+j] == zero<FLOATTYPE>() )
		{
			a[j*m+j] = std::numeric_limits<FLOATTYPE>::epsilon();
		}

		if( j != m-1 )
		{
			FLOATTYPE aux = one<FLOATTYPE>() / a[j*m+j];
			for( int i = j+1; i < m; ++i )
			{
				a[i*m+j] *= aux;
			}
		}
	}

	/* The decomposition has now replaced a. Solve the linear system using
	 * forward and back substitution
	 */
	int k = 0;
	for( int i = 0; i < m; ++i )
	{
		int j = idx[i];
		FLOATTYPE sum_ = x[j];
		x[j] = x[i];
		if( k != 0 )
		{
			for( int j = k-1; j < i; ++j )
			{
				sum_ -= a[i*m+j] * x[j];
			}
		}
		else if( sum_ != zero<FLOATTYPE>() )
		{
				k = i+1;
		}
		x[i] = sum_;
	}

	for( int i = m-1; i >= 0; --i )
	{
		FLOATTYPE sum_ = x[i];
		for( int j = i+1; j < m; ++j)
		{
			sum_ -= a[i*m+j] * x[j];
		}
		x[i] = sum_ / a[i*m+i];
	}

	return 1;
}

#endif /* LMPP_HAVE_LAPACK */

#ifdef LMPP_DBL_PREC
#ifdef LMPP_HAVE_LAPACK
template int Ax_eq_b_QR<double>( double* A, double* B, double* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_QRLS<double>( double* A, double* B, double* x, int m, int n, std::vector<char>& buffer );
template int Ax_eq_b_Chol<double>( double* A, double* B, double* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_LU<double>( double* A, double* B, double* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_SVD<double>( double* A, double* B, double* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_BK<double>( double* A, double* B, double* x, int m, std::vector<char>& buffer );
#else // LMPP_HAVE_LAPACK -- No LAPACK !
template int Ax_eq_b_LU_noLapack<double>( double* A, double* B, double* x, int m, std::vector<char>& buffer )
#endif // LMPP_HAVE_LAPACK
#endif // LMPP_DBL_PREC

#ifdef LMPP_SNGL_PREC
#ifdef LMPP_HAVE_LAPACK
template int Ax_eq_b_QR<float>( float* A, float* B, float* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_QRLS<float>( float* A, float* B, float* x, int m, int n, std::vector<char>& buffer );
template int Ax_eq_b_Chol<float>( float* A, float* B, float* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_LU<float>( float* A, float* B, float* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_SVD<float>( float* A, float* B, float* x, int m, std::vector<char>& buffer );
template int Ax_eq_b_BK<float>( float* A, float* B, float* x, int m, std::vector<char>& buffer );
#else // LMPP_HAVE_LAPACK -- No LAPACK !
template int Ax_eq_b_LU_noLapack<float>( float* A, float* B, float* x, int m, std::vector<char>& buffer );
#endif // LMPP_HAVE_LAPACK
#endif // LMPP_SNGL_PREC
