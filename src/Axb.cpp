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

#include <iostream>
//#include <cstdlib>
#include <cstring>
//#include <cmath>

#include "misc.h"
#include "config_lmpp.h"
#include "external_wrappers.h"

/* Solvers for the linear systems Ax=b. Solvers should NOT modify their A & B arguments! */

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
int Ax_eq_b_QR( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m )
{
	static int nb = 0;

	if(!A) return 1; /* NOP */

	/* calculate required memory size */
	int a_sz   = m*m;
	int tau_sz = m;
	int r_sz   = m*m; /* only the upper triangular part really needed */

	int info;
	int work_sz;
	if( nb == 0 )
	{
		work_sz = -1;
		FLOATTYPE aux;
		// workspace query; optimal size is returned in aux
		lm_geqrf<FLOATTYPE>( &m, &m, NULL, &m, NULL, &aux, &work_sz, &info );
		nb = ((int)aux)/m; // optimal worksize is m*nb
	}
	work_sz = nb*m;

	std::unique_ptr<FLOATTYPE[]> buf = std::make_unique<FLOATTYPE[]>( a_sz + tau_sz + r_sz + work_sz );

	FLOATTYPE* a    = buf.get();
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
	std::memcpy( r, a, r_sz*sizeof(FLOATTYPE));

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
		FLOATTYPE sum_ = FLOATTYPE(0.0);
		for( int j = 0; j < m; ++j )
		{
			sum_ += a[i*m+j] * B[j];
		}
		x[i] = sum_;
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
int Ax_eq_b_QRLS( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m, int n )
{
	if( m < n )
	{
		std::cerr << "Normal equations require that the number of rows is greater than number of columns in Ax_eq_b_QRLS() [" << m << " x " << n << "]! -- try transposing\n";
		exit(1);
	}

	static int nb = 0;

	if(!A) return 1; /* NOP */

	/* calculate required memory size */
	int a_sz   = m*n;
	int tau_sz = n;
	int r_sz   = n*n;

	int info;
	int work_sz;
	if( nb == 0 )
	{
		work_sz = -1;
		FLOATTYPE aux;
		// workspace query; optimal size is returned in aux
		lm_geqrf<FLOATTYPE>( &m, &m, NULL, &m, NULL, &aux, &work_sz, &info );
		nb = ((int)aux)/m; // optimal worksize is m*nb
	}
	work_sz = nb*m;

	std::unique_ptr<FLOATTYPE[]> buf = std::make_unique<FLOATTYPE[]>( a_sz + tau_sz + r_sz + work_sz );

	FLOATTYPE* a    = buf.get();
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
		FLOATTYPE sum_ = FLOATTYPE(0.0);
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
			r[i+j*n] = FLOATTYPE(0.0);
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
int Ax_eq_b_Chol( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m )
{
	if(!A) return 1; /* NOP */

	/* calculate required memory size */
	int a_sz = m*m;

	std::unique_ptr<FLOATTYPE[]> buf = std::make_unique<FLOATTYPE[]>( a_sz );

	FLOATTYPE* a = buf.get();

	/* store A into a and B into x. A is assumed symmetric,
	* hence no transposition is needed
	*/
	std::memcpy( a, A, a_sz * sizeof(FLOATTYPE) );
	std::memcpy( x, B, m    * sizeof(FLOATTYPE) );

	int info;
	/* Cholesky decomposition of A */
	//POTF2("L", (int *)&m, a, (int *)&m, (int *)&info);
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
int Ax_eq_b_LU( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m )
{
	if(!A) return 1; /* NOP */

	/* calculate required memory size */
	int a_sz    = m*m;
	int ipiv_sz = m;

	std::unique_ptr<FLOATTYPE[]> buf  = std::make_unique<FLOATTYPE[]>( a_sz    );
	std::unique_ptr<int[]>       ipiv = std::make_unique<int[]>      ( ipiv_sz );

	FLOATTYPE* a = buf.get();

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
	lm_getrf( &m, &m, a, &m, ipiv.get(), &info );  
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
	lm_getrs( "N", &m, &nrhs, a, &m, ipiv.get(), x, &m, &info );
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
int Ax_eq_b_SVD( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m )
{
	if(!A) return 1; /* NOP */

	int info;
	/* calculate required memory size */
#if 1 /* use optimal size */
	int work_sz = -1; // workspace query. Keep in mind that GESDD requires more memory than GESVD
	FLOATTYPE thresh;
	/* note that optimal work size is returned in thresh */
	//lm_gesvd( "A", "A", &m, &m, NULL, &m, NULL, NULL, &m, NULL, &m, &thresh, &work_sz, &info );
	lm_gesdd( "A", &m, &m, NULL, &m, NULL, NULL, &m, NULL, &m, &thresh, &work_sz, NULL, &info );
	work_sz = (int)thresh;
#else /* use minimum size */
	//int work_sz = 5*m;       // min worksize for GESVD
	int work_sz = m*(7*m+4); // min worksize for GESDD
#endif
	int a_sz    = m*m;
	int u_sz    = m*m;
	int s_sz    = m;
	int vt_sz   = m*m;
	int iwork_sz = 8*m;

	std::unique_ptr<FLOATTYPE[]> buf   = std::make_unique<FLOATTYPE[]>( a_sz + u_sz + s_sz + vt_sz + work_sz );
	std::unique_ptr<int[]>       iwork = std::make_unique<int[]>      ( iwork_sz );

	FLOATTYPE* a    = buf.get();
	FLOATTYPE* u    = a + a_sz;
	FLOATTYPE* s    = u + u_sz;
	FLOATTYPE* vt   = s + s_sz;
	FLOATTYPE* work = vt + vt_sz;

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
	lm_gesdd( "A", &m, &m, a, &m, s, u, &m, vt, &m, work, &work_sz, iwork.get(), &info );

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
	std::fill( a, a+a_sz, FLOATTYPE(0.0));
	FLOATTYPE thresh = s[0] * std::numeric_limits<FLOATTYPE>::epsilon();
	for( int rank = 0; rank < m && s[rank] > thresh; ++rank )
	{
		FLOATTYPE one_over_denom = FLOATTYPE(1.0)/s[rank];
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
		FLOATTYPE sum_ = FLOATTYPE(0.0);
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
int Ax_eq_b_BK( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m )
{
	if(!A) return 1; /* NOP */

	static int nb = 0;

	int nrhs=1;

	if(!A) return 1; /* NOP */

	/* calculate required memory size */
	int a_sz    = m*m;
	int ipiv_sz = m;

	int info;
	int work_sz;
	if( nb == 0)
	{
		FLOATTYPE aux;
		work_sz = -1; // workspace query; optimal size is returned in aux
		lm_sytrf( "L", &m, NULL, &m, NULL, &aux, &work_sz, &info );
		nb = ((int)aux)/m; // optimal worksize is m*nb
	}
	work_sz = ( nb > 0 ? nb*m : 1 );

	std::unique_ptr<FLOATTYPE[]> buf  = std::make_unique<FLOATTYPE[]>( a_sz + work_sz );
	std::unique_ptr<int[]>       ipiv = std::make_unique<int[]>      ( ipiv_sz );

	FLOATTYPE* a    = buf.get();
	FLOATTYPE* work = a + a_sz;

	/* store A into a and B into x; A is assumed to be symmetric, hence
	 * the column and row major order representations are the same
	 */
	std::memcpy(a, A, a_sz * sizeof(FLOATTYPE));
	std::memcpy(x, B, m    * sizeof(FLOATTYPE));

	/* LDLt factorization for A */
	lm_sytrf( "L", &m, a, &m, ipiv.get(), work, &work_sz, &info );
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
	lm_sytrs( "L", &m, &nrhs, a, &m, ipiv, x, &m, &info );
	if( info < 0 )
	{
		std::cerr << "LAPACK error: illegal value for argument " << -info << " of lm_sytrs in Ax_eq_b_SVD()\n";
		exit(1);
	}

	return 1;
}

//#else // no LAPACK

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
int Ax_eq_b_LU_noLapack( FLOATTYPE* A, FLOATTYPE* B, FLOATTYPE* x, int m )
{
	if(!A) return 1; /* NOP */

	/* calculate required memory size */
	int idx_sz  = m;
	int a_sz    = m*m;
	int work_sz = m;

	std::unique_ptr<FLOATTYPE[]> buf = std::make_unique<FLOATTYPE[]>( a_sz + work_sz );
	std::unique_ptr<int[]>       idx = std::make_unique<int[]>      ( idx_sz );

	FLOATTYPE* a    = buf.get();
	FLOATTYPE* work = a + a_sz;

	/* avoid destroying A, B by copying them to a, x resp. */
	std::memcpy( a, A, a_sz * sizeof(FLOATTYPE) );
	std::memcpy( x, B, m    * sizeof(FLOATTYPE) );

	auto abs_compare = [](FLOATTYPE x, FLOATTYPE y) { return std::abs(x) < std::abs(y); };

	/* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE max_ = *std::max_element( a+(i*m), a+(i*m+m), abs_compare );
		if( max_ == 0.0 )
		{
			std::cerr << "Singular matrix A in Ax_eq_b_LU_noLapack()!\n";
			return 0;
		}
		work[i] = FLOATTYPE(1.0)/max_;
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
		FLOATTYPE max_ = FLOATTYPE(0.0);
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
		if( a[j*m+j] == FLOATTYPE(0.0) )
		{
			a[j*m+j] = std::numeric_limits<FLOATTYPE>::epsilon();
		}

		if( j != m-1 )
		{
			FLOATTYPE aux = FLOATTYPE(1.0) / a[j*m+j];
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
		else if( sum_ != 0.0 )
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

#endif /* HAVE_LAPACK */
