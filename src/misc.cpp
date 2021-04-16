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
 * Miscelaneous functions for Levenberg-Marquardt nonlinear minimization. 
 ********************************************************************************/

#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <memory>

#include "config_lmpp.h"
//#undef LMPP_HAVE_LAPACK
#include "external_wrappers.h"

#include "misc.h"

#ifdef LMPP_HAVE_MKL
#include <mkl.h>
#endif

#ifdef LMPP_HAVE_LAPACK
template<typename FLOATTYPE>
static int levmar_pseudoinverse( FLOATTYPE* A, FLOATTYPE* B, int m );
#else /* !LMPP_HAVE_LAPACK */
template<typename FLOATTYPE>
static int levmar_LUinverse_noLapack( FLOATTYPE* A, FLOATTYPE* B, int m );
#endif /* LMPP_HAVE_LAPACK */

/* blocked multiplication of the transpose of the nxm matrix a with itself (i.e. a^T a)
 * using a block size of bsize. The product is returned in b.
 * Since a^T a is symmetric, its computation can be sped up by computing only its
 * upper triangular part and copying it to the lower part.
 *
 * More details on blocking can be found at 
 * http://www-2.cs.cmu.edu/afs/cs/academic/class/15213-f02/www/R07/section_a/Recitation07-SectionA.pdf
 */
template<typename FLOATTYPE>
void levmar_trans_mat_mat_mult( FLOATTYPE* a, FLOATTYPE* b, int n, int m )
{
#ifdef LMPP_HAVE_LAPACK /* use BLAS matrix multiply */
	FLOATTYPE alpha = one<FLOATTYPE>();
	FLOATTYPE beta  = zero<FLOATTYPE>();
	/* Fool BLAS to compute a^T*a avoiding transposing a: a is equivalent to a^T in column major,
	 * therefore BLAS computes a*a^T with a and a*a^T in column major, which is equivalent to
	 * computing a^T*a in row major!
	 */
	lm_gemm<FLOATTYPE>( "N", "T", &m, &m, &n, &alpha, a, &m, a, &m, &beta, b, &m );

#else /* no LAPACK, use blocking-based multiply */

	constexpr int bsize = __BLOCKSZ__;

	/* compute upper triangular part using blocking */
	for( int jj = 0; jj < m; jj += bsize )
	{
		for( int i = 0; i < m; ++i )
		{
			FLOATTYPE* bim = b + i*m;
			for( int j = std::max(jj, i); j < std::min(jj+bsize, m); ++j )
			{
				bim[j] = zero<FLOATTYPE>(); // b[i*m+j] = 0.0;
			}
		}

		for( int kk = 0; kk < n; kk += bsize )
		{
			for( int i = 0; i < m; ++i )
			{
				FLOATTYPE* bim = b + i*m;
				for( int j = std::max(jj, i); j < std::min(jj+bsize, m); ++j )
				{
					FLOATTYPE sum = zero<FLOATTYPE>();
					for( int k = kk; k <std::min(kk+bsize, n); ++k )
					{
						FLOATTYPE* akm = a + k*m;
						sum += akm[i]*akm[j]; // a[k*m+i] * a[k*m+j];
					}
					bim[j] += sum; // b[i*m+j] += sum;
				}
			}
		}
	}

	/* copy upper triangular part to the lower one */
	for( int i = 0; i < m; ++i )
	{
		for( int j = 0; j < i; ++j )
		{
			b[i*m+j] = b[j*m+i];
		}
	}

#endif /* LMPP_HAVE_LAPACK */
}

/* forward finite difference approximation to the Jacobian of func */
template<typename FLOATTYPE>
void levmar_fdif_forw_jac_approx( FPTR<FLOATTYPE> func,	 /* function to differentiate                         */
								  FLOATTYPE*      p,     /* I: current parameter estimate, mx1                */
								  FLOATTYPE*      hx,    /* I: func evaluated at p, i.e. hx=func(p), nx1      */
								  FLOATTYPE*      hxx,   /* W/O: work array for evaluating func(p+delta), nx1 */
								  FLOATTYPE       delta, /* increment for computing the Jacobian              */
								  FLOATTYPE*      jac,   /* O: array for storing approximated Jacobian, nxm   */
								  int             m,
								  int             n,
								  void*           adata )
{
	for( int j = 0; j < m; ++j )
	{
		/* determine d=max(1E-04*|p[j]|, delta), see HZ */
		FLOATTYPE d = std::abs( (FLOATTYPE)(1E-04) * p[j] ); // force evaluation
		if( d < delta ) d = delta;

		{
			FLOATTYPE tmp = p[j];
			p[j] += d;

			(*func)( p, hxx, m, n, adata );

			p[j] = tmp; /* restore */
		}

		d = one<FLOATTYPE>() / d; /* invert so that divisions can be carried out faster as multiplications */
		for( int i = 0; i < n; ++i )
		{
			jac[i*m+j] = (hxx[i] - hx[i]) * d;
		}
	}
}

/* central finite difference approximation to the Jacobian of func */
template<typename FLOATTYPE>
void levmar_fdif_cent_jac_approx( FPTR<FLOATTYPE> func,	 /* function to differentiate                         */
								  FLOATTYPE*      p,     /* I: current parameter estimate, mx1                */
								  FLOATTYPE*      hxm,   /* W/O: work array for evaluating func(p-delta), nx1 */
								  FLOATTYPE*      hxp,   /* W/O: work array for evaluating func(p+delta), nx1 */
								  FLOATTYPE       delta, /* increment for computing the Jacobian              */
								  FLOATTYPE*      jac,   /* O: array for storing approximated Jacobian, nxm   */
								  int             m,
								  int             n,
								  void*           adata)
{
	for( int j = 0; j < m; ++j )
	{
		/* determine d=max(1E-04*|p[j]|, delta), see HZ */
		FLOATTYPE d = std::abs( FLOATTYPE(1E-04) * p[j] ); // force evaluation
		if( d < delta ) d = delta;

		{
			FLOATTYPE aux = p[j];
			p[j] -= d;
			(*func)( p, hxm, m, n, adata );

			p[j] = aux + d;
			(*func)( p, hxp, m, n, adata );
			p[j] = aux; /* restore */
		}

		d = FLOATTYPE(0.5) / d; /* invert so that divisions can be carried out faster as multiplications */
		for( int i = 0; i < n; ++i )
		{
			jac[i*m+j] = (hxp[i] - hxm[i]) * d;
		}
	}
}

/* 
 * Check the Jacobian of a n-valued nonlinear function in m variables
 * evaluated at a point p, for consistency with the function itself.
 *
 * Based on fortran77 subroutine CHKDER by
 * Burton S. Garbow, Kenneth E. Hillstrom, Jorge J. More
 * Argonne National Laboratory. MINPACK project. March 1980.
 *
 *
 * func points to a function from R^m --> R^n: Given a p in R^m it yields hx in R^n
 * jacf points to a function implementing the Jacobian of func, whose correctness
 *     is to be tested. Given a p in R^m, jacf computes into the nxm matrix j the
 *     Jacobian of func at p. Note that row i of j corresponds to the gradient of
 *     the i-th component of func, evaluated at p.
 * p is an input array of length m containing the point of evaluation.
 * m is the number of variables
 * n is the number of functions
 * adata points to possible additional data and is passed uninterpreted
 *     to func, jacf.
 * err is an array of length n. On output, err contains measures
 *     of correctness of the respective gradients. if there is
 *     no severe loss of significance, then if err[i] is 1.0 the
 *     i-th gradient is correct, while if err[i] is 0.0 the i-th
 *     gradient is incorrect. For values of err between 0.0 and 1.0,
 *     the categorization is less certain. In general, a value of
 *     err[i] greater than 0.5 indicates that the i-th gradient is
 *     probably correct, while a value of err[i] less than 0.5
 *     indicates that the i-th gradient is probably incorrect.
 *
 *
 * The function does not perform reliably if cancellation or
 * rounding errors cause a severe loss of significance in the
 * evaluation of a function. therefore, none of the components
 * of p should be unusually small (in particular, zero) or any
 * other value which may cause loss of significance.
 */
template<typename FLOATTYPE>
void levmar_chkjac_impl( FPTR<FLOATTYPE> func, FPTR<FLOATTYPE> jacf, FLOATTYPE* p, int m, int n, void* adata, FLOATTYPE* err )
{
	const std::size_t fvec_sz  = n;
	const std::size_t fjac_sz  = n * m;
	const std::size_t pp_sz    = m;
	const std::size_t fvecp_sz = n;

	std::unique_ptr<FLOATTYPE[]> buf = std::make_unique<FLOATTYPE[]>( fvec_sz + fjac_sz + pp_sz + fvecp_sz );

	FLOATTYPE* fvec  = buf.get();
	FLOATTYPE* fjac  = fvec + fvec_sz;
	FLOATTYPE* pp    = fjac + fjac_sz;
	FLOATTYPE* fvecp = pp   + pp_sz;

	/* compute fvec = func(p) */
	(*func)( p, fvec, m, n, adata );

	/* compute the Jacobian at p */
	(*jacf)( p, fjac, m, n, adata );

	FLOATTYPE eps = std::sqrt( std::numeric_limits<FLOATTYPE>::epsilon() );

	/* compute pp */
	for( int j = 0; j < m; ++j )
	{
		pp[j] = ( p[j] == zero<FLOATTYPE>() ? eps : (p[j] + eps * std::abs(p[j])) );
	}

	/* compute fvecp=func(pp) */
	(*func)( pp, fvecp, m, n, adata );

	std::fill( err, err + n, zero<FLOATTYPE>() );

	for( int j = 0; j < m; ++j )
	{
		FLOATTYPE aux = ( p[j] == zero<FLOATTYPE>() ? one<FLOATTYPE>() : std::abs(p[j]) );
		for( int i = 0; i < n; ++i )
		{
			err[i] += aux * fjac[i*m+j];
		}
	}

	const FLOATTYPE epsf   = FLOATTYPE(100.) * std::numeric_limits<FLOATTYPE>::epsilon();
	const FLOATTYPE epslog = std::log10(eps);

	for( int i = 0; i < n; ++i )
	{
		FLOATTYPE aux = one<FLOATTYPE>();

		if( fvec [i] != zero<FLOATTYPE>()                      &&
			fvecp[i] != zero<FLOATTYPE>()                      &&
			std::abs(fvecp[i]-fvec[i]) >= epsf * std::abs(fvec[i]) )
		{
			aux = eps * std::abs( (fvecp[i]-fvec[i])/eps - err[i] ) / ( std::abs(fvec[i]) + std::abs(fvecp[i]) );
		}

		if( aux >= eps )
		{
			err[i] = zero<FLOATTYPE>();
		}
		else if( aux > std::numeric_limits<FLOATTYPE>::epsilon() )
		{
			err[i] = (std::log10(aux) - epslog) / epslog;
		}
		else
		{
			err[i] = one<FLOATTYPE>();
		}
	}
}

#ifdef LMPP_HAVE_LAPACK
/*
 * This function computes the pseudoinverse of a square matrix A
 * into B using SVD. A and B can coincide
 * 
 * The function returns 0 in case of error (e.g. A is singular),
 * the rank of A if successful
 *
 * A, B are mxm
 *
 */
template<typename FLOATTYPE>
static int levmar_pseudoinverse( FLOATTYPE* A, FLOATTYPE* B, int m )
{
	FLOATTYPE thresh;

	/* calculate required memory size */
	int a_sz    = m*m;
	int u_sz    = m*m;
	int s_sz    = m;
	int vt_sz   = m*m;
	//int worksz  = 5*m;       // min worksize for GESVD
	int worksz  = m*(7*m+4); // min worksize for GESDD
	int iworksz = 8*m;       // used only for gesdd

	std::unique_ptr<FLOATTYPE[]> buf   = std::make_unique<FLOATTYPE[]>( a_sz + u_sz + s_sz + vt_sz + worksz );
	std::unique_ptr<int[]>       iwork = std::make_unique<int[]>( iworksz );

	FLOATTYPE* a    = buf.get();
	FLOATTYPE* u    =  a + a_sz;
	FLOATTYPE* s    =  u + u_sz;
	FLOATTYPE* vt   =  s + s_sz;
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
	int info;
	//lm_gesvd( "A", "A", &m, &m, a, &m, s, u, &m, vt, &m, work, &worksz, &info );
	lm_gesdd( "A", &m, &m, a, &m, s, u, &m, vt, &m, work, &worksz, iwork.get(), &info );

	/* error treatment */
	if( info != 0 )
	{
		if( info < 0 )
		{
			std::cerr << "LAPACK error: illegal value for argument " << (-info) << " of lm_gesvd/lm_gesdd in levmar_pseudoinverse()\n";
		}
		else
		{
			std::cerr << "LAPACK error: dgesdd (dbdsdc)/dgesvd (dbdsqr) failed to converge in levmar_pseudoinverse() [info=" << info << "]\n";
		}
		return 0;
	}

	FLOATTYPE eps = std::numeric_limits<FLOATTYPE>::epsilon();

	/* compute the pseudoinverse in B */
	std::fill( B, B + a_sz, zero<FLOATTYPE>() );
	int rank = 0;
	for( thresh = eps*s[0]; rank < m && s[rank] > thresh; ++rank )
	{
		FLOATTYPE one_over_denom = one<FLOATTYPE>() / s[rank];

		for( int j = 0; j < m; ++j )
		{
			for( int i = 0; i < m; ++i )
			{
				B[i*m+j] += vt[rank+i*m] * u[j+rank*m] * one_over_denom;
			}
		}
	}

	return rank;
}
#else // no LAPACK

/*
 * This function computes the inverse of A in B. A and B can coincide
 *
 * The function employs LAPACK-free LU decomposition of A to solve m linear
 * systems A*B_i=I_i, where B_i and I_i are the i-th columns of B and I.
 *
 * A and B are mxm
 *
 * The function returns 0 in case of error, 1 if successful
 *
 */
template<typename FLOATTYPE>
static int levmar_LUinverse_noLapack( FLOATTYPE* A, FLOATTYPE* B, int m )
{
	/* calculate required memory size */
	int a_sz    = m*m;
	int x_sz    = m;
	int work_sz = m;
	int idx_sz  = m;

	std::unique_ptr<FLOATTYPE[]> buf = std::make_unique<FLOATTYPE[]>( a_sz + x_sz + work_sz );
	std::unique_ptr<int[]>       idx = std::make_unique<int[]>( idx_sz );

	FLOATTYPE* a    = buf.get();
	FLOATTYPE* x    = a + a_sz;
	FLOATTYPE* work = x + x_sz;

	/* avoid destroying A by copying it to a */
	std::copy( A, A + a_sz, a );

	auto abs_compare = [](FLOATTYPE x, FLOATTYPE y) { return std::abs(x) < std::abs(y); };

	/* compute the LU decomposition of a row permutation of matrix a; the permutation itself is saved in idx[] */
	for( int i = 0; i < m; ++i )
	{
		FLOATTYPE max_ = *std::max_element( a+(i*m), a+(i*m+m), abs_compare );
		if( max_ == zero<FLOATTYPE>() )
		{
			std::cerr << "Singular matrix A in levmar_LUinverse_noLapack()!\n";
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

	/* The decomposition has now replaced a. Solve the m linear systems using
	 * forward and back substitution
	 */
	for( int l = 0; l < m; ++l )
	{
		std::fill( x, x+m, zero<FLOATTYPE>() );
		x[l] = one<FLOATTYPE>();

		int k = 0;
		for( int i = 0; i < m; ++i )
		{
			FLOATTYPE sum_ = x[idx[i]];
			x[idx[i]]      = x[i];
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

		for( int i = m-1; i >= 0; --i)
		{
			FLOATTYPE sum_ = x[i];
			for( int j = i+1; j < m; ++j )
			{
				sum_ -= a[i*m+j] * x[j];
			}
			x[i] = sum_ / a[i*m+i];
		}

		for( int i = 0; i < m; ++i )
		{
			B[i*m+l] = x[i];
		}
	}

	return 1;
}
#endif /* LMPP_HAVE_LAPACK */

/*
 * This function computes in C the covariance matrix corresponding to a least
 * squares fit. JtJ is the approximate Hessian at the solution (i.e. J^T*J, where
 * J is the Jacobian at the solution), sumsq is the sum of squared residuals
 * (i.e. goodnes of fit) at the solution, m is the number of parameters (variables)
 * and n the number of observations. JtJ can coincide with C.
 * 
 * if JtJ is of full rank, C is computed as sumsq/(n-m)*(JtJ)^-1
 * otherwise and if LAPACK is available, C=sumsq/(n-r)*(JtJ)^+
 * where r is JtJ's rank and ^+ denotes the pseudoinverse
 * The diagonal of C is made up from the estimates of the variances
 * of the estimated regression coefficients.
 * See the documentation of routine E04YCF from the NAG fortran lib
 *
 * The function returns the rank of JtJ if successful, 0 on error
 *
 * A and C are mxm
 *
 */
template<typename FLOATTYPE>
int levmar_covar( FLOATTYPE* JtJ, FLOATTYPE* C, FLOATTYPE sumsq, int m, int n )
{
#ifdef LMPP_HAVE_LAPACK
	int rnk = levmar_pseudoinverse<FLOATTYPE>( JtJ, C, m );
	if(!rnk) return 0;
#else
	#ifdef _MSC_VER
		#pragma message("LAPACK not available, LU will be used for matrix inversion when computing the covariance; this might be unstable at times")
	#else
		#warning LAPACK not available, LU will be used for matrix inversion when computing the covariance; this might be unstable at times
	#endif // _MSC_VER

	int rnk = levmar_LUinverse_noLapack<FLOATTYPE>( JtJ, C, m );
	if(!rnk) return 0;

	rnk = m; /* assume full rank */
#endif /* LMPP_HAVE_LAPACK */

	FLOATTYPE fact = sumsq / (FLOATTYPE)(n-rnk);
	for( int i = 0; i < m*m; ++i ) C[i] *= fact;

	return rnk;
}

/*  standard deviation of the best-fit parameter i.
 *  covar is the mxm covariance matrix of the best-fit parameters (see also LEVMAR_COVAR()).
 *
 *  The standard deviation is computed as \sigma_{i} = \sqrt{C_{ii}} 
 */
template<typename FLOATTYPE>
FLOATTYPE levmar_stddev_impl( FLOATTYPE* covar, int m, int i )
{
	return std::sqrt( covar[i*m+i] );
}

/* Pearson's correlation coefficient of the best-fit parameters i and j.
 * covar is the mxm covariance matrix of the best-fit parameters (see also LEVMAR_COVAR()).
 *
 * The coefficient is computed as \rho_{ij} = C_{ij} / sqrt(C_{ii} C_{jj})
 */
template<typename FLOATTYPE>
FLOATTYPE levmar_corcoef_impl( FLOATTYPE* covar, int m, int i, int j )
{
	return (FLOATTYPE)( covar[i*m+j] / std::sqrt( covar[i*m+i] * covar[j*m+j] ) );
}

/* coefficient of determination.
 * see  http://en.wikipedia.org/wiki/Coefficient_of_determination
 */
template<typename FLOATTYPE>
FLOATTYPE levmar_R2_impl( FPTR<FLOATTYPE> func, FLOATTYPE* p, FLOATTYPE* x, int m, int n, void *adata )
{
	std::unique_ptr<FLOATTYPE[]> hx = std::make_unique<FLOATTYPE[]>(n);

	/* hx = f(p) */
	(*func)( p, hx.get(), m, n, adata );

	FLOATTYPE aux = zero<FLOATTYPE>();
	for( int i = n-1; i >= 0; --i )
	{
		aux += x[i];
	}
	FLOATTYPE xavg  = aux/(FLOATTYPE)n;
	FLOATTYPE SSerr = zero<FLOATTYPE>();
	FLOATTYPE SStot = zero<FLOATTYPE>();
	for( int i = n-1; i >= 0; --i )
	{
		SSerr += (x[i] - hx[i]) * (x[i] - hx[i]);
		SStot += (x[i] -  xavg) * (x[i] -  xavg);
	}

	return one<FLOATTYPE>() - SSerr/SStot;
}

/* check box constraints for consistency */
template<typename FLOATTYPE>
int levmar_box_check( FLOATTYPE* lb, FLOATTYPE* ub, int m )
{
	if( !lb || !ub ) return 1;
	for( int i = 0; i < m; ++i )
	{
		if( lb[i] > ub[i] ) return 0;
	}
	return 1;
}

#ifdef LMPP_HAVE_LAPACK
/* compute the Cholesky decomposition of C in W, s.t. C=W^t W and W is upper triangular */
template<typename FLOATTYPE>
int levmar_chol( FLOATTYPE* C, FLOATTYPE* W, int m )
{
	/* copy weights array C to W so that LAPACK won't destroy it;
	* C is assumed symmetric, hence no transposition is needed
	*/
	for( int i = 0; i < m*m; ++i )
	{
		W[i] = C[i];
	}

	/* Cholesky decomposition */
	int info;
	lm_potf2( "L", &m, W, &m, &info );
	/* error treatment */
	if( info != 0 )
	{
		if( info < 0)
		{
			std::cerr << "LAPACK error: illegal value for argument " << (-info) << " of lm_potf2 / in levmar_chol()\n";
		}
		else{
			std::cerr << "LAPACK error: the leading minor of order " << info << " is not positive definite,\n"
					  << "and the Cholesky factorization could not be completed in levmar_chol()\n";
		}
		return LMPP_ERROR;
	}

	/* the decomposition is in the lower part of W (in column-major order!).
	* zeroing the upper part makes it lower triangular which is equivalent to
	* upper triangular in row-major order
	*/
	for( int i = 0; i < m; ++i )
	{
		for( int j = i+1; j < m; ++j )
		{
			W[i+j*m] = zero<FLOATTYPE>();
		}
	}

	return 0;
}
#endif /* LMPP_HAVE_LAPACK */

/* Compute e=x-y for two n-vectors x and y and return the squared L2 norm of e.
 * e can coincide with either x or y; x can be NULL, in which case it is assumed
 * to be equal to the zero vector.
 * Uses loop unrolling and blocking to reduce bookkeeping overhead & pipeline
 * stalls and increase instruction-level parallelism; see http://www.abarnett.demon.co.uk/tutorial.html
 */
template<typename FLOATTYPE>
FLOATTYPE levmar_L2nrmxmy( FLOATTYPE* e, const FLOATTYPE* x, FLOATTYPE* y, int n )
{
	constexpr int blocksize   = 8;
	constexpr int bpwr        = 3; /* 8=2^3 */
	register  FLOATTYPE sum0  = zero<FLOATTYPE>();
	register  FLOATTYPE sum1  = zero<FLOATTYPE>();
	register  FLOATTYPE sum2  = zero<FLOATTYPE>();
	register  FLOATTYPE sum3  = zero<FLOATTYPE>();

	/* n may not be divisible by blocksize, 
	* go as near as we can first, then tidy up.
	*/ 
	int blockn = (n>>bpwr)<<bpwr; /* (n / blocksize) * blocksize; */

	/* unroll the loop in blocks of `blocksize'; looping downwards gains some more speed */
	if(x)
	{
		int j;
		for( int i = blockn-1; i > 0; i -= blocksize )
		{
			j  = i; e[j] = x[j] - y[j]; sum0 += e[j] * e[j]; // i
			j -= 1; e[j] = x[j] - y[j]; sum1 += e[j] * e[j]; // i-1
			j -= 1; e[j] = x[j] - y[j]; sum2 += e[j] * e[j]; // i-2
			j -= 1; e[j] = x[j] - y[j]; sum3 += e[j] * e[j]; // i-3
			j -= 1; e[j] = x[j] - y[j]; sum0 += e[j] * e[j]; // i-4
			j -= 1; e[j] = x[j] - y[j]; sum1 += e[j] * e[j]; // i-5
			j -= 1; e[j] = x[j] - y[j]; sum2 += e[j] * e[j]; // i-6
			j -= 1; e[j] = x[j] - y[j]; sum3 += e[j] * e[j]; // i-7
		}

		/*
		* There may be some left to do.
		* This could be done as a simple for() loop, 
		* but a switch is faster (and more interesting) 
		*/ 
		int i = blockn;
		if( i < n )
		{
			/* Jump into the case at the place that will allow
			* us to finish off the appropriate number of items. 
			*/ 
			switch(n - i)
			{
				case 7 : e[i] = x[i] - y[i]; sum0 += e[i] * e[i]; ++i;
				case 6 : e[i] = x[i] - y[i]; sum1 += e[i] * e[i]; ++i;
				case 5 : e[i] = x[i] - y[i]; sum2 += e[i] * e[i]; ++i;
				case 4 : e[i] = x[i] - y[i]; sum3 += e[i] * e[i]; ++i;
				case 3 : e[i] = x[i] - y[i]; sum0 += e[i] * e[i]; ++i;
				case 2 : e[i] = x[i] - y[i]; sum1 += e[i] * e[i]; ++i;
				case 1 : e[i] = x[i] - y[i]; sum2 += e[i] * e[i]; //++i;
			}
		}
	}
	else
	{ /* x==0 */
		int j;
		for( int i = blockn-1; i > 0; i -= blocksize )
		{
			j  = i; e[j] = -y[j]; sum0 += e[j] * e[j]; // i
			j -= 1; e[j] = -y[j]; sum1 += e[j] * e[j]; // i-1
			j -= 1; e[j] = -y[j]; sum2 += e[j] * e[j]; // i-2
			j -= 1; e[j] = -y[j]; sum3 += e[j] * e[j]; // i-3
			j -= 1; e[j] = -y[j]; sum0 += e[j] * e[j]; // i-4
			j -= 1; e[j] = -y[j]; sum1 += e[j] * e[j]; // i-5
			j -= 1; e[j] = -y[j]; sum2 += e[j] * e[j]; // i-6
			j -= 1; e[j] = -y[j]; sum3 += e[j] * e[j]; // i-7
		}

		/*
		* There may be some left to do.
		* This could be done as a simple for() loop, 
		* but a switch is faster (and more interesting) 
		*/ 
		int i = blockn;
		if( i < n )
		{ 
			/* Jump into the case at the place that will allow
			* us to finish off the appropriate number of items. 
			*/ 
			switch(n - i)
			{ 
				case 7 : e[i] = -y[i]; sum0 += e[i] * e[i]; ++i;
				case 6 : e[i] = -y[i]; sum1 += e[i] * e[i]; ++i;
				case 5 : e[i] = -y[i]; sum2 += e[i] * e[i]; ++i;
				case 4 : e[i] = -y[i]; sum3 += e[i] * e[i]; ++i;
				case 3 : e[i] = -y[i]; sum0 += e[i] * e[i]; ++i;
				case 2 : e[i] = -y[i]; sum1 += e[i] * e[i]; ++i;
				case 1 : e[i] = -y[i]; sum2 += e[i] * e[i]; //++i;
			}
		}
	}

	return sum0+sum1+sum2+sum3;
}

#ifdef LMPP_DBL_PREC
	template void   levmar_trans_mat_mat_mult  <double>( double*, double*, int, int );
	template void   levmar_fdif_forw_jac_approx<double>( FPTR<double>, double*, double*, double*, double, double*, int, int, void* );
	template void   levmar_fdif_cent_jac_approx<double>( FPTR<double>, double*, double*, double*, double, double*, int, int, void* );
	#ifdef LMPP_HAVE_LAPACK
		template int levmar_pseudoinverse      <double>( double*, double*, int );
	#else
		template int levmar_LUinverse_noLapack <double>( double*, double*, int );
	#endif
	template int    levmar_covar      <double>( double*, double*, double, int, int );
	template int    levmar_box_check  <double>( double*, double*, int );
	template int    levmar_chol       <double>( double*, double*, int );
	template double levmar_L2nrmxmy   <double>( double*, const double*, double*, int );
	template void   levmar_chkjac_impl<double>( FPTR<double> func, FPTR<double> jacf, double* p, int m, int n, void* adata, double* err );

	double levmar_stddev( double* covar, int m, int i )
	{
		return levmar_stddev_impl<double>( covar, m, i );
	}

	double levmar_corcoef( double* covar , int m, int i, int j )
	{
		return levmar_corcoef_impl( covar, m, i, j );
	}

	double levmar_R2( FPTR<double> func, double* p, double* x, int m, int n, void* adata )
	{
		return levmar_R2_impl( func, p, x, m, n, adata );
	}
#endif

#ifdef LMPP_SNGL_PREC
	template void  levmar_trans_mat_mat_mult  <float>( float* a, float* b, int n, int m );
	template void  levmar_fdif_forw_jac_approx<float>( FPTR<float>, float*, float*, float*, float, float*, int, int, void* );
	template void  levmar_fdif_cent_jac_approx<float>( FPTR<float>, float*, float*, float*, float, float*, int, int, void* );
	#ifdef LMPP_HAVE_LAPACK
		template int levmar_pseudoinverse     <float>( float*, float*, int );
	#else
		template int levmar_LUinverse_noLapack<float>( float*, float*, int );
	#endif
	template int   levmar_covar      <float>( float*, float*, float, int, int );
	template int   levmar_box_check  <float>( float*, float*, int );
	template int   levmar_chol       <float>( float*, float*, int );
	template float levmar_L2nrmxmy   <float>( float*, const float*, float*, int );
	template void  levmar_chkjac_impl<float>( FPTR<float> func, FPTR<float> jacf, float* p, int m, int n, void* adata, float* err );

	float levmar_stddev( float* covar, int m, int i )
	{
		return levmar_stddev_impl<float>( covar, m, i );
	}

	float levmar_corcoef( float* covar , int m, int i, int j )
	{
		return levmar_corcoef_impl( covar, m, i, j );
	}

	float levmar_R2( FPTR<float> func, float* p, float* x, int m, int n, void* adata )
	{
		return levmar_R2_impl( func, p, x, m, n, adata );
	}
#endif // LMPP_SNGL_PREC

