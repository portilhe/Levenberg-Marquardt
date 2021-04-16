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
	//  Copyright (C) 2009  Manolis Lourakis (lourakis at ics forth gr)
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

/*******************************************************************************
 * Wrappers for linear inequality constrained Levenberg-Marquardt minimization.
 *******************************************************************************/

#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>

#include "levmar.h"
#include "misc.h"


#ifndef LMPP_HAVE_LAPACK

#ifdef _MSC_VER
#pragma message("Linear inequalities constrained optimization requires LAPACK and was not compiled!")
#else
#warning Linear inequalities constrained optimization requires LAPACK and was not compiled!
#endif // _MSC_VER

#else // LAPACK present

#if !defined(LMPP_DBL_PREC) && !defined(LMPP_SNGL_PREC)
#error At least one of LM_DBL_PREC, LM_SNGL_PREC should be defined!
#endif // LMPP_HAVE_LAPACK


template<typename FLOATTYPE>
struct lmbleic_data
{
	FLOATTYPE*      jac;
	int             nineqcnstr; // #inequality constraints
	FPTR<FLOATTYPE> func;
	FPTR<FLOATTYPE> jacf;
	void*           adata;
};

/* wrapper ensuring that the user-supplied function is called with the right number of variables (i.e. m) */
template<typename FLOATTYPE>
static void lmbleic_func( FLOATTYPE* pext, FLOATTYPE* hx, int mm, int n, void* adata )
{
	lmbleic_data<FLOATTYPE>* data = reinterpret_cast<lmbleic_data<FLOATTYPE>*>(adata);
	int m = mm - data->nineqcnstr;
	(*(data->func))( pext, hx, m, n, data->adata );
}

/* wrapper for computing the Jacobian at pext. The Jacobian is nxmm */
template<typename FLOATTYPE>
static void lmbleic_jacf( FLOATTYPE* pext, FLOATTYPE* jacext, int mm, int n, void *adata )
{
	lmbleic_data<FLOATTYPE>* data = reinterpret_cast<lmbleic_data<FLOATTYPE>*>(adata);
	int m = mm - data->nineqcnstr;

	FLOATTYPE* jac = data->jac;
	(*(data->jacf))( pext, jac, m, n, data->adata );

	for( int i = 0; i < n; ++i )
	{
		FLOATTYPE* jacextimm = jacext + i*mm;
		FLOATTYPE* jacim     = jac    + i*m;
		std::copy( jacim, jacim + m, jacextimm );                      // jacext[i*mm+j] = jac[i*m+j];
		std::fill( jacextimm + m, jacextimm + mm, zero<FLOATTYPE>() ); // jacext[i*mm+j] = 0.0;
	}
}


/*
 * This function is similar to levmar_der except that the minimization is
 * performed subject to the box constraints lb[i] <= p[i] <= ub[i], the linear
 * equation constraints A*p = b, A being k1xm, b k1x1, and the linear inequality
 * constraints C*p >= d, C being k2xm, d k2x1. 
 *
 * The inequalities are converted to equations by introducing surplus variables,
 * i.e. c^T*p >= d becomes c^T*p - y = d, with y >= 0. To transform all inequalities
 * to equations, a total of k2 surplus variables are introduced; a problem with only
 * box and linear constraints results then and is solved with levmar_blec_der()
 * Note that opposite direction inequalities should be converted to the desired
 * direction by negating, i.e. c^T*p <= d becomes -c^T*p >= -d
 *
 * This function requires an analytic Jacobian. In case the latter is unavailable,
 * use levmar_bleic_dif() bellow
 *
 */
template<typename FLOATTYPE>
int levmar_bleic_der_impl( FPTR<FLOATTYPE> func,    /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						   FPTR<FLOATTYPE> jacf,    /* function to evaluate the Jacobian \part x / \part p */ 
						   FLOATTYPE*      p,       /* I/O: initial parameter estimates. On output has the estimated solution */
						   FLOATTYPE*      x,       /* I: measurement vector. NULL implies a zero vector */
						   int             m,       /* I: parameter vector dimension (i.e. #unknowns) */
						   int             n,       /* I: measurement vector dimension */
						   FLOATTYPE*      lb,      /* I: vector of lower bounds. If NULL, no lower bounds apply */
						   FLOATTYPE*      ub,      /* I: vector of upper bounds. If NULL, no upper bounds apply */
						   FLOATTYPE*      A,       /* I: equality constraints matrix, k1xm. If NULL, no linear equation constraints apply */
						   FLOATTYPE*      b,       /* I: right hand constraints vector, k1x1 */
						   int             k1,      /* I: number of constraints (i.e. A's #rows) */
						   FLOATTYPE*      C,       /* I: inequality constraints matrix, k2xm */
						   FLOATTYPE*      d,       /* I: right hand constraints vector, k2x1 */
						   int             k2,      /* I: number of inequality constraints (i.e. C's #rows) */
						   int             itmax,   /* I: maximum number of iterations */
						   FLOATTYPE       opts[4], /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
						                             * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
						                             */
						   FLOATTYPE info[LMPP_INFO_SZ],
						                            /* O: information regarding the minimization. Set to NULL if don't care
						                             * info[0]= ||e||_2 at initial p.
						                             * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
						                             * info[5]= # iterations,
						                             * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
						                             *                                 2 - stopped by small Dp
						                             *                                 3 - stopped by itmax
						                             *                                 4 - singular matrix. Restart from current p with increased mu 
						                             *                                 5 - no further error reduction is possible. Restart with increased mu
						                             *                                 6 - stopped by small ||e||_2
						                             *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
						                             * info[7]= # function evaluations
						                             * info[8]= # Jacobian evaluations
						                             * info[9]= # linear systems solved, i.e. # attempts for reducing error
						                             */
						   FLOATTYPE*      work,    /* working memory at least LM_BLEIC_DER_WORKSZ() reals large, allocated if NULL */
						   FLOATTYPE*      covar,   /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						   void*           adata )  /* pointer to possibly additional data, passed uninterpreted to func & jacf.
						                             * Set to NULL if not needed
						                             */
{
	if( !jacf )
	{
		std::cerr << "No function specified for computing the Jacobian in levmar_bleic_der()\n";
		std::cerr << "If no such function is available, use levmar_bleic_dif() rather than levmar_bleic_der()\n";
		return LMPP_ERROR;
	}

	if( !A || !b ) k1 = 0; // sanity check

	if( !C || !d )
	{
		std::cerr << "levmar_bleic_der(): missing inequality constraints, use levmar_blec_der() in this case!\n";
		return LMPP_ERROR;
	}

	if( n < m-k1 )
	{
		std::cerr << "levmar_bleic_der(): cannot solve a problem with fewer measurements + equality constraints ["
				  << n << " + " << k1 << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	lmbleic_data<FLOATTYPE> data;

	const int mm  = m  + k2;
	const int k12 = k1 + k2;
	std::vector<FLOATTYPE> buf_v;
	buf_v.resize( 3*mm + k12*mm + k12 + n*m + (covar ? mm*mm : 0) );

	FLOATTYPE* pext   = buf_v.data();     /* corresponding to  p for the full set of variables; pext=[p, surplus], pext is mm */
	FLOATTYPE* lbext  = pext  + mm;       /* corresponding to lb for the full set of variables */
	FLOATTYPE* ubext  = lbext + mm;       /* corresponding to ub for the full set of variables */
	FLOATTYPE* Aext   = ubext + mm;       /* corresponding to  A for the full set of variables; Aext is (k1+k2)xmm */
	FLOATTYPE* bext   = Aext  + k12 * mm; /* corresponding to  b for the full set of variables; bext (k1+k2) */
	data.jac          = bext  + k12;
	FLOATTYPE* covext = covar ? data.jac + n*m : nullptr; /* corresponding to covar for the full set of variables; covext is mmxmm */
	data.nineqcnstr   = k2;
	data.func         = func;
	data.jacf         = jacf;
	data.adata        = adata;

	/* compute y s.t. C*p - y=d, i.e. y=C*p-d.
	 * y is stored in the last k2 elements of pext
	 */
	for( int i = 0; i < k2; ++i )
	{
		FLOATTYPE sum_ = zero<FLOATTYPE>();
		for( int j = 0; j < m; ++j )
		{
			sum_ += C[i*m+j] * p[j];
		}
		pext[i+m] = sum_ - d[i];

		/* surplus variables must be >=0 */
		lbext[i+m] = zero<FLOATTYPE>();
		ubext[i+m] = std::numeric_limits<FLOATTYPE>::max();
	}

	/* set the first m elements of pext equal to p */
	for( int i = 0; i < m; ++i )
	{
		pext[i]  = p[i];
		lbext[i] = lb ? lb[i] : std::numeric_limits<FLOATTYPE>::min();
		ubext[i] = ub ? ub[i] : std::numeric_limits<FLOATTYPE>::max();
	}

	/* setup the constraints matrix */
	/* original linear equation constraints */
	for( int i = 0; i < k1; ++i )
	{
		std::copy( A + i*m, A + i*m + m, Aext + i*m );                   // Aext[i*mm+j] = A[i*m+j];
		std::fill( Aext + i*mm + m, Aext + i*mm + mm, zero<FLOATTYPE>() ); // Aext[i*mm+j] = 0.0;
		bext[i] = b[i];
	}

	/* linear equation constraints resulting from surplus variables */
	for( int i = 0; i < k2; ++i )
	{
		const int ii = i + k1;
		std::copy( C + i*m, C + i*m + m, Aext + ii*mm );                     // Aext[ii*mm+j] = C[i*m+j];
		std::fill( Aext + ii*mm + m, Aext + ii*mm + mm, zero<FLOATTYPE>() ); // Aext[ii*mm+j] = 0.0;
		Aext[ii*mm+m+i] = FLOATTYPE(-1.0);
		bext[ii]        = d[i];
	}

	FLOATTYPE locinfo[LMPP_INFO_SZ];
	if( !info )
		info = locinfo; /* make sure that levmar_blec_der() is called with non-null info */

	/* note that the default weights for the penalty terms are being used below */
	int ret = levmar_blec_der( lmbleic_func<FLOATTYPE>, lmbleic_jacf<FLOATTYPE>, pext, x, mm, n, lbext, ubext,
							   Aext, bext, k12, nullptr, itmax, opts, info, work, covext, (void *)&data );

	/* copy back the minimizer */
	std::copy( pext, pext + m, p );

#if 0
	std::cout << "Surplus variables for the minimizer:\n";
	for( int i = m; i < mm; ++i )
		std::cout << pext[i];
	std::cout << "\n\n";
#endif

	if( covar )
	{
		for( int i = 0; i < m; ++i )
		{
			for( int j = 0; j < m; ++j )
			{
				covar[i*m+j] = covext[i*mm+j];
			}
		}
	}

	return ret;
}


/* Similar to the LEVMAR_BLEIC_DER() function above, except that the Jacobian is approximated
* with the aid of finite differences (forward or central, see the comment for the opts argument)
*/
template<typename FLOATTYPE>
int levmar_bleic_dif_impl( FPTR<FLOATTYPE> func,    /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						   FLOATTYPE*      p,       /* I/O: initial parameter estimates. On output has the estimated solution */
						   FLOATTYPE*      x,       /* I: measurement vector. NULL implies a zero vector */
						   int             m,       /* I: parameter vector dimension (i.e. #unknowns) */
						   int             n,       /* I: measurement vector dimension */
						   FLOATTYPE*      lb,      /* I: vector of lower bounds. If NULL, no lower bounds apply */
						   FLOATTYPE*      ub,      /* I: vector of upper bounds. If NULL, no upper bounds apply */
						   FLOATTYPE*      A,       /* I: equality constraints matrix, k1xm. If NULL, no linear equation constraints apply */
						   FLOATTYPE*      b,       /* I: right hand constraints vector, k1x1 */
						   int             k1,      /* I: number of constraints (i.e. A's #rows) */
						   FLOATTYPE*      C,       /* I: inequality constraints matrix, k2xm */
						   FLOATTYPE*      d,       /* I: right hand constraints vector, k2x1 */
						   int             k2,      /* I: number of inequality constraints (i.e. C's #rows) */
						   int             itmax,   /* I: maximum number of iterations */
						   FLOATTYPE       opts[5], /* I: opts[0-3] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
						                             * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
						                             * the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
						                             * If \delta<0, the Jacobian is approximated with central differences which are more accurate
						                             * (but slower!) compared to the forward differences employed by default. 
						                             */
						   FLOATTYPE info[LMPP_INFO_SZ],
						                            /* O: information regarding the minimization. Set to NULL if don't care
						                             * info[0]= ||e||_2 at initial p.
						                             * info[1-4]=[ ||e||_2, ||J^T e||_inf,  ||Dp||_2, mu/max[J^T J]_ii ], all computed at estimated p.
						                             * info[5]= # iterations,
						                             * info[6]=reason for terminating: 1 - stopped by small gradient J^T e
						                             *                                 2 - stopped by small Dp
						                             *                                 3 - stopped by itmax
						                             *                                 4 - singular matrix. Restart from current p with increased mu 
						                             *                                 5 - no further error reduction is possible. Restart with increased mu
						                             *                                 6 - stopped by small ||e||_2
						                             *                                 7 - stopped by invalid (i.e. NaN or Inf) "func" values. This is a user error
						                             * info[7]= # function evaluations
						                             * info[8]= # Jacobian evaluations
						                             * info[9]= # linear systems solved, i.e. # attempts for reducing error
						                             */
						   FLOATTYPE*      work,    /* working memory at least LM_BLEIC_DER_WORKSZ() reals large, allocated if NULL */
						   FLOATTYPE*      covar,   /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						   void*           adata )  /* pointer to possibly additional data, passed uninterpreted to func & jacf.
													 * Set to NULL if not needed
													 */
{
	if( !A || !b ) k1 = 0; // sanity check

	if( !C || !d )
	{
		std::cerr << "levmar_bleic_dif(): missing inequality constraints, use levmar_blec_dif() in this case!\n";
		return LMPP_ERROR;
	}

	if( n < m-k1 )
	{
		std::cerr << "levmar_bleic_dif(): cannot solve a problem with fewer measurements + equality constraints ["
			<< n << " + " << k1 << "] than unknowns [" << m << "]\n";
		return LMPP_ERROR;
	}

	lmbleic_data<FLOATTYPE> data;

	const int mm  = m  + k2;
	const int k12 = k1 + k2;
	std::vector<FLOATTYPE> buf_v;
	buf_v.resize( 3*mm + k12*mm + k12 + (covar ? mm*mm : 0) );

	FLOATTYPE* pext   = buf_v.data();     /* corresponding to  p for the full set of variables; pext=[p, surplus], pext is mm */
	FLOATTYPE* lbext  = pext  + mm;       /* corresponding to lb for the full set of variables */
	FLOATTYPE* ubext  = lbext + mm;       /* corresponding to ub for the full set of variables */
	FLOATTYPE* Aext   = ubext + mm;       /* corresponding to  A for the full set of variables; Aext is (k1+k2)xmm */
	FLOATTYPE* bext   = Aext  + k12 * mm; /* corresponding to  b for the full set of variables; bext (k1+k2) */
	data.jac          = nullptr;
	FLOATTYPE* covext = covar ? bext + k12 : nullptr; /* corresponding to covar for the full set of variables; covext is mmxmm */
	data.nineqcnstr   = k2;
	data.func         = func;
	data.jacf         = nullptr;
	data.adata        = adata;

	/* compute y s.t. C*p - y=d, i.e. y=C*p-d.
	* y is stored in the last k2 elements of pext
	*/
	for( int i = 0; i < k2; ++i )
	{
		FLOATTYPE sum_ = zero<FLOATTYPE>();
		for( int j = 0; j < m; ++j )
		{
			sum_ += C[i*m+j] * p[j];
		}
		pext[i+m] = sum_ - d[i];

		/* surplus variables must be >=0 */
		lbext[i+m] = zero<FLOATTYPE>();
		ubext[i+m] = std::numeric_limits<FLOATTYPE>::max();
	}

	/* set the first m elements of pext equal to p */
	for( int i = 0; i < m; ++i )
	{
		pext[i]  = p[i];
		lbext[i] = lb ? lb[i] : std::numeric_limits<FLOATTYPE>::min();
		ubext[i] = ub ? ub[i] : std::numeric_limits<FLOATTYPE>::max();
	}

	/* setup the constraints matrix */
	/* original linear equation constraints */
	for( int i = 0; i < k1; ++i )
	{
		std::copy( A + i*mm, A + i*mm + m, Aext + i*m );                   // Aext[i*mm+j] = A[i*m+j];
		std::fill( Aext + i*mm + m, Aext + i*mm + mm, zero<FLOATTYPE>() ); // Aext[i*mm+j] = 0.0;
		bext[i] = b[i];
	}

	/* linear equation constraints resulting from surplus variables */
	for( int i = 0; i < k2; ++i )
	{
		const int ii = i + k1;
		std::copy( C + i*m, C + i*m + m, Aext + ii*mm );                     // Aext[ii*mm+j] = C[i*m+j];
		std::fill( Aext + ii*mm + m, Aext + ii*mm + mm, zero<FLOATTYPE>() ); // Aext[ii*mm+j] = 0.0;
		Aext[ii*mm+m+i] = FLOATTYPE(-1.0);
		bext[ii]        = d[i];
	}

	FLOATTYPE locinfo[LMPP_INFO_SZ];
	if( !info )
		info = locinfo; /* make sure that levmar_blec_dif() is called with non-null info */

	/* note that the default weights for the penalty terms are being used below */
	int ret = levmar_blec_dif( lmbleic_func<FLOATTYPE>, pext, x, mm, n, lbext, ubext, Aext, bext, k12,
							   nullptr, itmax, opts, info, work, covext, (void *)&data );

	/* copy back the minimizer */
	std::copy( pext, pext + m, p );

#if 0
	std::cout << "Surplus variables for the minimizer:\n";
	for( int i = m; i < mm; ++i )
		std::cout << pext[i];
	std::cout << "\n\n";
#endif

	if( covar )
	{
		for( int i = 0; i < m; ++i )
		{
			for( int j = 0; j < m; ++j )
			{
				covar[i*m+j] = covext[i*mm+j];
			}
		}
	}

	return ret;
}


#ifdef LMPP_DBL_PREC
int levmar_bleic_der( FPTR<double> func,
					  FPTR<double> jacf,
					  double*      p,
					  double*      x,
					  int          m,
					  int          n,
					  double*      lb,
					  double*      ub,
					  double*      A,
					  double*      b,
					  int          k1,
					  double*      C,
					  double*      d,
					  int          k2,
					  int          itmax,
					  double*      opts,
					  double*      info,
					  double*      work,
					  double*      covar,
					  void*        adata )
{
	return levmar_bleic_der_impl<double>( func, jacf, p, x, m, n, lb, ub, A, b, k1,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

int levmar_bleic_dif( FPTR<double> func,
					  double*      p,
					  double*      x,
					  int          m,
					  int          n,
					  double*      lb,
					  double*      ub,
					  double*      A,
					  double*      b,
					  int          k1,
					  double*      C,
					  double*      d,
					  int          k2,
					  int          itmax,
					  double*      opts,
					  double*      info,
					  double*      work,
					  double*      covar,
					  void*        adata )
{
	return levmar_bleic_dif_impl<double>( func, p, x, m, n, lb, ub, A, b, k1, C, d, k2,
										  itmax, opts, info, work, covar, adata );
}

/* convenience wrappers to levmar_bleic_der/levmar_bleic_dif */

/* box & linear inequality constraints */
int levmar_blic_der( FPTR<double> func,
					 FPTR<double> jacf,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[4],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata )
{
	return levmar_bleic_der_impl<double>( func, jacf, p, x, m, n, lb, ub, nullptr, nullptr, 0,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

int levmar_blic_dif( FPTR<double> func,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[5],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata )
{
	return levmar_bleic_dif_impl<double>( func, p, x, m, n, lb, ub, nullptr, nullptr, 0,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

/* linear equation & inequality constraints */
int levmar_leic_der( FPTR<double> func,
					 FPTR<double> jacf,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      A,
					 double*      b,
					 int          k1,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[4],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata )
{
	return levmar_bleic_der_impl<double>( func, jacf, p, x, m, n, nullptr, nullptr, A, b, k1,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

template<typename FLOATTYPE>
int levmar_leic_dif( FPTR<double> func,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      A,
					 double*      b,
					 int          k1,
					 double*      C,
					 double*      d,
					 int          k2,
					 int          itmax,
					 double       opts[5],
					 double       info[LMPP_INFO_SZ],
					 double*      work,
					 double*      covar,
					 void*        adata )
{
	return levmar_bleic_dif_impl<double>( func, p, x, m, n, nullptr, nullptr, A, b, k1,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

/* linear inequality constraints */
int levmar_lic_der( FPTR<double> func,
					FPTR<double> jacf,
					double*      p,
					double*      x,
					int          m,
					int          n,
					double*      C,
					double*      d,
					int          k2,
					int          itmax,
					double       opts[4],
					double       info[LMPP_INFO_SZ],
					double*      work,
					double*      covar,
					void*        adata )
{
	return levmar_bleic_der_impl<double>( func, jacf, p, x, m, n, nullptr, nullptr, nullptr, nullptr, 0,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

int levmar_lic_dif( FPTR<double> func,
					double*      p,
					double*      x,
					int          m,
					int          n,
					double*      C,
					double*      d,
					int          k2,
					int          itmax,
					double       opts[4],
					double       info[LMPP_INFO_SZ],
					double*      work,
					double*      covar,
					void*        adata )
{
	return levmar_bleic_dif_impl<double>( func, p, x, m, n, nullptr, nullptr, nullptr, nullptr, 0,
										  C, d, k2, itmax, opts, info, work, covar, adata );
}

#endif /* LMPP_DBL_PREC */

#ifdef LMPP_SNGL_PREC
int levmar_bleic_der( FPTR<float> func,
					  FPTR<float> jacf,
					  float*      p,
					  float*      x,
					  int         m,
					  int         n,
					  float*      lb,
					  float*      ub,
					  float*      A,
					  float*      b,
					  int         k1,
					  float*      C,
					  float*      d,
					  int         k2,
					  int         itmax,
					  float*      opts,
					  float*      info,
					  float*      work,
					  float*      covar,
					  void*       adata )
{
	return levmar_bleic_der_impl<float>( func, jacf, p, x, m, n, lb, ub, A, b, k1,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

int levmar_bleic_dif( FPTR<float> func,
					  float*      p,
					  float*      x,
					  int         m,
					  int         n,
					  float*      lb,
					  float*      ub,
					  float*      A,
					  float*      b,
					  int         k1,
					  float*      C,
					  float*      d,
					  int         k2,
					  int         itmax,
					  float*      opts,
					  float*      info,
					  float*      work,
					  float*      covar,
					  void*       adata )
{
	return levmar_bleic_dif_impl<float>( func, p, x, m, n, lb, ub, A, b, k1, C, d, k2,
										 itmax, opts, info, work, covar, adata );
}

/* convenience wrappers to levmar_bleic_der/levmar_bleic_dif */

/* box & linear inequality constraints */
int levmar_blic_der( FPTR<float> func,
					 FPTR<float> jacf,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[4],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata )
{
	return levmar_bleic_der_impl<float>( func, jacf, p, x, m, n, lb, ub, nullptr, nullptr, 0,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

int levmar_blic_dif( FPTR<float> func,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[5],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata )
{
	return levmar_bleic_dif_impl<float>( func, p, x, m, n, lb, ub, nullptr, nullptr, 0,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

/* linear equation & inequality constraints */
int levmar_leic_der( FPTR<float> func,
					 FPTR<float> jacf,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      A,
					 float*      b,
					 int         k1,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[4],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata )
{
	return levmar_bleic_der_impl<float>( func, jacf, p, x, m, n, nullptr, nullptr, A, b, k1,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

template<typename FLOATTYPE>
int levmar_leic_dif( FPTR<float> func,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      A,
					 float*      b,
					 int         k1,
					 float*      C,
					 float*      d,
					 int         k2,
					 int         itmax,
					 float       opts[5],
					 float       info[LMPP_INFO_SZ],
					 float*      work,
					 float*      covar,
					 void*       adata )
{
	return levmar_bleic_dif_impl<float>( func, p, x, m, n, nullptr, nullptr, A, b, k1,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

/* linear inequality constraints */
int levmar_lic_der( FPTR<float> func,
					FPTR<float> jacf,
					float*      p,
					float*      x,
					int          m,
					int          n,
					float*      C,
					float*      d,
					int          k2,
					int          itmax,
					float       opts[4],
					float       info[LMPP_INFO_SZ],
					float*      work,
					float*      covar,
					void*        adata )
{
	return levmar_bleic_der_impl<float>( func, jacf, p, x, m, n, nullptr, nullptr, nullptr, nullptr, 0,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

int levmar_lic_dif( FPTR<float> func,
					float*      p,
					float*      x,
					int          m,
					int          n,
					float*      C,
					float*      d,
					int          k2,
					int          itmax,
					float       opts[4],
					float       info[LMPP_INFO_SZ],
					float*      work,
					float*      covar,
					void*        adata )
{
	return levmar_bleic_dif_impl<float>( func, p, x, m, n, nullptr, nullptr, nullptr, nullptr, 0,
										 C, d, k2, itmax, opts, info, work, covar, adata );
}

#endif /* LMPP_SNGL_PREC */

#endif /* LMPP_HAVE_LAPACK */

