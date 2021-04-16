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
	//  Copyright (C) 2004-06  Manolis Lourakis (lourakis at ics forth gr)
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
 * This file implements combined box and linear equation constraints.
 *
 * Note that the algorithm implementing linearly constrained minimization does
 * so by a change in parameters that transforms the original program into an
 * unconstrained one. To employ the same idea for implementing box & linear
 * constraints would require the transformation of box constraints on the
 * original parameters to box constraints for the new parameter set. This
 * being impossible, a different approach is used here for finding the minimum.
 * The trick is to remove the box constraints by augmenting the function to
 * be fitted with penalty terms and then solve the resulting problem (which
 * involves linear constrains only) with the functions in lmlec.c
 *
 * More specifically, for the constraint a <= x[i] <= b to hold, the term C[i]=
 * (2*x[i]-(a+b))/(b-a) should be within [-1, 1]. This is enforced by adding
 * the penalty term w[i]*max((C[i])^2-1, 0) to the objective function, where
 * w[i] is a large weight. In the case of constraints of the form a <= x[i],
 * the term C[i] = a-x[i] has to be non positive, thus the penalty term is
 * w[i]*max(C[i], 0). If x[i] <= b, C[i] = x[i]-b has to be non negative and
 * the penalty is w[i]*max(C[i], 0). The derivatives needed for the Jacobian
 * are as follows:
 * For the constraint a <= x[i] <= b:
 *     4*(2*x[i]-(a+b))/(b-a)^2  if x[i] not in [a, b],
 *     0                         otherwise
 * For the constraint a <= x[i]:
 *    -1  if x[i] <= a,
 *     0  otherwise
 * For the constraint x[i] <= b:
 *     1  if b <= x[i],
 *     0  otherwise
 *
 * Note that for the above to work, the weights w[i] should be large enough;
 * depending on your minimization problem, the default values might need some
 * tweaking (see arg "wghts" below).
 *******************************************************************************/

#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>

#include "levmar.h"
#include "misc.h"

#ifndef LMPP_HAVE_LAPACK

#ifdef _MSC_VER
#pragma message("Combined box and linearly constrained optimization requires LAPACK and was not compiled!")
#else
#warning Combined box and linearly constrained optimization requires LAPACK and was not compiled!
#endif // _MSC_VER

#else // LAPACK present


#define LMPP_BC_WEIGHT  1E+04

enum class BCType
{
	Interval,
	Low,
	High,
};

template<typename FLOATTYPE>
struct lmblec_data {
	std::vector<FLOATTYPE> x;
	FLOATTYPE*             lb;
	FLOATTYPE*             ub;
	std::vector<FLOATTYPE> w;
	std::vector<BCType>    bctype;
	FPTR<FLOATTYPE>        func;
	FPTR<FLOATTYPE>        jacf;
	void*                  adata;
};


/* augmented measurements */
template<typename FLOATTYPE>
static void lmblec_func( FLOATTYPE* p, FLOATTYPE* hx, int m, int n, void* adata )
{
	lmblec_data<FLOATTYPE> *data = reinterpret_cast<lmblec_data<FLOATTYPE>*>(adata);

	int         nn  = n - m;
	FLOATTYPE*  lb  = data->lb;
	FLOATTYPE*  ub  = data->ub;
	auto const& w   = data->w;
	auto const& typ = data->bctype;

	(*(data->func))( p, hx, m, nn, data->adata );

	for( int i = nn; i < n; ++i )
	{
		const int j = i - nn;
		switch( typ[j] )
		{
			case BCType::Interval:
				{
					FLOATTYPE aux = ( two<FLOATTYPE>()*p[j] - (lb[j]+ub[j]) ) / (ub[j]-lb[j]);
					hx[i]         = w[j] * std::max<FLOATTYPE>( sq(aux) - one<FLOATTYPE>(), zero<FLOATTYPE>() );
				}
				break;

			case BCType::Low:
				hx[i] = w[j] * std::max( lb[j] - p[j], zero<FLOATTYPE>() );
				break;

			case BCType::High:
				hx[i] = w[j] * std::max( p[j] - ub[j], zero<FLOATTYPE>() );
				break;
		}
	}
}


/* augmented Jacobian */
template<typename FLOATTYPE>
static void lmblec_jacf( FLOATTYPE* p, FLOATTYPE* jac, int m, int n, void *adata )
{
	lmblec_data<FLOATTYPE> *data = reinterpret_cast<lmblec_data<FLOATTYPE>*>(adata);

	int         nn  = n - m;
	FLOATTYPE*  lb  = data->lb;
	FLOATTYPE*  ub  = data->ub;
	auto const& w   = data->w;
	auto const& typ = data->bctype;
	(*(data->jacf))( p, jac, m, nn, data->adata );

	/* clear all extra rows */
	std::fill( jac + nn*m, jac + n*m, zero<FLOATTYPE>() );

	for( int i = nn; i < n; ++i )
	{
		const int j = i - nn;
		switch( typ[j] )
		{
			case BCType::Interval:
				if( lb[j] <= p[j] && p[j] <= ub[j])
					continue; // corresp. jac element already 0
				/* out of interval */
				{
					FLOATTYPE aux = ub[j] - lb[j];
					aux           = FLOATTYPE(4.0) * ( two<FLOATTYPE>() * p[j] - (lb[j]+ub[j]) ) / sq(aux);
					jac[i*m+j]    = w[j] * aux;
				}
				break;

			case BCType::Low: // lb[j] <= p[j] ? 0.0 : -1.0;
				if( lb[j] <= p[j] )
					continue; // corresp. jac element already 0
				/* smaller than lower bound */
				jac[i*m+j] = -w[j];
				break;

			case BCType::High: // p[j] <= ub[j] ? 0.0 : 1.0;
				if( p[j] <= ub[j] )
					continue; // corresp. jac element already 0
				/* greater than upper bound */
				jac[i*m+j]=w[j];
				break;
		}
	}
}

/* 
 * This function seeks the parameter vector p that best describes the measurements
 * vector x under box & linear constraints.
 * More precisely, given a vector function  func : R^m --> R^n with n>=m,
 * it finds p s.t. func(p) ~= x, i.e. the squared second order (i.e. L2) norm of
 * e=x-func(p) is minimized under the constraints lb[i]<=p[i]<=ub[i] and A p=b;
 * A is kxm, b kx1. Note that this function DOES NOT check the satisfiability of
 * the specified box and linear equation constraints.
 * If no lower bound constraint applies for p[i], use -DBL_MAX/-FLT_MAX for lb[i];
 * If no upper bound constraint applies for p[i], use DBL_MAX/FLT_MAX for ub[i].
 *
 * This function requires an analytic Jacobian. In case the latter is unavailable,
 * use LEVMAR_BLEC_DIF() bellow
 *
 * Returns the number of iterations (>=0) if successful, LM_ERROR if failed
 *
 * For more details on the algorithm implemented by this function, please refer to
 * the comments in the top of this file.
 *
 */
template<typename FLOATTYPE>
int levmar_blec_der_impl( FPTR<FLOATTYPE> func,    /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						  FPTR<FLOATTYPE> jacf,    /* function to evaluate the Jacobian \part x / \part p                                */
						  FLOATTYPE*      p,       /* I/O: initial parameter estimates. On output has the estimated solution             */
						  FLOATTYPE*      x,       /* I: measurement vector. NULL implies a zero vector                                  */
						  int             m,       /* I: parameter vector dimension (i.e. #unknowns)                                     */
						  int             n,       /* I: measurement vector dimension                                                    */
						  FLOATTYPE*      lb,      /* I: vector of lower bounds. If NULL, no lower bounds apply                          */
						  FLOATTYPE*      ub,      /* I: vector of upper bounds. If NULL, no upper bounds apply                          */
						  FLOATTYPE*      A,       /* I: constraints matrix, kxm                                                         */
						  FLOATTYPE*      b,       /* I: right hand constraints vector, kx1                                              */
						  int             k,       /* I: number of constraints (i.e. A's #rows)                                          */
						  FLOATTYPE*      wghts,   /* mx1 weights for penalty terms, defaults used if NULL                               */
						  int             itmax,   /* I: maximum number of iterations                                                    */
						  FLOATTYPE       opts[4], /* I: minim. options [\mu, \epsilon1, \epsilon2, \epsilon3]. Respectively the scale factor for initial \mu,
						                            * stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2. Set to NULL for defaults to be used
						                            */
						  FLOATTYPE       info[LMPP_INFO_SZ],
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
						  FLOATTYPE*      work,    /* working memory at least LM_BLEC_DER_WORKSZ() reals large, allocated if NULL */
						  FLOATTYPE*      covar,   /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						  void*           adata )  /* pointer to possibly additional data, passed uninterpreted to func & jacf.
						                            * Set to NULL if not needed
						                            */
{
	if( !jacf )
	{
		std::cerr << "No function specified for computing the Jacobian in levmar_blec_der()\n";
		std::cerr << "If no such function is available, use levmar_blec_dif() rather than levmar_blec_der()\n";
		return LMPP_ERROR;
	}

	if( !lb && !ub )
	{
		std::cerr << "levmar_blec_der(): lower and upper bounds for box constraints cannot be both NULL, use levmar_lec_der() in this case!\n";
		return LMPP_ERROR;
	}

	if( !levmar_box_check(lb, ub, m) )
	{
		std::cerr << "levmar_blec_der(): at least one lower bound exceeds the upper one\n";
		return LMPP_ERROR;
	}

	lmblec_data<FLOATTYPE> data;

	/* measurement vector needs to be extended by m */
	if(x) /* nonzero x */
	{
		data.x.resize(n+m);
		FLOATTYPE* data_x = data.x.data();
		std::copy( x, x + n, data_x );
		std::fill( data_x + n, data_x + n+m, zero<FLOATTYPE>() );
	}

	data.w.resize(m);
	data.bctype.resize(m);

	/* note: at this point, one of lb, ub are not NULL */
	for( int i = 0; i < m; ++i )
	{
		data.w[i] = !wghts ? FLOATTYPE(LMPP_BC_WEIGHT) : wghts[i];
		if( !lb )
			data.bctype[i] = BCType::High;

		else if( !ub )
			data.bctype[i] = BCType::Low;

		else if( ub[i] != std::numeric_limits<FLOATTYPE>::max() &&
				 lb[i] != std::numeric_limits<FLOATTYPE>::min() )
			data.bctype[i] = BCType::Interval;

		else if( lb[i] != std::numeric_limits<FLOATTYPE>::min() )
			data.bctype[i] = BCType::Low;

		else
			data.bctype[i] = BCType::High;
	}

	data.lb    = lb;
	data.ub    = ub;
	data.func  = func;
	data.jacf  = jacf;
	data.adata = adata;

	FLOATTYPE locinfo[LMPP_INFO_SZ];
	if( !info )
		info = locinfo; /* make sure that levmar_lec_der() is called with non-null info */

	return levmar_lec_der( lmblec_func<FLOATTYPE>, lmblec_jacf, p, data.x.empty() ? nullptr : data.x.data(),
						   m, n+m, A, b, k, itmax, opts, info, work, covar, (void *)&data );
}

/* Similar to the levmar_blec_der() function above, except that the Jacobian is approximated
 * with the aid of finite differences (forward or central, see the comment for the opts argument)
 */
template<typename FLOATTYPE>
int levmar_blec_dif_impl( FPTR<FLOATTYPE> func,    /* functional relation describing measurements. A p \in R^m yields a \hat{x} \in  R^n */
						  FLOATTYPE*      p,       /* I/O: initial parameter estimates. On output has the estimated solution             */
						  FLOATTYPE*      x,       /* I: measurement vector. NULL implies a zero vector                                  */
						  int             m,       /* I: parameter vector dimension (i.e. #unknowns)                                     */
						  int             n,       /* I: measurement vector dimension                                                    */
						  FLOATTYPE*      lb,      /* I: vector of lower bounds. If NULL, no lower bounds apply                          */
						  FLOATTYPE*      ub,      /* I: vector of upper bounds. If NULL, no upper bounds apply                          */
						  FLOATTYPE*      A,       /* I: constraints matrix, kxm                                                         */
						  FLOATTYPE*      b,       /* I: right hand constraints vector, kx1                                              */
						  int             k,       /* I: number of constraints (i.e. A's #rows)                                          */
						  FLOATTYPE*      wghts,   /* mx1 weights for penalty terms, defaults used if NULL                               */
						  int             itmax,   /* I: maximum number of iterations                                                    */
						  FLOATTYPE       opts[5], /* I: opts[0-3] = minim. options [\mu, \epsilon1, \epsilon2, \epsilon3, \delta]. Respectively the
						 	                       * scale factor for initial \mu, stopping thresholds for ||J^T e||_inf, ||Dp||_2 and ||e||_2 and
						 	                       * the step used in difference approximation to the Jacobian. Set to NULL for defaults to be used.
						 	                       * If \delta<0, the Jacobian is approximated with central differences which are more accurate
						 	                       * (but slower!) compared to the forward differences employed by default. 
						 	                       */
						  FLOATTYPE       info[LMPP_INFO_SZ],
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
						  FLOATTYPE*      work,    /* working memory at least LM_BLEC_DIF_WORKSZ() reals large, allocated if NULL */
						  FLOATTYPE*      covar,   /* O: Covariance matrix corresponding to LS solution; mxm. Set to NULL if not needed. */
						  void*           adata )  /* pointer to possibly additional data, passed uninterpreted to func.
						 						   * Set to NULL if not needed
						 						   */
{
	if( !lb && !ub )
	{
		std::cerr << "levmar_blec_dif(): lower and upper bounds for box constraints cannot be both NULL, use levmar_lec_dif() in this case!\n";
		return LMPP_ERROR;
	}

	if( !levmar_box_check(lb, ub, m) )
	{
		std::cerr << "levmar_blec_dif(): at least one lower bound exceeds the upper one\n";
		return LMPP_ERROR;
	}

	lmblec_data<FLOATTYPE> data;

	/* measurement vector needs to be extended by m */
	if(x) /* nonzero x */
	{
		data.x.resize(n+m);
		FLOATTYPE* data_x = data.x.data();
		std::copy( x, x + n, data_x );
		std::fill( data_x + n, data_x + n+m, zero<FLOATTYPE>() );
	}

	data.w.resize(m);
	data.bctype.resize(m);

	/* note: at this point, one of lb, ub are not NULL */
	for( int i = 0; i < m; ++i )
	{
		data.w[i] = !wghts ? FLOATTYPE(LMPP_BC_WEIGHT) : wghts[i];
		if( !lb )
			data.bctype[i] = BCType::High;

		else if( !ub )
			data.bctype[i] = BCType::Low;

		else if( ub[i] != std::numeric_limits<FLOATTYPE>::max() &&
				 lb[i] != std::numeric_limits<FLOATTYPE>::min() )
			data.bctype[i] = BCType::Interval;

		else if( lb[i] != std::numeric_limits<FLOATTYPE>::min() )
			data.bctype[i] = BCType::Low;

		else
			data.bctype[i] = BCType::High;
	}

	data.lb    = lb;
	data.ub    = ub;
	data.func  = func;
	data.jacf  = nullptr;
	data.adata = adata;

	FLOATTYPE locinfo[LMPP_INFO_SZ];
	if( !info )
		info = locinfo; /* make sure that levmar_lec_dif() is called with non-null info */

	return levmar_lec_dif( lmblec_func<FLOATTYPE>, p, data.x.empty() ? nullptr : data.x.data(),
						   m, n+m, A, b, k, itmax, opts, info, work, covar, (void *)&data );
}

#ifdef LMPP_DBL_PREC
int levmar_blec_der( FPTR<double> func,
					 FPTR<double> jacf,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      A,
					 double*      b,
					 int          k,
					 double*      wghts,
					 int          itmax,
					 double*      opts,
					 double*      info,
					 double*      work,
					 double*      covar,
					 void*        adata )
{
	return levmar_blec_der_impl<double>( func, jacf, p, x, m, n, lb, ub, A, b, k, wghts, itmax, opts, info, work, covar, adata );
}

int levmar_blec_dif( FPTR<double> func,
					 double*      p,
					 double*      x,
					 int          m,
					 int          n,
					 double*      lb,
					 double*      ub,
					 double*      A,
					 double*      b,
					 int          k,
					 double*      wghts,
					 int          itmax,
					 double*      opts,
					 double*      info,
					 double*      work,
					 double*      covar,
					 void*        adata )
{
	return levmar_blec_dif_impl<double>( func, p, x, m, n, lb, ub, A, b, k, wghts, itmax, opts, info, work, covar, adata );
}
#endif /* LM_DBL_PREC */

#ifdef LMPP_SNGL_PREC
int levmar_blec_der( FPTR<float> func,
					 FPTR<float> jacf,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      A,
					 float*      b,
					 int         k,
					 float*      wghts,
					 int         itmax,
					 float*      opts,
					 float*      info,
					 float*      work,
					 float*      covar,
					 void*       adata )
{
	return levmar_blec_der_impl<float>( func, jacf, p, x, m, n, lb, ub, A, b, k, wghts, itmax, opts, info, work, covar, adata );
}

int levmar_blec_dif( FPTR<float> func,
					 float*      p,
					 float*      x,
					 int         m,
					 int         n,
					 float*      lb,
					 float*      ub,
					 float*      A,
					 float*      b,
					 int         k,
					 float*      wghts,
					 int         itmax,
					 float*      opts,
					 float*      info,
					 float*      work,
					 float*      covar,
					 void*       adata )
{
	return levmar_blec_dif_impl<float>( func, p, x, m, n, lb, ub, A, b, k, wghts, itmax, opts, info, work, covar, adata );
}
#endif /* LMPP_SNGL_PREC */

#endif /* LMPP_HAVE_LAPACK */
