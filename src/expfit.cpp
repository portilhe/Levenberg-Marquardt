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

	////////////////////////////////////////////////////////////////////////////////////
	//  Example program that shows how to use levmar in order to fit the three-
	//  parameter exponential model x_i = p[0]*exp(-p[1]*i) + p[2] to a set of
	//  data measurements; example is based on a similar one from GSL.
	//
	//  Copyright (C) 2008  Manolis Lourakis (lourakis at ics forth gr)
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

//#include <cstdio>
//#include <cstdlib>
//#include <cmath>

#include <random>
#include <chrono>
#include <vector>
#include <iostream>

#include "levmar.h"

#ifndef LMPP_DBL_PREC
#error Example program assumes that levmar has been compiled with double precision, see LM_DBL_PREC!
#endif


/* the following macros concern the initialization of a random number generator for adding noise */
#undef REPEATABLE_RANDOM


#ifdef _MSC_VER // MSVC
#include <process.h>
#define GETPID  _getpid
#elif defined(__GNUC__) // GCC
#include <sys/types.h>
#include <unistd.h>
#define GETPID  getpid
#else
#warning Do not know the name of the function returning the process id for your OS/compiler combination
#define GETPID  0
#endif /* _MSC_VER */


/* Gaussian noise with mean m and variance s, uses the Box-Muller transformation */
double gNoise( double m, double s, std::mt19937& gen)
{
	std::normal_distribution<> d{m,s};
	return d(gen);
}

/* model to be fitted to measurements: x_i = p[0]*exp(-p[1]*i) + p[2], i=0...n-1 */
void expfunc( double *p, double *x, int m, int n, void *data )
{
  for( int i = 0; i < n; ++i )
    x[i] = p[0] * exp(-p[1]*i) + p[2];
}

/* Jacobian of expfunc() */
void jacexpfunc( double *p, double *jac, int m, int n, void *data )
{
  /* fill Jacobian row by row */
	int j = 0;
	for( int i = 0; i < n; ++i )
	{
		jac[j++] = exp(-p[1]*i);
		jac[j++] = -p[0]*i*exp(-p[1]*i);
		jac[j++] = 1.0;
	}
}

using mt19937rt = std::mt19937::result_type;

int main()
{
	std::random_device rd;
	// seed value is designed specifically to make initialization
	// parameters of std::mt19937 (instance of std::mersenne_twister_engine<>)
	// different across executions of application
	auto t1 = std::chrono::system_clock::now().time_since_epoch();
	auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();
	mt19937rt seed = rd() ^ ((mt19937rt)std::chrono::duration_cast<std::chrono::seconds>(t1).count() +
							 (mt19937rt)std::chrono::duration_cast<std::chrono::microseconds>(t2).count() );
#ifdef REPEATABLE_RANDOM
	std::mt19937 gen{0};
#else
	std::mt19937 gen{seed};
#endif


	constexpr int m =  3; // 3 parameters
	constexpr int n = 40; // 40 measurements

	std::vector<double> p(m);
	std::vector<double> x(n);
	std::vector<double> opts(LMPP_OPTS_SZ);
	std::vector<double> info(LMPP_INFO_SZ);

	/* generate some measurement using the exponential model with
	 * parameters (5.0, 0.1, 1.0), corrupted with zero-mean
	 * Gaussian noise of s=0.1
	 */
	for( int i = 0; i < n; ++i )
	    x[i] = (5.0*exp(-0.1*i) + 1.0) + gNoise( 0.0, 0.1, gen );

	/* initial parameters estimate: (1.0, 0.0, 0.0) */
	p[0] = 1.0;
	p[1] = 0.0;
	p[2] = 0.0;

	/* optimization control parameters; passing to levmar NULL instead of opts reverts to defaults */
	opts[0] = LMPP_INIT_MU;
	opts[1] = 1E-15;
	opts[2] = 1E-15;
	opts[3] = 1E-20;
	opts[4] = LMPP_DIFF_DELTA; // relevant only if the finite difference Jacobian version is used 

	/* invoke the optimization function */
	// with analytic Jacobian
	int ret = levmar_der( expfunc, jacexpfunc, p.data(), x.data(), m, n, 1000, opts.data(), info.data(), nullptr, nullptr, nullptr );
	// without Jacobian
	//int ret = levmar_dif( expfunc, p.data(), x.data(), m, n, 1000, opts.data(), info.data(), nullptr, nullptr, nullptr );

	std::cout << "Levenberg-Marquardt returned in " << info[5] << " iter, reason "
			  << info[6] << ", sumsq " << info[1] << " [" << info[0] << "]\n";
	std::cout << "Best fit parameters: " << p[0] << " " << p[1] << " " << p[2] << "\n";

	return 0;
}
