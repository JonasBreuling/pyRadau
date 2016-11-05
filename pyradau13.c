#include <stdio.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

typedef void (*rhs_t)(int *n, double *x, double *y, double *f, float *rpar, int *ipar);
typedef void (*jac_t)(int *n, double *x, double *y, double *dfy, int *ldfy, float *rpar, int *ipar);
typedef void (*mas_t)(int *n, double *am, int *lmas, float *rpar, int *ipar);
typedef void (*sol_t)(int *nr, double *xold, double *x, double *y, double *cont, int *lrc, int *n,
		float *rpar, int *ipar, int *irtrn);

#ifdef __cplusplus
extern "C" {
#endif
	// Signature of Fortran RADAU function
	void radau_(
		int    *N,       // Dimension
		rhs_t   FCN,     // RHS function
		double *X,       // Initial time
		double *Y,       // Initial y array
		double *XEND,    // Final integration time
		double *H,       // Initial step size
		double *RTOL,    // Tolerances
		double *ATOL,    //
		int    *ITOL,    // Whether rtol and atol are vector valued
		jac_t   JAC ,    // Jacobian function
		int    *IJAC,    // Whether to use JAC (1) or finite differences
		int    *MLJAC,   // Band-width of the jacobian. N for full.
		int    *MUJAC,   // Upper band-width (0 is MLJAC == N)
		mas_t   MAS ,    // Mass matrix function
		int    *IMAS,    // Whether to use the mass matrix function
		int    *MLMAS,   // Band-width of the mass matrix
		int    *MUMAS,   // Upper band-widh of the mass matrix
		sol_t   SOLOUT,  // Dense output function
		int    *IOUT,    // Wether to call the dense output function
		double *WORK,    // Temporary array of size LWORK
		int    *LWORK,   // N*(LJAC+LMAS+7*N+3*NSMAX+3)+20
		int    *IWORK,   // Temporary array of size LIWORK
		int    *LIWORK,  // (2+(NSMAX-1)/2)*N+20
		float  *RPAR,    // User-supplied RHS arguments
		int    *IPAR,    // See RPAR
		int    *IDID,    // Return value
						 //  IDID= 1  COMPUTATION SUCCESSFUL,
						 //  IDID= 2  COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)
						 //  IDID=-1  INPUT IS NOT CONSISTENT,
						 //  IDID=-2  LARGER NMAX IS NEEDED,
						 //  IDID=-3  STEP SIZE BECOMES TOO SMALL,
						 //  IDID=-4  MATRIX IS REPEATEDLY SINGULAR.
		char *ERRBUF     // Buffer to hold error messages, at least 1024 bytes
	);

	double contra_(
		int *I,          // Dimension
		double *X,       // Time
		double *CONT,    // Continous data
		int *LRC         // Dimensionality of CONT
	);
#ifdef __cplusplus
}
#endif

static const char *idid_error_strings[] = { "COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)", NULL, NULL, "INPUT IS NOT CONSISTENT", "LARGER NMAX IS NEEDED", "STEP SIZE BECOMES TOO SMALL", "MATRIX IS REPEATEDLY SINGULAR" };

struct radau_options {
	PyObject *dense_callback;
	PyObject *rhs_fn;
	PyObject *jacobian;
	PyArrayObject *mass_matrix;
	PyArrayObject *y_out;
};

typedef struct {
    PyObject_HEAD

	PyArrayObject *y;
	double t0;
	double t1;
	double *cont;
	int lrc;
} RadauDenseEvaluator;

static PyObject *RadauDenseEvaluator_call(RadauDenseEvaluator *self, PyObject *args, PyObject *kwargs) {
	double time;
	if(self->cont == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "This object is only valid in the dense callback function");
		return NULL;
	}
	if(!PyArg_ParseTuple(args, "d", &time)) return NULL;
	if(time < self->t0 || time > self->t1) {
		char err[255];
		sprintf(err, "Time must fulfil %e <= t <= %e", self->t0, self->t1);
		PyErr_SetString(PyExc_ValueError, err);
		return NULL;
	}

	PyArrayObject *retval = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(self->y), PyArray_DIMS(self->y), NPY_DOUBLE);
	double *data = PyArray_DATA(retval);

	int n;
	for(n = 1; n<PyArray_SIZE(self->y)+1; n++) {
		data[n-1] = contra_(&n, &time, self->cont, &(self->lrc));
	}

	return (PyObject *)retval;
}

static PyTypeObject RadauDenseEvaluatorType = {
    PyObject_HEAD_INIT(NULL)
    .tp_name = "pyradau13.RadauDenseEvaluator",
    .tp_basicsize = sizeof(RadauDenseEvaluator),
    .tp_call = (ternaryfunc)RadauDenseEvaluator_call,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Dense evaluator"
};



static void radau_rhs(int *n, double *x, double *y, double *f, float *rpar, int *ipar) {
	struct radau_options *options = (struct radau_options *)ipar;

	PyObject *y_current = PyArray_SimpleNewFromData(PyArray_NDIM(options->y_out), PyArray_DIMS(options->y_out), NPY_DOUBLE, y);
	PyObject *rhs_retval = PyObject_CallFunction(options->rhs_fn, "dO", *x, y_current);
	Py_DECREF(y_current);

	if(rhs_retval == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The RHS function must return a value");
		}
		return;
	}
	PyArrayObject *rv_array = (PyArrayObject *)PyArray_FROM_OTF(rhs_retval, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if(rv_array == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The RHS function must return a vector");
		}
		return;
	}
	int use_n = PyArray_SIZE(rv_array);
	if(use_n >= *n) use_n = *n;
	memcpy(f, (double *)PyArray_DATA(rv_array), use_n * sizeof(double));
	Py_DECREF(rv_array);
	Py_DECREF(rhs_retval);
}

static void radau_jacobian(int *n, double *x, double *y, double *dfy, int *ldfy, float *rpar, int *ipar) {
	struct radau_options *options = (struct radau_options *)ipar;

	PyObject *y_current = PyArray_SimpleNewFromData(PyArray_NDIM(options->y_out), PyArray_DIMS(options->y_out), NPY_DOUBLE, y);
	PyObject *jacobian_retval = PyObject_CallFunction(options->jacobian, "dO", *x, y_current);
	Py_DECREF(y_current);

	if(jacobian_retval == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The jacobian function must return a value");
		}
		return;
	}
	PyArrayObject *rv_array = (PyArrayObject *)PyArray_FROM_OTF(jacobian_retval, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_F_CONTIGUOUS);
	if(rv_array == NULL) {
		if(PyErr_Occurred() == NULL) {
			PyErr_SetString(PyExc_RuntimeError, "The jacobian function must return a matrix");
		}
		return;
	}
	int use_n = PyArray_SIZE(rv_array);
	if(use_n >= *ldfy * *n) use_n = *ldfy * *n;
	memcpy(dfy, (double *)PyArray_DATA(rv_array), use_n * sizeof(double));
	Py_DECREF(rv_array);
	Py_DECREF(jacobian_retval);
}

static void radau_mass_matrix(int *n, double *am, int *lmas, float *rpar, int *ipar) {
	struct radau_options *options = (struct radau_options *)ipar;

	int use_n = PyArray_SIZE(options->mass_matrix);
	if(use_n >= *n * *lmas) use_n = *n * *lmas;
	memcpy(am, (double *)PyArray_DATA(options->mass_matrix), use_n * sizeof(double));
}


static void radau_dense_feedback(int *nr, double *xold, double *x, double *y, double *cont, int *lrc, int *n, float *rpar, int *ipar, int *irtrn) {
	struct radau_options *options = (struct radau_options *)ipar;

	PyObject *y_current = PyArray_SimpleNewFromData(PyArray_NDIM(options->y_out), PyArray_DIMS(options->y_out), NPY_DOUBLE, y);
	if(y_current == NULL) return;
	PyObject *y_copy = (PyObject *)PyArray_Copy((PyArrayObject *)y_current);
	if(y_copy == NULL) return;

	RadauDenseEvaluator *evaluator = PyObject_NEW(RadauDenseEvaluator, &RadauDenseEvaluatorType);
	evaluator->t1 = *x;
	evaluator->t0 = *xold;
	evaluator->lrc = *lrc;
	evaluator->cont = cont;
	evaluator->y = (PyArrayObject *)y_copy;

	PyObject *rhs_retval = PyObject_CallFunction(options->dense_callback, "ddOO", *xold, *x, y_copy, evaluator);
	Py_DECREF(y_copy);
	Py_DECREF(y_current);
	evaluator->cont = NULL;
	Py_DECREF(evaluator);


	if(PyErr_Occurred() != NULL) {
		// Must abort: An error occured
		*irtrn = 1;
	}
	else if(rhs_retval == NULL || PyObject_IsTrue(rhs_retval)) {
		*irtrn = -1;
	}
	else {
		*irtrn = 1;
	}
	if(rhs_retval) Py_DECREF(rhs_retval);
}

static PyObject *radau13(PyObject *self, PyObject *args, PyObject *kwargs) {
	// Parse arguments
	static char *kwlist[] = {"rhs_fn", "y0", "t", "initial_order", "min_order", "max_order",
		"max_steps", "dense_callback", "t0", "h0", "abstol", "reltol", "jacobian_fn", "mass_matrix",
		"max_step_size", "hessenberg_form", "newton_max_steps", "newton_start_zero", "dae_index_1_count",
		"dae_index_2_count", "dae_index_3_count", "classical_step_size_control",
		"step_size_safety_factor", "jacobian_recomputation_factor", "keep_step_size_lower_limit",
		"keep_step_size_upper_limit", "step_size_change_lower_limit",
		"step_size_change_upper_limit", "order_increase_if_contractivity_below",
		"order_decrease_if_contractivity_above", "order_decrease_only_if_step_size_ratio_above",
		"order_decrease_only_if_step_size_ratio_below",
		NULL};

	PyObject *rhs_fn, *y0, *times, *dense_callback = NULL, *hessenberg_form = NULL,
			 *newton_start_zero = NULL, *classical_step_size_control = NULL, *abstol = NULL,
			 *reltol = NULL, *jacobian_fn = NULL, *mass_matrix = NULL;
	double t_final, t_0 = 0., h_0 = 1e-6, max_step_size = 0., step_size_safety_factor = 0.,
		   jacobian_recomputation_factor = 0., keep_step_size_lower_limit = 0.,
		   keep_step_size_upper_limit = 0., step_size_change_lower_limit = 0.,
		   step_size_change_upper_limit = 0., order_increase_if_contractivity_below = 0.,
		   order_decrease_if_contractivity_above = 0., order_decrease_only_if_step_size_ratio_above = 0.,
		   order_decrease_only_if_step_size_ratio_below = 0.;
	int order = 13., min_order = 0, max_order = 0;
	unsigned int max_steps = 10000, newton_max_steps = 0, dae_index_1_count = 0, dae_index_2_count = 0,
				 dae_index_3_count = 0;

	if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|iiiIOddOOOOdOIOIIIOdddddddddd", kwlist,
				&rhs_fn, &y0, &times, &order, &min_order, &max_order, &max_steps, &dense_callback,
				&t_0, &h_0, &abstol, &reltol, &jacobian_fn, &mass_matrix, &max_step_size, &hessenberg_form,
				&newton_max_steps, &newton_start_zero, &dae_index_1_count, &dae_index_2_count, &dae_index_3_count,
				&classical_step_size_control, &step_size_safety_factor,
				&jacobian_recomputation_factor, &keep_step_size_lower_limit,
				&keep_step_size_upper_limit, &step_size_change_lower_limit,
				&step_size_change_upper_limit, &order_increase_if_contractivity_below,
				&order_decrease_if_contractivity_above,
				&order_decrease_only_if_step_size_ratio_above,
				&order_decrease_only_if_step_size_ratio_below
			)) return NULL;

	if(dense_callback != NULL && !PyCallable_Check(dense_callback)) {
		PyErr_SetString(PyExc_ValueError, "dense_callback, if set, must be callable");
		return NULL;
	}
	if(order != 13 && order != 5 && order != 9) {
		PyErr_SetString(PyExc_ValueError, "For the first step, only orders 13, 9 and 5 are implemented.");
		return NULL;
	}
	if((min_order != 0 && min_order != 1 && min_order != 5 && min_order != 9 && min_order != 13) ||
		(max_order != 0 && max_order != 1 && max_order != 5 && max_order != 9 && max_order != 13)) {
		PyErr_SetString(PyExc_ValueError, "Only orders 13, 9, 5 and 1 are implemented.");
		return NULL;
	}
	if(!PyCallable_Check(rhs_fn)) {
		PyErr_SetString(PyExc_ValueError, "rhs_fn must be callable");
		return NULL;
	}
	if(jacobian_fn && !PyCallable_Check(rhs_fn)) {
		PyErr_SetString(PyExc_ValueError, "jacobian_fn must be callable");
		return NULL;
	}
	PyArrayObject *y0_array = (PyArrayObject *)PyArray_FROM_OTF(y0, NPY_DOUBLE,  NPY_ARRAY_IN_ARRAY | NPY_ARRAY_C_CONTIGUOUS);
	if(!y0_array) {
		PyErr_SetString(PyExc_ValueError, "y0 must be convertible to a numpy array");
		return NULL;
	}
	PyArrayObject *times_array = (PyArrayObject *)PyArray_FROM_OTF(times, NPY_DOUBLE,  NPY_ARRAY_IN_ARRAY | NPY_ARRAY_C_CONTIGUOUS);
	if(!times_array) {
		PyErr_SetString(PyExc_ValueError, "t must be convertible to a numpy array");
		Py_DECREF(y0_array);
		return NULL;
	}
	PyArrayObject *mass_matrix_array = mass_matrix ? (PyArrayObject *)PyArray_FROM_OTF(mass_matrix, NPY_DOUBLE,  NPY_ARRAY_IN_ARRAY | NPY_ARRAY_F_CONTIGUOUS) : NULL;
	if(mass_matrix && !mass_matrix_array) {
		PyErr_SetString(PyExc_ValueError, "The mass matrix must be convertible to a numpy array");
		Py_DECREF(y0_array);
		Py_DECREF(times_array);
		return NULL;
	}
	PyArrayObject *reltol_array = reltol ? (PyArrayObject *)PyArray_FROM_OTF(reltol, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_C_CONTIGUOUS) : NULL;
	if(!reltol_array) {
		npy_intp dim = 1;
		reltol_array = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
		*((double *)PyArray_DATA(reltol_array)) = 1e-9;
	}
	PyArrayObject *abstol_array = abstol ? (PyArrayObject *)PyArray_FROM_OTF(abstol, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_C_CONTIGUOUS) : NULL;
	if(!abstol_array) {
		npy_intp dim = 1;
		abstol_array = (PyArrayObject *)PyArray_SimpleNew(1, &dim, NPY_DOUBLE);
		*((double *)PyArray_DATA(abstol_array)) = 1e-13;
	}
	if(PyArray_SIZE(abstol_array) != PyArray_SIZE(reltol_array) || (PyArray_SIZE(abstol_array) > 1 && PyArray_SIZE(abstol_array) != PyArray_SIZE(y0_array))) {
		PyErr_SetString(PyExc_ValueError, "reltol/abstol must either both be scalar or both vectors of the same length as y0, and must both be set if either is.");
		Py_DECREF(y0_array);
		Py_DECREF(reltol_array);
		Py_DECREF(abstol_array);
		if(mass_matrix_array) Py_DECREF(mass_matrix_array);
		return NULL;
	}

	// Prepare memory for RADAU
	int n = PyArray_SIZE(y0_array);
	PyArrayObject *y_out = (PyArrayObject *)PyArray_Copy(y0_array); // PyArray_FromArray(y0_array, PyArray_DESCR(y0_array), NPY_ARRAY_WRITEABLE | NPY_ARRAY_ENSURECOPY);
	struct radau_options options = { dense_callback, rhs_fn, jacobian_fn, mass_matrix_array, y_out };
	int lwork  = n * ((mass_matrix ? n : 0) + n + 7*n + 3*7 + 3) + 20;
	int liwork = (2 + (7 - 1) / 2) * n + 20;
	double *work = malloc(lwork * sizeof(double));
	int *iwork = malloc(lwork * sizeof(int));
	int no[] = { 0, 0, 0, 0 };
	int nc[] = { n, n, n };
	int bw[] = { 0, 0 };
	int yes = 1;
	int idid = 1;

	int time_levels = PyArray_SIZE(times_array);
	int current_level;

	memset(iwork, 0, sizeof(int) * liwork);
	memset(work, 0, sizeof(double) * lwork);
	iwork[0] = hessenberg_form && PyObject_IsTrue(hessenberg_form);
	iwork[1] = max_steps;
	iwork[2] = newton_max_steps;
	iwork[3] = newton_start_zero && PyObject_IsTrue(newton_start_zero);
	iwork[4] = dae_index_1_count;
	iwork[5] = dae_index_2_count;
	iwork[6] = dae_index_3_count;
	iwork[7] = classical_step_size_control && PyObject_IsTrue(classical_step_size_control);
	iwork[10] = (min_order + 1) / 2;
	iwork[11] = (max_order + 1) / 2;
	iwork[12] = (order + 1) / 2;
	work[1] = step_size_safety_factor;
	work[2] = jacobian_recomputation_factor;
	work[4] = keep_step_size_lower_limit;
	work[5] = keep_step_size_upper_limit;
	work[6] = max_step_size;
	work[7] = step_size_change_lower_limit;
	work[8] = step_size_change_upper_limit;
	work[9] = order_increase_if_contractivity_below;
	work[10] = order_decrease_if_contractivity_above;
	work[11] = order_decrease_only_if_step_size_ratio_below;
	work[12] = order_decrease_only_if_step_size_ratio_above;

	PyObject *list_retval = PyList_New(0);

	char errbuf[1024];
	memset(errbuf, 0, sizeof(errbuf));

	for(current_level = 0; current_level < time_levels; current_level++) {
		t_final = ((double *)PyArray_DATA(times_array))[current_level];

		if(t_final < t_0) {
			PyErr_SetString(PyExc_ValueError, "Integration times must be increasing");
			break;
		}

		// Call RADAU
		if(t_final > t_0)
		radau_(
			&nc[0],                            // Dimension
			radau_rhs,                         // RHS function
			&t_0,                              // Initial time
			PyArray_DATA(y_out),               // Initial y array and output
			&t_final,                          // Final integration time
			&h_0,                              // Initial step size
			PyArray_DATA(reltol_array),        // Tolerances
			PyArray_DATA(abstol_array),        //
			PyArray_SIZE(reltol_array) > 1 ? (&yes) : (&no[0]),  // Whether rtol and atol are vector valued
			radau_jacobian,                    // Jacobian function
			jacobian_fn ? &yes : &no[1],       // Whether to use JAC (1) or finite differences
			&nc[1],                            // Band-width of the jacobian. N for full.
			&bw[0],                            // Upper band-width (0 is MLJAC == N)
			radau_mass_matrix,                 // Mass matrix function
			mass_matrix ? &yes : &no[1],       // Whether to use the mass matrix function
			&nc[2],                            // Band-width of the mass matrix
			&bw[1],                            // Upper band-widh of the mass matrix
			radau_dense_feedback,              // Dense output function
			dense_callback ? &yes : &no[3],    // Wether to call the dense output function
			&work[0],                          // Temporary array of size LWORK
			&lwork,                            // N*(LJAC+LMAS+7*N+3*NSMAX+3)+20
			&iwork[0],                         // Temporary array of size LIWORK
			&liwork,                           // (2+(NSMAX-1)/2)*N+20
			NULL,                              // User-supplied RHS arguments
			(int*)&options,                    // See RPAR
			&idid,                             // Return value
											   // IDID= 1  COMPUTATION SUCCESSFUL,
											   // IDID= 2  COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)
											   // IDID=-1  INPUT IS NOT CONSISTENT,
											   // IDID=-2  LARGER NMAX IS NEEDED,
											   // IDID=-3  STEP SIZE BECOMES TOO SMALL,
											   // IDID=-4  MATRIX IS REPEATEDLY SINGULAR.
			errbuf
		);
		t_0 = t_final;

		if(idid != 1) break;
		if(time_levels > 1) {
			PyObject *cpy = PyArray_Copy(y_out);
			if(!cpy) return NULL;
			PyList_Append(list_retval, cpy);
		}
	}

	free(work);
	free(iwork);
	Py_DECREF(y0_array);
	Py_DECREF(reltol_array);
	Py_DECREF(abstol_array);
	if(mass_matrix_array) Py_DECREF(mass_matrix_array);

	if((idid != 1 && (idid != 2 || PyErr_Occurred() != NULL)) || y_out == NULL) {
		if(PyErr_Occurred() == NULL) {
			int i;
			for(i=sizeof(errbuf)-1; i>0 && (errbuf[i] == ' ' || errbuf[i] == '\n' || errbuf[i] == 0 || errbuf[i] == '\t'); i--) {
				errbuf[i] = 0;
			}
			char err[1024];
			snprintf(err, sizeof(err), "radau failed with idid = %d (%s)%s%s", idid, idid_error_strings[2 - idid], errbuf ? "\n" : "", errbuf);
			PyErr_SetString(PyExc_RuntimeError, err);
		}
		if(time_levels > 1) {
			Py_DECREF(list_retval);
		}
		Py_DECREF(y_out);
		return NULL;
	}

	if(PyErr_Occurred() != NULL) {
		Py_DECREF(list_retval);
		Py_DECREF(y_out);
		return NULL;
	}

	if(time_levels > 1) {
		PyObject *np_retval = PyArray_FromAny(list_retval, NULL, 0, 0, 0, NULL);
		Py_DECREF(list_retval);
		Py_DECREF(y_out);
		return np_retval;
	}
	else {
		Py_DECREF(list_retval);
		return (PyObject *)y_out;
	}
}

static PyMethodDef methods[] = {
	{"radau13", (PyCFunction)radau13, METH_VARARGS | METH_KEYWORDS,
		"Solve an ODE system using the RADAU13 integrator\n\n"
		"Syntax: radau13(rhs_fn, y0, t, initial_order=13, max_steps=10000,\n"
		"                dense_callback=None, t0=0, h0=1e-6, abstol=1e-13,\n"
		"                reltol=1e-9, max_step_size=0)\n\n"
		"Arguments:\n"
		" - rhs_fn must be a callable of the type rhs_fn(t, y), where\n"
		"     t is the current integration time, and\n"
		"     y is the current state vector.\n"
		"     It should return a vector df/dt.\n"
		" - y0 must be the initial state vector.\n"
		" - t must be the desired new time level, or an increasing list\n"
		"   of time levels.\n\n"
		"Optional arguments:\n"
		" - initial_order must be one of 13, 9 or 5.\n"
		" - min_order must be one of 13, 9, 5 or 1\n"
		" - max_order must be one of 13, 9, 5 or 1\n"
		" - max_steps denotes the maximal step count to take.\n"
		" - dense_callback must be either not set, or a callable of the\n"
		"     type dense_callback(told, t, y, cont). It is called after each\n"
		"     internal integration step. cont is a callable that takes an argument\n"
		"     from [told, t] and returns an approximation to the value at that time.\n"
		"     If dense_callback returns a value that evaluates to True, integration\n"
		"     is halted.\n"
		" - t0 denotes the initial time.\n"
		" - h0 denotes the initial step size.\n"
		" - abstol/reltol denote the integrator tolerances (scalar or vector)\n"
		" - jacobian_fn must be a callable of the type jabocian_fn(t, y), where\n"
		"     t is the current integration time, and\n"
		"     y is the current state vector.\n"
		"     It should return a matrix.\n"
		" - mass_matrix must be the mass matrix for implicit problems,\n"
		"     M \\dot x = f(t, x).\n"
		" - max_step_size denotes the maximal step size. It is only required\n"
		"     if you need fixed times in dense_callback; for normal operation\n"
		"     abstol/reltol are fine.\n\n"
		"Further optional arguments for fine grained control, only needed\n"
		"in complex situations; see RADAU.F documentation for details:\n"
		" max_step_size hessenberg_form newton_max_steps newton_start_zero\n"
		" dae_index_1_count dae_index_2_count dae_index_3_count\n"
		" classical_step_size_control step_size_safety_factor\n"
		" jacobian_recomputation_factor keep_step_size_lower_limit\n"
		" keep_step_size_upper_limit step_size_change_lower_limit\n"
		" step_size_change_upper_limit order_increase_if_contractivity_below\n"
		" order_decrease_if_contractivity_above\n"
		" order_decrease_only_if_step_size_ratio_above\n"
		" order_decrease_only_if_step_size_ratio_below\n\n"
		"Return value:\n"
		" The function returns the state at time t_final, or throws a\n"
		" RuntimeError if it vails. The runtime error contains the error message\n"
		" from RADAU13, which can be\n"
		"   IDID= 1  COMPUTATION SUCCESSFUL,\n"
		"   IDID= 2  COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)\n"
		"   IDID=-1  INPUT IS NOT CONSISTENT,\n"
		"   IDID=-2  LARGER NMAX IS NEEDED,\n"
		"   IDID=-3  STEP SIZE BECOMES TOO SMALL,\n"
		"   IDID=-4  MATRIX IS REPEATEDLY SINGULAR.\n"
	},
	{NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "pyradau13",
        "Python wrapper around the RADAU13 integrator",
        -1,
        methods
};

PyObject * PyInit_pyradau13(void) {
#else
PyMODINIT_FUNC initpyradau13(void) {
#endif

	RadauDenseEvaluatorType.tp_new = PyType_GenericNew;
	if(PyType_Ready(&RadauDenseEvaluatorType) < 0)
#if PY_MAJOR_VERSION >= 3
		return NULL;
#else
		return;
#endif

#if PY_MAJOR_VERSION >= 3
	PyObject *module = PyModule_Create(&moduledef);
#else
	(void)Py_InitModule("pyradau13", methods);
#endif

	import_array();

#if PY_MAJOR_VERSION >= 3
	return module;
#endif
}
