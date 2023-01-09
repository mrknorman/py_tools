#ifndef PY_TOOLS_H
#define PY_TOOLS_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>

#include <Python.h>
#include "numpy/arrayobject.h"

#include "text.h"

#define INIT_PY(...) do { \
    if (!Py_IsInitialized()) { initPy(); }\
    if(Py_IsInitialized()) {__VA_ARGS__}\
    else { \
        if (verbosity > 0)\
        {\
            fprintf(\
                stderr, \
                "PyFunctions: \nPython not intilizised.\n"\
            );\
        }\
    return 1000; }\
    } while(0)

bool initPy() {		
    
	Py_InitializeEx(0);
	PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

	return Py_IsInitialized();
}

void deinitPy() {
	
	// Deintilisation and reintilization causes segfault with some common python
	// packages like NumPy.
    
    Py_DECREF(PyImport_ImportModule("threading"));
    Py_FinalizeEx();
}

PyObject *makePyArrayFloat(
    const float   *array, 
    const int32_t  num_elements
    ) {

	import_array();
	npy_intp dims = (npy_intp) num_elements;
	
	return PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, (void *) array);	
}

PyObject *makePyArrayShort(
    const uint16_t *array, 
    const int32_t   num_elements
    ) {

	import_array();
	npy_intp dims = (npy_intp) num_elements;
	
	return PyArray_SimpleNewFromData(1, &dims, NPY_SHORT, (void *) array);	
}

PyObject *makePyArrayInt(
    const int32_t *array, 
    const int32_t  num_elements
    ) {

	import_array();
	npy_intp dims = (npy_intp) num_elements;
	
	return PyArray_SimpleNewFromData(1, &dims, NPY_INT, (void *) array);	
}

PyObject *makeListString(
          char    **string, 
    const int32_t   num_elements
    )  {

    PyObject* list = PyList_New(num_elements);
    for (int32_t index = 0; index < num_elements; index++) 
	{
        PyList_SetItem(list, index, PyUnicode_FromString(string[index]));
    }
	
    return list;
}

float *makeArr(
          PyObject *list, 
    const int32_t   num_elements
    ) {

    float *array = malloc(sizeof(float)*(size_t)num_elements);
    for (int32_t index = 0; index < num_elements; index++) 
	{
        array[index] = (float) PyFloat_AsDouble(PyList_GetItem(list, index));
    }
	
    return array;
}

bool runPythonFunc(
	const int32_t    verbosity,
    const int32_t    num_args, 
    const char      *module_path_string, 
    const char      *function_name, 
          PyObject **args, 
          float    **output_array, 
          int32_t    output_length
    ) {
	
	int32_t error_code = 0;
	
    PyObject *pModule, *pFunc;
    PyObject *pArgs, *pValue;
	
	path_s model_path = newPath(module_path_string);
	
	PyObject *sys_path = PySys_GetObject("path");
	PyList_Append(sys_path, PyUnicode_FromString(model_path.directory));

    pModule = PyImport_ImportModule(model_path.base);

    if (pModule != NULL) 
	{
        pFunc = PyObject_GetAttrString(pModule,  function_name);

        if (pFunc && PyCallable_Check(pFunc)) 
		{
            pArgs = PyTuple_New(num_args);
            for (int32_t index = 0; index < num_args; index++) 
			{
                pValue = args[index];
                if (!pValue) {
				
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
					
					if (verbosity > 0)
					{
						fprintf(
							stderr, 
							"runPythonFunc:\n Cannot convert argument: %i.\n", 
							index
						);
					}
					
                    error_code += 1;
					return error_code;
                }
				
                PyTuple_SetItem(pArgs, index, pValue);
            }

            pValue = PyObject_CallObject(pFunc, pArgs);

            *output_array = makeArr(pValue, output_length);

            Py_DECREF(pArgs);
            if (pValue != NULL)
			{
                Py_DECREF(pValue);
            } 
			else 
			{
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
				
				if (verbosity > 0)
				{
					fprintf(stderr,"runPythonFunc:\n Call failed.\n");
				}
				
                error_code += 10;
				return error_code;
            }
			
        } 
		else 
		{
            if (PyErr_Occurred()) 
			{
				PyErr_Print();
			}
			
			if (verbosity > 0)
			{
				fprintf(
                    stderr, 
                    "runPythonFunc:\n Cannot find function: \"%s\".\n", 
                    function_name
                );
			}
		}
		
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } 
	else 
	{
        PyErr_Print();
		
		if (verbosity > 0)
		{
			fprintf(
                stderr, 
                "runPythonFunc:\n Failed to load: \"%s\".\n", 
                model_path.base
            );
		}
		
        error_code += 100;
		return error_code;
    }

    return error_code;
}

bool checkArrayLengthEquality(
	const int32_t verbosity,
    const int32_t length_1, 
    const int32_t length_2
    ) {
	
	bool result = false;
	if (length_1 != length_2) 
	{
	
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"X array length (%i) must equal Y array length (%i)."
				"Exiting.\n", 
				length_1, 
				length_2
			); 		
		}
		
		result = false;
	} 
	else 
	{
		result = true;
	}
	
	return result;
}

//Python Plotting Functions:

typedef struct Axis{
    char *name;
    char *label;
} axis_s;

PyObject *convertAxisToPyObject(
    const axis_s axis
    ) {
    
    // Unpack axis:
    const char *name  = axis.name;
    const char *label = axis.label;
    
    // Set type format:
    const char *template = "ss";
    return Py_BuildValue(template, name, label);
}

typedef struct Axes{ 
    char    *name;
    int32_t  num_axis;
    axis_s  *axis;
} axes_s;

PyObject *convertAxesToPyObject(
    const axes_s axes
    ) {
    
    // Unpack axes:
    const char    *name     = axes.name;
    const int32_t  num_axis = axes.num_axis;
    const axis_s  *axis     = axes.axis;
    
    PyObject *py_axis = PyTuple_New(num_axis);
    for (int32_t index = 0; index < num_axis; index++)
    {
        PyTuple_SetItem(
            py_axis,
            index,
            convertAxisToPyObject(axis[index])
        );
    }
    
    // Set type format:
    const char *template = "sO";
    return Py_BuildValue(template, name, py_axis);
}

typedef struct SeriesValues{
    char   *axis_name;
    float  *values;
} series_values_s;

PyObject *convertSeriesValuesToPyObject(
    const series_values_s values,
    const int32_t         num_elements
    ) {
    
    // Unpack axes:
    const char  *axis_name   = values.axis_name;
    const float *values_     = values.values;
    
    PyObject *py_values = makePyArrayFloat(values_, num_elements);
    
    // Set type format:
    const char *template = "sO";
    return Py_BuildValue(template, axis_name, py_values);
}

typedef struct Series{
    char   *label;
    char   *axes_name;
    int32_t num_elements;   
    int32_t num_axis;

    series_values_s *values;
} series_s;

PyObject *convertSeriesToPyObject(
    const int32_t  verbosity,
    const series_s series
    ) {
    
    // Unpack series:
    const char            *label        = series.label;
    const char            *axes_name    = series.axes_name;
    const int32_t          num_elements = series.num_elements;
    const int32_t          num_axis     = series.num_axis;
    const series_values_s *values       = series.values;
    
    PyObject *py_values = PyTuple_New(num_axis);
    for (int32_t index = 0; index < num_axis; index++)
    {
        if ( PyTuple_SetItem(
            py_values,
            index,
            convertSeriesValuesToPyObject(values[index], num_elements)
        ) != 0 ) {
         fprintf(
                stderr, 
                "convertSeriesToPyObject: \n"
                "Warning cannot set tuple item values %s.",
                values[index].axis_name
            );
        }
    }
    
    // Set type format:
    const char *template = "ssO";
    
    PyObject *tuple = Py_BuildValue(template, label, axes_name, py_values);
    if ((tuple == NULL) && (verbosity > 0))
    {
        fprintf(
            stderr, 
            "ConvertFigureToPyObject:\n"
            "Warning! Cannot build value.\n"
        );
    }
    
    return tuple;
}

typedef struct Figure{
    
    axes_s   *axes;
    int32_t   num_axes;

    series_s *series;
    int32_t   num_series;
} figure_s;

PyObject *convertFigureToPyObject(
    const int32_t  verbosity, 
    const figure_s figure
    ) {
    
    const int32_t   num_axes = figure.num_axes;
    const axes_s   *axes     = figure.axes;
    
    PyObject *py_axes = PyTuple_New(num_axes);
    for (int32_t index = 0; index < num_axes; index++)
    {
        if (PyTuple_SetItem(
            py_axes,
            index,
            convertAxesToPyObject(axes[index])
            ) != 0
        ) {
            if (verbosity > 0)
            {
                fprintf(
                    stderr, 
                    "ConvertFigureToPyObject: \n"
                    "Warning cannot set tuple item axis %s.",
                    axes[index].name
                );
                
                return NULL;
            }
        }
    }
    
    const int32_t   num_series = figure.num_series;
    const series_s *series     = figure.series;
    
    PyObject *py_series = PyTuple_New(num_series);
    for (int32_t index = 0; index < num_series; index++)
    {
        if (PyTuple_SetItem(
            py_series,
            index,
            convertSeriesToPyObject(verbosity, series[index])
            ) != 0
        ) {
            if (verbosity > 0)
            {
                fprintf(
                    stderr, 
                    "ConvertFigureToPyObject: \n"
                    "Warning cannot set tuple item series %s.",
                    series[index].label
                );
                
                return NULL;
            }
        }
    }
    
    if (py_series == NULL) 
    {
        fprintf(stderr, "Warning! Py series is NULL! \n");
    }
    else if (py_axes == NULL) 
    {
        fprintf(stderr, "Warning! Py axes is NULL! \n");
    }
    
    PyObject *tuple = Py_BuildValue("OO", py_axes, py_series);
    
    if ((tuple == NULL) && (verbosity > 0))
    {
        fprintf(
            stderr, 
            "ConvertFigureToPyObject:\n"
            "Warning! Cannot build value.\n"
        );
    }
    
    return tuple;
    
}

int32_t plotFigure(
	const int32_t   verbosity,
    const figure_s  figure,
    const char     *output_file_path
    ) {
    
    int32_t error_code = 0;
    
    const char* output_directory_path = newPath(output_file_path).directory; 
        
	INIT_PY(
        if (checkCreateDirectory(verbosity, output_directory_path))
            {
            float  *output_array = NULL;

            const int32_t output_length = 1;
            const int32_t num_args      = 3;
            PyObject *args[] = {
                PyLong_FromLong((long)verbosity),
                convertFigureToPyObject(verbosity, figure),
                PyUnicode_FromString(output_file_path),
                NULL
            };

            error_code += runPythonFunc(
                verbosity,
                num_args,
                "./include/py_functions", 
                "plot_figure", 
                args, 
                &output_array, 
                output_length
            );
        }
        else
        {
            error_code = 10000;
        }
    );
     
	return error_code;
}

#endif