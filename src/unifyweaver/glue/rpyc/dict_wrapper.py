class DictWrapper:
    def __init__(self, arg=None,**args):
        if arg is None:
            self.dict = dict(**args)
        else:
            self.dict=arg
            self.dict.update(dict(**args))
    
    def get(self, key, default=None):  
        return self.dict.get(key, default)
    
    def update(self, other):
        # Handle different types of input
        if hasattr(other, 'dict'):  # Another DictWrapper
            self.dict.update(other.dict)
        else:  # Assume it's a dictionary-like object
            self.dict.update(other)
    def set(self,key,value):
        self.update({key:value})
    def update_kv(self, key, value):
        # Alternative method for single key-value updates
        self.dict[key] = value
    
    def update_pairs(self, **kwargs):
        # Alternative method using keyword arguments
        self.dict.update(kwargs)
    
    def __repr__(self):
        return f"DictWrapper({self.dict})"

def mkDictWrapper(**args):
    return DictWrapper(**args)
def wrapped_exec(code, initial_vars=None):
    """
    Execute Python code and return results in a DictWrapper.
    
    Args:
        code: Python code string to execute
        initial_vars: Optional dictionary of initial variables
    
    Returns:
        DictWrapper containing all variables from the execution
    """
    # Create execution namespace
    if isinstance(initial_vars,DictWrapper):
        initial_vars=initial_vars.dict
        
    namespace = {} if initial_vars is None else initial_vars
    
    # Execute the code in the namespace
    exec(code, globals(), namespace)
    
    # Return wrapped results
    return DictWrapper(namespace)
def call_in_namespace(func=None, ns={}, f_name="func", r_name="result", args=[], kwargs={}):
    #if ns is None:
    #    ns = {}  # Ensuring a fresh dictionary per function cal    
    #elif not isinstance(ns,dict):
    #    args_out = [ns] + list(args)
    #    ns={}
    #if func is not None:
    #    ns.update({f_name:func})  # Store function reference in namespace
    if isinstance(ns,DictWrapper):
        ns=ns.dict
    if func is not None:
        if callable(func):
            ns[f_name]=func
        elif isinstance(func,dict):
            if ns is None:
                ns=func
                func = None
            elif isinstance(ns,dict):
                ns.update(func)
        elif isinstance(func, str):
            f_name=func
            func=None
    ns[r_name] = ns[f_name](*args,**kwargs)  # Directly call and store result
    
    return ns[r_name]  # Retrieve the computed result
    
def wrapped_call(func=None, ns=None, f_name="func", r_name="result"):
    if isinstance(func, dict):  
        if ns is None:  
            ns = func  # If no namespace was provided, use the dictionary itself
            func = None
        else:  
            ns.update(func)  # If ns exists, merge the dictionaries
            func = none
    if ns is None:
        ns={}
    
    return lambda *args, **kwargs: call_in_namespace(func=func, ns=ns, f_name=f_name, r_name=r_name, args=args, kwargs=kwargs)


def wrapped_exec_with_globals(code, use_globals=True):
    """
    Alternative version that can optionally include global variables.
    """
    if use_globals:
        # Use current globals as the execution environment
        local_vars = {}
        exec(code, globals(), local_vars)
        return DictWrapper(local_vars)
    else:
        # Clean environment
        namespace = {}
        exec(code, {}, namespace)
        return DictWrapper(namespace)
def call_function(func, *args, **kwargs):
    """
    Wrapper to call any Python callable object.
    This bypasses potential issues with __call__ in Janus.
    """
    return func(*args, **kwargs)
def call_with_args(func, args_list):
    """
    Call a function with arguments from a list, expanding them as positional arguments.
    
    Args:
        func: The callable function
        args_list: List of arguments to expand and pass to the function
    
    Returns:
        Result of the function call
    """
    return func(*args_list)

def call_with_kwargs(func, kwargs_dict):
    """
    Call a function with keyword arguments from a dictionary.
    
    Args:
        func: The callable function  
        kwargs_dict: Dictionary of keyword arguments
    
    Returns:
        Result of the function call
    """
    return func(**kwargs_dict)

def call_with_mixed_args(func, args_list=None, kwargs_dict=None):
    """
    Call a function with both positional and keyword arguments.
    
    Args:
        func: The callable function
        args_list: List of positional arguments (optional)
        kwargs_dict: Dictionary of keyword arguments (optional)
    
    Returns:
        Result of the function call
    """
    args = args_list or []
    kwargs = kwargs_dict or {}
    return func(*args, **kwargs)
def wrap(d):
    if isinstance(d,DictWrapper):
        return d
    else:
        return DictWrapper(d)
def get_dict_value(d, key, default=None):
    return d.get(key, default)
    
def set_dict_value(d, key, value):
    out=wrap(d)
    out.set(key,value)
    return out
def getattr_wrapped(obj,key):
    value=getattr(obj,key)
    if isinstance(value,dict):
        value=wrap(value)
    return value
class BoundCallable:
    def __init__(self, func, context, func_name=None):
        self.func = func
        self.context = context
        self.name = func_name or getattr(func, '__name__', 'anonymous')
    
    def __call__(self, *args, **kwargs):
        try:
            return self.func(self.context, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Error in {self.name}: {e}") from e
    
    def __repr__(self):
        return f"BoundCallable({self.name}, context={type(self.context).__name__})"

import inspect
import sys

def get_module_classes(module=None):
    """
    Get all classes defined in a module.
    
    Args:
        module: Module object (defaults to current module)
    
    Returns:
        Dictionary mapping class names to class objects
    """
    if module is None:
        module = sys.modules[__name__]
    
    # Get classes defined in this module (not imported ones)
    classes = {}
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__:
            classes[name] = obj
    
    return classes

def get_class_constructor(class_name, module=None):
    """
    Get a specific class constructor by name.
    
    Args:
        class_name: Name of the class
        module: Module object (defaults to current module)
    
    Returns:
        Class constructor or None if not found
    """
    classes = get_module_classes(module) # Inherits current module default
    return classes.get(class_name)

def get_all_constructors(module=None):
    """
    Get all class constructors as a dictionary.
    Same as get_module_classes but more explicit naming.
    """
    return get_module_classes(module)

def create_constructor_factory(module=None):
    """
    Create a factory function that returns constructors by name.
    
    Returns:
        Function that takes class_name and returns constructor
    """
    classes = get_module_classes(module)
    return lambda class_name: classes.get(class_name)
def get_current_module():
    """Get reference to current module."""
    return sys.modules[__name__]
def call_and_wrap(func, *args, **kwargs):
    """
    Call a Python function and automatically wrap dictionary results.
    
    This solves the problem where Python functions return dictionaries
    that lose their object behavior when passed to Prolog.
    
    Args:
        func: The Python function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        - DictWrapper if result is a dict
        - Original result otherwise
    """
    result = func(*args, **kwargs)
    if isinstance(result, dict):
        return DictWrapper(result)
    return result
    