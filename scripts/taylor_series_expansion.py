import sympy

def generate_taylor_expansion(func_expr_str, var_symbol_str, expansion_point, order):
    """
    Generates the Taylor expansion of a function as a symbolic expression.

    Args:
        func_expr_str (str): A string representation of the function (e.g., "sin(x)", "exp(x)/cos(x)").
        var_symbol_str (str): The string for the variable (e.g., "x").
        expansion_point (float or int): The point around which to expand the function.
        order (int): The order of the Taylor polynomial.

    Returns:
        sympy.Expr: The symbolic Taylor expansion polynomial.
        None: If an error occurs during parsing or calculation.
    """
    try:
        # Define the symbolic variable
        var = sympy.symbols(var_symbol_str)

        # Parse the function string.
        # For security, it's better to define a dictionary of allowed functions
        # if this were to be used with arbitrary untrusted input.
        # For this example, we'll assume trusted input for common functions.
        local_dict = {var_symbol_str: var}
        local_dict.update(sympy.__dict__) # Add sympy functions like sin, cos, exp to the scope

        # Evaluate the string expression to get the symbolic function
        # Make sure to use sympy's functions (e.g., sympy.sin, sympy.exp)
        # For example, if func_expr_str is "sin(x)", this will become sympy.sin(var)
        func = sympy.sympify(func_expr_str, locals=local_dict)

        # Check if the result of sympify is a valid expression involving the variable
        if not isinstance(func, sympy.Expr) or var not in func.free_symbols:
            if not func.is_constant(): # Allow constant functions
                 print(f"Error: Could not create a valid symbolic function from '{func_expr_str}' with variable '{var_symbol_str}'.")
                 print(f"Parsed function: {func}, Free symbols: {func.free_symbols}")
                 return None


        # Calculate the Taylor series expansion
        # The 'n' parameter in sympy.series is the degree of the (x-x0) term.
        # So for an 'order'-th polynomial, we need terms up to (x-x0)**order.
        # The series function includes an O() term, which we'll remove.
        taylor_series = sympy.series(func, var, expansion_point, order + 1)

        # Remove the O() term to get the polynomial
        taylor_polynomial = taylor_series.removeO()

        return taylor_polynomial

    except (SyntaxError, TypeError, NameError, AttributeError) as e:
        print(f"An error occurred: {e}")
        print("Please ensure the function string and variable are correct and use sympy functions (e.g., sympy.sin, sympy.cos).")
        return None

if __name__ == "__main__":
    print("Symbolic Taylor Expansion Generator")
    print("-----------------------------------")

    # --- Example 1: sin(x) around 0 ---
    func_str_1 = "sin(x)"
    var_str_1 = "x"
    point_1 = 0
    order_1 = 4 # Requesting a 4th order polynomial

    print(f"\nExample 1: Expanding f({var_str_1}) = {func_str_1} around {var_str_1}0 = {point_1} up to order {order_1}")
    taylor_poly_1 = generate_taylor_expansion(func_str_1, var_str_1, point_1, order_1)

    if taylor_poly_1 is not None:
        print(f"The Taylor polynomial is: P{order_1}({var_str_1}) = {taylor_poly_1}")
        # For LaTeX pretty printing:
        # print(f"LaTeX: {sympy.latex(taylor_poly_1)}")

    # --- Example 2: exp(y) around 1 ---
    func_str_2 = "exp(y)"
    var_str_2 = "y"
    point_2 = 1
    order_2 = 3 # Requesting a 3rd order polynomial

    print(f"\nExample 2: Expanding f({var_str_2}) = {func_str_2} around {var_str_2}0 = {point_2} up to order {order_2}")
    taylor_poly_2 = generate_taylor_expansion(func_str_2, var_str_2, point_2, order_2)

    if taylor_poly_2 is not None:
        print(f"The Taylor polynomial is: P{order_2}({var_str_2}) = {taylor_poly_2}")
        # print(f"LaTeX: {sympy.latex(taylor_poly_2)}")

    # --- Example 3: 1/(1-z) around 0 ---
    func_str_3 = "1/(1-z)"
    var_str_3 = "z"
    point_3 = 0
    order_3 = 5

    print(f"\nExample 3: Expanding f({var_str_3}) = {func_str_3} around {var_str_3}0 = {point_3} up to order {order_3}")
    taylor_poly_3 = generate_taylor_expansion(func_str_3, var_str_3, point_3, order_3)

    if taylor_poly_3 is not None:
        print(f"The Taylor polynomial is: P{order_3}({var_str_3}) = {taylor_poly_3}")

    # --- Example 4: A more complex function log(1+x**2) ---
    func_str_4 = "log(1+x**2)" # sympy.log is natural logarithm
    var_str_4 = "x"
    point_4 = 0
    order_4 = 6

    print(f"\nExample 4: Expanding f({var_str_4}) = {func_str_4} around {var_str_4}0 = {point_4} up to order {order_4}")
    taylor_poly_4 = generate_taylor_expansion(func_str_4, var_str_4, point_4, order_4)

    if taylor_poly_4 is not None:
        print(f"The Taylor polynomial is: P{order_4}({var_str_4}) = {taylor_poly_4}")


    # --- Interactive Input ---
    print("\n--- Try your own function ---")
    try:
        user_func_str = input("Enter the function (e.g., cos(x), x**3 + 2*x): ")
        user_var_str = input("Enter the variable (e.g., x): ")
        user_point_str = input("Enter the expansion point (e.g., 0): ")
        user_order_str = input("Enter the desired order of the polynomial: ")

        user_point = float(user_point_str) # Or int() if preferred
        user_order = int(user_order_str)

        if user_order < 0:
            print("Order must be a non-negative integer.")
        else:
            print(f"\nExpanding f({user_var_str}) = {user_func_str} around {user_var_str}0 = {user_point} up to order {user_order}")
            user_taylor_poly = generate_taylor_expansion(user_func_str, user_var_str, user_point, user_order)

            if user_taylor_poly is not None:
                print(f"The Taylor polynomial is: P{user_order}({user_var_str}) = {user_taylor_poly}")
                print(f"LaTeX form: {sympy.latex(user_taylor_poly)}")

    except ValueError:
        print("Invalid input for point or order. Please enter numerical values.")
    except Exception as e:
        print(f"An unexpected error occurred during interactive input: {e}")

