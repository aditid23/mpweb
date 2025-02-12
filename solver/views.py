from django.shortcuts import render
import numpy as np
import re
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.optimize import linprog


# Function to render the home page
def home(request):
    return render(request, 'home.html')


# Function to render the LP graph page
def mpweb(request):
    return render(request, "graphical_lp.html")


# Function to render the LP simplex page
def lpsimplex(request):
    return render(request, 'simplex_lp.html')


def transport(request):
    return render(request, 'transportation_lp.html')


# Break the input into coefficients of the constraints
def parse_equation(eq):
    eq = eq.replace(" ", "")
    eq = re.sub(r'(?<!\d)x', '1x', eq)
    eq = re.sub(r'(?<!\d)y', '1y', eq)
    match = re.match(r'([-+]?\d*\.?\d*)x([-+]?\d*\.?\d*)y(<=|>=|=)([-+]?\d*\.?\d+)', eq)
    if not match:
        match_x = re.match(r'([-+]?\d*\.?\d*)x(<=|>=|=)([-+]?\d*\.?\d+)', eq)
        match_y = re.match(r'([-+]?\d*\.?\d*)y(<=|>=|=)([-+]?\d*\.?\d+)', eq)
        if match_x:
            a = float(match_x.group(1) or 1)
            b = 0
            op = match_x.group(2)
            c = float(match_x.group(3))
            return a, b, c, op
        elif match_y:
            a = 0
            b = float(match_y.group(1) or 1)
            op = match_y.group(2)
            c = float(match_y.group(3))
            return a, b, c, op
        else:
            return None
    a = float(match.group(1) or 1)
    b = float(match.group(2) or 1)
    op = match.group(3)
    c = float(match.group(4))
    return a, b, c, op


# Identify the coefficients of x and y of the objective function
def parse_objective(objective):
    objective = objective.replace(" ", "")

    x_coeff = 0
    y_coeff = 0

    x_match = re.search(r'([+-]?\d*\.?\d*)x', objective)
    y_match = re.search(r'([+-]?\d*\.?\d*)y', objective)

    if x_match:
        x_coeff_str = x_match.group(1)
        x_coeff = float(x_coeff_str) if x_coeff_str not in ["", "+", "-"] else (1 if x_coeff_str != "-" else -1)

    if y_match:
        y_coeff_str = y_match.group(1)
        y_coeff = float(y_coeff_str) if y_coeff_str not in ["", "+", "-"] else (1 if y_coeff_str != "-" else -1)

    return x_coeff, y_coeff


# Calculate the intersection point of the two equations
def find_intersection(eq1, eq2):
    a1, b1, c1, _ = eq1
    a2, b2, c2, _ = eq2
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None

    x = (c1 * b2 - c2 * b1) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return x, y


# Check whether all constraints are satisfied
def is_feasible(point, constraints, tol=1e-6):
    x, y = point
    for a, b, c, op in constraints:
        value = a * x + b * y
        if (op == "<=" and value > c + tol) or (op == ">=" and value < c - tol):
            return False
    return True


# Define the limit of the axes based on the constraints
def determine_axis_limits(constraints):
    x_vals = []
    y_vals = []

    for a, b, c, op in constraints:
        if b != 0:
            x_vals.extend([0, c / a if a != 0 else 0])
            y_vals.append(c / b)
        else:
            x_vals.append(c / a)

    x_min = min(x_vals) - 5
    x_max = max(x_vals) + 5
    y_min = min(y_vals) - 5
    y_max = max(y_vals) + 5

    return (x_min, x_max), (y_min, y_max)


# Solve the linear programming problem and produce the graph
def solve_lp(request):
    if request.method == "POST":
        mode = request.POST.get("mode")
        objective = request.POST.get("objective")
        constraints_input = request.POST.getlist("constraints[]")

        try:
            c1, c2 = parse_objective(objective)
        except ValueError:
            return render(request, "graphical_lp.html", {"error_message": "Invalid objective function format."})

        constraints = []
        for eq in constraints_input:
            parsed = parse_equation(eq)
            if parsed:
                constraints.append(parsed)
            else:
                return render(request, "graphical_lp.html", {"error_message": f"Invalid constraint: {eq}"})

        corner_points = []
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                point = find_intersection(constraints[i], constraints[j])
                if point and is_feasible(point, constraints):
                    corner_points.append(point)

        if not corner_points:
            return render(request, "graphical_lp.html", {"error_message": "No feasible region found."})

        corner_points = np.array(corner_points)
        objective_values = c1 * corner_points[:, 0] + c2 * corner_points[:, 1]
        optimal_idx = np.argmax(objective_values) if mode == "maximize" else np.argmin(objective_values)
        optimal_point = corner_points[optimal_idx]
        optimal_value = objective_values[optimal_idx]

        x_range, y_range = determine_axis_limits(constraints)
        x = np.linspace(*x_range, 500)
        y = np.linspace(*y_range, 500)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(10, 8))
        plt.title("Feasible Region", fontsize=16)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.axhline(0, color='black', linewidth=0.7, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.7, linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.6)

        feasible_region = np.ones_like(X, dtype=bool)
        for a, b, c, op in constraints:
            if op == "<=":
                feasible_region &= (a * X + b * Y <= c)
            elif op == ">=":
                feasible_region &= (a * X + b * Y >= c)

        plt.contourf(X, Y, feasible_region, levels=[0.5, 1], colors=["green"], alpha=0.3)

        for a, b, c, op in constraints:
            if b != 0:
                y_line = (c - a * x) / b
                plt.plot(x, y_line, label=f"{a}x + {b}y {op} {c}", linewidth=1.5)
            else:
                x_line = c / a
                plt.axvline(x_line, label=f"{a}x {op} {c}", linewidth=1.5)

        for point in corner_points:
            plt.scatter(point[0], point[1], color='blue', zorder=5)
            plt.text(point[0], point[1], f"({point[0]:.2f}, {point[1]:.2f})", fontsize=10, color='blue', ha='left', fontweight='bold')

        plt.scatter(*optimal_point, color='red', label='Optimal Point', zorder=6)
        plt.text(optimal_point[0], optimal_point[1], f"({optimal_point[0]:.2f}, {optimal_point[1]:.2f})", fontsize=10, color='red', ha='left', fontweight='bold')

        plt.legend(loc="upper right", fontsize=10)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()

        return render(
            request,
            "graphical_lp.html",
            {
                "solution": {
                    "constraints": constraints_input,
                    "corner_points": corner_points.tolist(),
                    "optimal_point": optimal_point.tolist(),
                    "optimal_value": optimal_value,
                    "graph_url": f"data:image/png;base64,{image_base64}",
                    "objective_function": objective,
                    "mode": mode,
                    "mode_text": "Maximize" if mode == "maximize" else "Minimize"
                }
            },
        )
    return render(request, "graphical_lp.html")


# Simplex method implementation
def simplex(c, A, b):
    """
    Solve a linear programming problem using the simplex method.

    Parameters:
        c (np.array): Coefficients of the objective function.
        A (np.array): Coefficient matrix of constraints.
        b (np.array): Right-hand side values of constraints.

    Returns:
        solution (np.array): Optimal solution.
        optimal_value (float): Optimal value of the objective function.
    """
    num_constraints, num_variables = A.shape

    slack_vars = np.eye(num_constraints)
    tableau = np.hstack((A, slack_vars, b.reshape(-1, 1)))

    obj_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, obj_row))

    num_total_vars = num_variables + num_constraints

    while True:
        if np.all(tableau[-1, :-1] >= 0):
            break

        pivot_col = np.argmin(tableau[-1, :-1])

        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  
        pivot_row = np.argmin(ratios)

        if np.all(ratios == np.inf):
            raise ValueError("The problem is unbounded.")

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    solution = np.zeros(num_total_vars)
    for i in range(num_constraints):
        basic_var_index = np.where(tableau[i, :-1] == 1)[0]
        if len(basic_var_index) == 1 and basic_var_index[0] < num_total_vars:
            solution[basic_var_index[0]] = tableau[i, -1]

    optimal_value = tableau[-1, -1]
    return solution[:num_variables], optimal_value


def solve_simplex(request):
    if request.method == "POST":
        mode = request.POST.get("mode")
        objective = request.POST.get("objective")
        constraints_input = request.POST.getlist("constraints[]")

        try:
            c1, c2 = parse_objective(objective)
            c = np.array([c1, c2])  
        except ValueError:
            return render(request, "simplex_lp.html", {"error_message": "Invalid objective function format."})

        A = []
        b = []
        for eq in constraints_input:
            parsed = parse_equation(eq)
            if parsed:
                a, b_coeff, c_val, op = parsed
                if op == "<=":
                    A.append([a, b_coeff])
                    b.append(c_val)
                elif op == ">=":
                    A.append([-a, -b_coeff])  
                    b.append(-c_val)
                else:
                    return render(request, "simplex_lp.html", {"error_message": "Equality constraints are not supported."})
            else:
                return render(request, "simplex_lp.html", {"error_message": f"Invalid constraint: {eq}"})

        A = np.array(A)
        b = np.array(b)

        try:
            solution, optimal_value = simplex(c, A, b)

            if mode == "minimize":
                optimal_value = -optimal_value

            return render(
                request,
                "simplex_lp.html",
                {
                    "solution": {
                        "constraints": constraints_input,
                        "optimal_point": solution.tolist(),
                        "optimal_value": optimal_value,
                        "objective_function": objective,
                        "mode": mode,
                        "mode_text": "Maximize" if mode == "maximize" else "Minimize",
                    }
                },
            )
        except ValueError as e:
            return render(request, "simplex_lp.html", {"error_message": str(e)})

    return render(request, "simplex_lp.html")


# Transportation problem solver

def solve_transportation_problem(cost_matrix, supply, demand):
    """
    Solves the transportation problem using linear programming.

    Parameters:
        cost_matrix (2D list or numpy array): Cost matrix (m x n) where m is the number of sources and n is the number of destinations.
        supply (list): List of supply capacities for each source.
        demand (list): List of demand requirements for each destination.

    Returns:
        result: Dictionary with the solution, total cost, and status.
    """
    debug_info = [] 
    if sum(supply) != sum(demand):
        debug_info.append("Supply and demand are not balanced.")
        return {"solution": None, "total_cost": None, "status": "Supply and demand must be balanced.", "debug_info": debug_info}

    cost_matrix = np.array(cost_matrix)
    supply = np.array(supply)
    demand = np.array(demand)
    m, n = cost_matrix.shape

    c = cost_matrix.flatten()

    A_eq = []
    b_eq = []

    for i in range(m):
        row_constraint = [0] * (m * n)
        for j in range(n):
            row_constraint[i * n + j] = 1
        A_eq.append(row_constraint)
        b_eq.append(supply[i])

    for j in range(n):
        col_constraint = [0] * (m * n)
        for i in range(m):
            col_constraint[i * n + j] = 1
        A_eq.append(col_constraint)
        b_eq.append(demand[j])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if result.success:
        solution_matrix = result.x.reshape(m, n)
        total_cost = result.fun
        debug_info.append(f"Optimal transportation plan found with total cost: {total_cost}")
        return {
            "solution": solution_matrix.tolist(),
            "total_cost": total_cost,
            "status": "Optimal solution found",
            "debug_info": debug_info
        }
    else:
        debug_info.append(f"Error: {result.message}")
        return {
            "solution": None,
            "total_cost": None,
            "status": f"Error: {result.message}",
            "debug_info": debug_info
        }

def transportation_view(request):
    if request.method == "POST":
        try:
            num_sources = int(request.POST.get("num_sources", 0))
            num_destinations = int(request.POST.get("num_destinations", 0))

            if num_sources <= 0 or num_destinations <= 0:
                return render(request, "transportation_lp.html", {
                    'error_message': "Number of sources and destinations must be greater than zero.",
                    'num_sources': num_sources,
                    'num_destinations': num_destinations
                })

            cost_matrix = []
            for i in range(num_sources):
                row = [float(request.POST.get(f"cost_matrix_{i}_{j}", 0)) for j in range(num_destinations)]
                cost_matrix.append(row)

            supply = [int(request.POST.get(f"supply_{i}", 0)) for i in range(num_sources)]
            demand = [int(request.POST.get(f"demand_{i}", 0)) for i in range(num_destinations)]

            result = solve_transportation_problem(cost_matrix, supply, demand)

            if result["solution"] is None:
                return render(request, "transportation_lp.html", {
                    'error_message': f"Error: {result['status']}",
                    'num_sources': num_sources,
                    'num_destinations': num_destinations,
                    'cost_matrix': cost_matrix,
                    'supply': supply,
                    'demand': demand
                })

            return render(request, "transportation_solution.html", {
                "solution": result["solution"],
                "total_cost": result["total_cost"],
                "status": result["status"],
                "debug_info": result["debug_info"]
            })

        except Exception as e:
            return render(request, "transportation_lp.html", {
                'error_message': f"Error: {str(e)}"
            })

    return render(request, "transportation_lp.html")
