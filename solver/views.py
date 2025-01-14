from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
import os
import re

def parse_equation(equation):
    """
    Parses a linear equation of the form 'ax + by <= c' or 'ax + by >= c'
    into coefficients a, b, and c and the operator.
    """
    equation = equation.replace(" ", "")
    if "<=" in equation:
        lhs, rhs = equation.split("<=")
        inequality = "<="
    elif ">=" in equation:
        lhs, rhs = equation.split(">=")
        inequality = ">="
    else:
        raise ValueError("Invalid equation format. Use '<=' or '>='.")

    # Default coefficients for x and y
    a, b = 0, 0

    # Regular expression to match the equation form 'ax + by'
    match = re.match(r"([+-]?\d*)x([+-]?\d*)y", lhs)

    if match:
        a_str, b_str = match.groups()

        a = int(a_str) if a_str not in ["", "+", "-"] else (1 if a_str in ["", "+"] else -1)
        b = int(b_str) if b_str not in ["", "+", "-"] else (1 if b_str in ["", "+"] else -1)
    else:
        if "x" in lhs:
            a = int(lhs.replace("x", "1").replace("+", ""))
        if "y" in lhs:
            b = int(lhs.replace("y", "1").replace("+", ""))

    c = float(rhs)

    return a, b, c, inequality

def find_intersection(eq1, eq2):
    """
    Given two equations, return the intersection point (x, y) if it exists.
    """
    x, y = symbols('x y')
    eq1 = Eq(eq1[0]*x + eq1[1]*y, eq1[2])
    eq2 = Eq(eq2[0]*x + eq2[1]*y, eq2[2])
    solution = solve((eq1, eq2), (x, y))

    # Ensure we return the solution as a tuple of (x, y) values
    if solution:
        x_val = float(solution[x]) if x in solution else None
        y_val = float(solution[y]) if y in solution else None
        return x_val, y_val
    return None

def is_feasible(point, constraints):
    """
    Check if a given point (x, y) satisfies all constraints.
    """
    x, y = point
    for a, b, c, op in constraints:
        if op == "<=" and (a * x + b * y > c):
            return False
        elif op == ">=" and (a * x + b * y < c):
            return False
    return True
def solve_lp(request):
    if request.method == "POST":
        mode = request.POST.get("mode")
        objective = request.POST.get("objective")
        constraints_raw = request.POST.getlist("constraints[]")

        try:
            c1, c2 = [float(x) for x in objective.replace("x", "").replace("y", "").split("+")]
        except ValueError:
            return HttpResponse("Invalid objective function format.")

        constraints = []
        for eq in constraints_raw:
            try:
                constraints.append(parse_equation(eq))
            except ValueError as e:
                return HttpResponse(str(e))

        x = np.linspace(-10, 20, 400)
        plt.figure(figsize=(8, 6))

        # Plot constraints and shade feasible region
        for a, b, c, op in constraints:
            if b != 0:
                y = (c - a * x) / b
                if op == "<=":
                    plt.fill_between(x, 0, y, where=(y >= 0), color='green', alpha=0.3, label=f"{a}x + {b}y <= {c}")
                elif op == ">=":
                    plt.fill_between(x, y, 20, where=(y <= 20), color='green', alpha=0.3, label=f"{a}x + {b}y >= {c}")

        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title("Feasible Region")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        # Find corner points
        corner_points = []
        for i, eq1 in enumerate(constraints):
            for j, eq2 in enumerate(constraints):
                if i < j:
                    solution = find_intersection(eq1, eq2)
                    if solution and is_feasible(solution, constraints):
                        corner_points.append(solution)

        # Highlight all corner points on the graph
        for point in corner_points:
            plt.scatter(point[0], point[1], color='blue', zorder=5, label=f"Corner Point ({point[0]:.2f}, {point[1]:.2f})")

        # Find optimal value
        optimal_value = None
        optimal_point = None
        for point in corner_points:
            obj_value = c1 * point[0] + c2 * point[1]
            if optimal_value is None or (mode == "maximize" and obj_value > optimal_value) or (mode == "minimize" and obj_value < optimal_value):
                optimal_value = obj_value
                optimal_point = point

        # Highlight the optimal point on the graph
        if optimal_point:
            plt.scatter(optimal_point[0], optimal_point[1], color='red', zorder=6, label="Optimal Point")
            plt.text(optimal_point[0], optimal_point[1], f"Optimal: {optimal_value:.2f}", fontsize=12, color="red")

        plt.legend()

        # Save the graph
        graph_path = os.path.join('static', 'graphs', 'lp_solution.png')
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        plt.savefig(graph_path)
        plt.close()

        return render(request, 'index.html', {
            'solution': {
                'constraints': constraints_raw,
                'corner_points': corner_points,
                'optimal_point': optimal_point,
                'optimal_value': optimal_value,
                'graph_url': f"/static/graphs/lp_solution.png",
            }
        })
    return render(request, 'index.html')

def index(request):
    return render(request, 'index.html')