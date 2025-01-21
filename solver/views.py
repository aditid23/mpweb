from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import re
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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
            return render(request, "index.html", {"error_message": "Invalid objective function format."})

        constraints = []
        for eq in constraints_input:
            parsed = parse_equation(eq)
            if parsed:
                constraints.append(parsed)
            else:
                return render(request, "index.html", {"error_message": f"Invalid constraint: {eq}"})

        corner_points = []
        for i in range(len(constraints)):
            for j in range(i + 1, len(constraints)):
                point = find_intersection(constraints[i], constraints[j])
                if point and is_feasible(point, constraints):
                    corner_points.append(point)

        if not corner_points:
            return render(request, "index.html", {"error_message": "No feasible region found."})

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
            "index.html",
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
    return render(request, "index.html")


def index(request):
    return render(request, "index.html")