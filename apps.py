import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

def formulate_and_solve_lp_scores_data(y, x, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb=0, ub=1, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))

    if lagrangian_multiplier > 0:
        # Constraints penalization terms
        constraint_terms = gp.QuadExpr(0)

        # Constraint 1: predictions[i] <= 110 * binary_vars[i]
        for i in range(num_targets):
            constraint_terms.add(predictions[i] - 110 * binary_vars[i])

        # Constraint 2: predictions[1] >= 50 * binary_vars[2]
        constraint_terms.add(50 * binary_vars[2] - predictions[1])

        # Constraint 3: predictions[1] + predictions[2] >= 110 * binary_vars[0]
        constraint_terms.add(110 * binary_vars[0] - predictions[1] - predictions[2])

        # Add Lagrangian relaxation term to the objective function
        sse += lagrangian_multiplier * constraint_terms * num_instances
    else:
        # Create constraints
        for i in range(num_targets):
            model.addConstr(predictions[i] <= 1.1 * binary_vars[i], f"z_relationship_{i}")

        model.addConstr(predictions[1] >= .50 * binary_vars[2], "y_constraint_1")
        model.addConstr(predictions[1] + predictions[2] >= 1.1 * binary_vars[0], "y_constraint_2")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}, Indicator = {binary_vars[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


def formulate_and_solve_lp_forecasting_data(y, x, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create a new model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Decision Variables
    predictions = model.addVars(num_targets, lb=0, ub=100, name="predictions")
    indicator_70 = model.addVars(num_targets-1, vtype=GRB.BINARY, name=f"indicator_70")
    indicator_120 = model.addVars(num_targets-1, vtype=GRB.BINARY, name=f"indicator_120")

    # Objective
    # Minimize the sum of squared errors (SSE) between predictions and actual values
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))
    model.setObjective(sse, GRB.MINIMIZE)

    # Constraints
    X_df = pd.DataFrame(x[:, -3:])
    if len(X_df.loc[X_df.sum(axis=1) > 120]):
        model.addConstr(predictions[0] <= 10)
    if len(X_df[X_df.loc[:, [1, 2]].sum(axis=1) > 70]):
        model.addConstr(predictions[0] <= 50)

    # Constraint: If sum >= 70, then next prediction should be less than 50
    model.addConstr(predictions[0] + max(X_df[2]) >= 70 * indicator_70[0])
    model.addConstr(predictions[0] <= 50 + 50 * (1 - indicator_70[0]))

    # Constraint: If sum >= 120, then next prediction should be less than 10
    model.addConstr(predictions[2] + predictions[1] + max(X_df[2]) >= 120 * indicator_120[0])
    model.addConstr(predictions[2] <= 10 + 90 * (1 - indicator_120[0]))

    model.addConstr(predictions[1] + X_df.loc[:, 1:2].sum(axis=1).max() >= 120 * indicator_120[1])
    model.addConstr(predictions[1] <= 10 + 90 * (1 - indicator_120[1]))

    for i in range(2, num_targets):
        # Constraint: If sum >= 70, then next prediction should be less than 50
        model.addConstr(predictions[i - 1] + predictions[i - 2] >= 70 * indicator_70[i-1])
        model.addConstr(predictions[i] <= 50 + 50 * (1 - indicator_70[i-1]))

        if i > 2:
            # Constraint: If sum >= 120, then next prediction should be less than 10
            model.addConstr(predictions[i - 1] + predictions[i - 2] + predictions[i - 3] >= 120 * indicator_120[i-1])
            model.addConstr(predictions[i] <= 10 + 90 * (1 - indicator_120[i-1]))

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    # Optimize the model
    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


def formulate_and_solve_lp_class_data(y, x, initial_solution, lagrangian_multiplier=0, verbose=False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb=0, ub=100, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))

    if lagrangian_multiplier > 0:
        # Constraints penalization terms
        constraint_terms = gp.QuadExpr(0)

        # One prediction constraint
        constraint_terms.add(binary_vars.sum() - 1)

        # Constraint 1: predictions[i] <= 110 * binary_vars[i]
        for i in range(num_targets):
            constraint_terms.add(predictions[i] - 110 * binary_vars[i])

        # Add Lagrangian relaxation term to the objective function
        sse += lagrangian_multiplier * constraint_terms * num_instances
    else:
        # Create constraints
        model.addConstr(binary_vars.sum() <= 1, "one_prediction_constraint")
        for i in range(num_targets):
            model.addConstr(predictions[i] <= 1.1 * binary_vars[i], f"z_relationship_{i}")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}, Indicator = {binary_vars[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


def formulate_and_solve_lp_cars_data(y, x, initial_solution, lagrangian_multiplier=0, verbose=False, bigM=100000):
    # Create a new Gurobi model
    model = gp.Model("Binary and Continuous Variables")
    if not verbose:
        model.Params.LogToConsole = 0

    # Define variables
    predictions = model.addVars(1, lb=0, name="y")  # Continuous variables
    binary_vars = model.addVars(1, vtype=GRB.BINARY, name="z")  # Binary indicators

    # Objective: Minimize Sum of Squared Errors (SSE)
    sse = gp.quicksum((predictions[0] - y[i][1]) * (predictions[0] - y[i][1]) for i in range(y.shape[0]))

    if lagrangian_multiplier > 0:
        # Constraint penalization terms
        constraint_terms = gp.QuadExpr(0)

        # Constraint 1: predictions[0] >= binary_vars[0]
        constraint_terms.add(predictions[0] - binary_vars[0])

        # Constraint 2: predictions[0] <= bigM * binary_vars[0]
        constraint_terms.add(bigM * binary_vars[0] - predictions[0])

        # Add Lagrangian relaxation term to the objective function
        sse += lagrangian_multiplier * constraint_terms
    else:
        # Constraints
        model.addConstr(predictions[0] >= binary_vars[0], "y_constraint")
        model.addConstr(predictions[0] <= bigM * binary_vars[0], "y_upper_bound_constraint")

    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    # Optimize the model
    model.optimize()

    preds = np.array([binary_vars[0].X, predictions[0].X])

    # Display the results
    if verbose:
        print(f"Optimal Solution: {preds}")
        print(f"Objective (Sum of Squared Errors): {model.objVal}")

    return preds

def formulate_and_solve_lp_synthetic_data(y, x, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb=0, ub=1, name="y")
    # binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))

    model.addConstr(gp.quicksum(predictions[i] for i in range(num_targets)) == num_targets * 0.5, "y_constraint_1")
    model.addConstr(gp.quicksum(predictions[i] for i in range(int(num_targets/2))) <= int(num_targets/2) * 0.4, "y_constraint_2")
    model.addConstr(gp.quicksum(predictions[i] for i in range(int(num_targets/2)+1,num_targets)) <= (int(num_targets/2)+1) * 0.6, "y_constraint_3")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


def formulate_and_solve_lp_synthetic_manifold_data(y, x, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb = 0, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))

    
    j=0
    for i in range(num_targets//2,num_targets,2):
        model.addConstr(predictions[i-1]-predictions[i] == (j+1)/10, "y_dif_constraint_"+str(j))
        j+=1
    
    model.addConstr(gp.quicksum(predictions[i] for i in range((num_targets//2))) == 1, "y_constraint_1")
    

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds

def formulate_and_solve_lp_synthetic_illustrative_data(y, x, w, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, lb = 0, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    # sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))
    sse = gp.quicksum((gp.quicksum(w[j] * (predictions[j] - y[i][j]) for j in range(num_targets))) ** 2 for i in range(num_instances))
        
    j=0
    for i in range(num_targets//2,num_targets,2):
        model.addConstr(predictions[i-1]-predictions[i] == (j+1)/10, "y_dif_constraint_"+str(j))
        j+=1
    
    model.addConstr(gp.quicksum(predictions[i] for i in range((num_targets//2))) == 1, "y_constraint_1")
    
    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)
    # print("Objective (Sum of Squared Errors):", model.objVal/num_instances)
    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds, (model.objVal/num_instances)/num_targets

def formulate_and_solve_lp_hts_data(y, x, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    predictions = model.addVars(num_targets, vtype=GRB.INTEGER, lb = 0, name="y")
    binary_vars = model.addVars(num_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))
    
    model.addConstr(binary_vars.sum() <= 13-9, "sparsity_constraint")
    for i in range(num_targets):
        model.addConstr(predictions[i] <= 100*binary_vars[i], f"max_demand_{i}")
    
    model.addConstr(gp.quicksum(predictions[i] for i in range(num_targets)) == 15, "aggregation_constraint")
    

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds

def formulate_and_solve_lp_hts_global_data(y, x, initial_solution, lagrangian_multiplier = 0, verbose = False):
    num_instances = y.shape[0]
    num_targets = y.shape[1]
    num_real_targets = 12
    num_derived_targets = num_targets-num_real_targets

    # Create model
    model = gp.Model("Minimize SSE")
    if not verbose:
        model.Params.LogToConsole = 0

    # Create decision variables
    # predictions = model.addVars(num_targets, lb=-GRB.INFINITY, name="y")
    predictions = model.addVars(num_targets, vtype=GRB.INTEGER, lb = 0, name="y")
    binary_vars = model.addVars(num_real_targets, vtype=GRB.BINARY, name="z")

    # Create objective function
    sse = gp.quicksum((predictions[j] - y[i][j]) * (predictions[j] - y[i][j]) for i in range(num_instances) for j in range(num_targets))
    
    model.addConstr(binary_vars.sum() <= 12-6, "sparsity_constraint")
    for i in range(num_real_targets):
        model.addConstr(predictions[i] <= 10*binary_vars[i], f"max_demand_{i}")
    
    model.addConstr(predictions[0] + predictions[1] + predictions[2] == predictions[12], "level_demand_1")
    model.addConstr(predictions[3] + predictions[4] + predictions[5] == predictions[13], "level_demand_2")
    model.addConstr(predictions[6] + predictions[7] + predictions[8] == predictions[14], "level_demand_3")
    model.addConstr(predictions[9] + predictions[10] + predictions[11] == predictions[15], "level_demand_4")
    
    model.addConstr(predictions[12] + predictions[13] + predictions[14] + predictions[15] == predictions[16], "level_demand_5")

    # Solve the problem
    model.setObjective(sse, GRB.MINIMIZE)

    if sum(initial_solution) > 0:
        model.update()
        for idx, v in enumerate([v for v in model.getVars() if v.VarName.startswith('y')]):
            v.Start = initial_solution[idx]

    model.optimize()

    if verbose:
        print("Optimal Solution:")
        for i in range(num_targets):
            print(f"Target {i + 1}: Prediction = {predictions[i].X}")
        print("Objective (Sum of Squared Errors):", model.objVal)

    preds = np.array([predictions[i].X for i in range(num_targets)])

    return preds


