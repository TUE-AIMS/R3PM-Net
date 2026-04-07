import numpy as np
from tabulate import tabulate

def print_table(reports):
    # Prepare data for the table
    table = []
    for method, report in reports.items():
        row = [
            method,
            # report['mean_rmse'],
            report['mean_rotation_error'],
            report['mean_translation_error'],
            report['mean_cd'],
            # report['mean_error'],
            report['mean_fitness'],
            report['mean_inlier_rmse'],
            report['mean_computation_time']
        ]
        table.append(row)

    # headers for the table
    # NOTE: report['mean_fitness'] is used as registration recall (success rate).
    # headers = ['Method', 'RMSE', 'RE', 'TE', 'Time', 'CD', 'Res. Err.', 'Reg. Recall', 'Inlier RMSE']
    headers = ['Method', 'RRE', 'RTE', 'CD', 'Fitness', 'Inlier RMSE', 'Time']

    print(tabulate(table, headers=headers, tablefmt='grid'))

def print_table_no_gt_info(reports):
    # Prepare data for the table
    table = []
    for method, report in reports.items():
        row = [
            method,
            report['mean_cd'],
            report['mean_fitness'],
            report['mean_inlier_rmse'],
            report['mean_computation_time']
        ]
        table.append(row)

    # headers for the table
    # NOTE: report['mean_fitness'] is used as registration recall (success rate).
    headers = ['Method','CD', 'Fitness', 'Inlier RMSE', 'Time']

    print(tabulate(table, headers=headers, tablefmt='grid'))